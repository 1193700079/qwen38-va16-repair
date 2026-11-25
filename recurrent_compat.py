"""Drop-in no-spec recurrent GDN replacement for the Qwen3.8 / Qwen3-Next runtime.

Scope (verified contracts only):
  * normal no-spec decode (varlen, one token per sequence) and multi-token varlen;
  * non-varlen batched decode that honors ``ssm_state_indices``;
  * ``ssm_state_indices`` accepted as 1D [N] or 2D [N, 1];
  * fp32 internal work and an fp32 returned final state (FLA contract); in-place
    cache writes use the cache's own dtype;
  * duplicate slot aliases are resolved by a sequential read-modify-write fallback
    (sequence n+1 observes sequence n's write) rather than a parallel race;
  * speculative modes (``num_accepted_tokens`` or multi-column 2D state indices)
    are rejected loudly with NotImplementedError BEFORE any write, instead of
    silently producing wrong values.

This module is self-contained: it does not import or modify the shared
``server_gdn_vacc_fast`` / ``gdn_vacc_ref`` sources. The recurrence is the gated
delta rule:
    S <- exp(g) * S
    S <- S + beta * (v - S k) (x) k
    o  = S q
with q, k l2-normalized (eps=1e-6) and q scaled by ``scale`` when requested.
"""
import torch

_EPS = 1e-6


def _l2norm(x, eps=_EPS):
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + eps)


def _normalize_slots(ssi, n):
    """Map state indices to a length-n list of integer slots.

    Accepts None (identity), 1D [n], or 2D [n, 1]. A 2D tensor with width > 1 is
    the speculative multi-query layout and is rejected (not verified here).
    """
    if ssi is None:
        return list(range(n))
    if isinstance(ssi, torch.Tensor):
        if ssi.dim() == 1:
            if ssi.shape[0] != n:
                raise ValueError(f"ssm_state_indices length {ssi.shape[0]} != {n}")
            return ssi.detach().to("cpu").tolist()
        if ssi.dim() == 2:
            if ssi.shape[1] != 1:
                raise NotImplementedError(
                    "speculative multi-column ssm_state_indices (2D width>1) is not "
                    "verified; refusing rather than producing silent wrong values")
            if ssi.shape[0] != n:
                raise ValueError(f"ssm_state_indices rows {ssi.shape[0]} != {n}")
            return ssi[:, 0].detach().to("cpu").tolist()
        raise NotImplementedError(
            f"ssm_state_indices with {ssi.dim()} dims is not supported")
    # plain python sequence
    seq = list(ssi)
    if len(seq) != n:
        raise ValueError(f"ssm_state_indices length {len(seq)} != {n}")
    out = []
    for item in seq:
        if isinstance(item, (list, tuple)):
            if len(item) != 1:
                raise NotImplementedError(
                    "speculative multi-column ssm_state_indices is not verified")
            out.append(int(item[0]))
        else:
            out.append(int(item))
    return out


def _recurrence(S, qe_t, ke_t, vf_t, gf_t, bf_t):
    """One gated-delta-rule step. S [HV,V,K]; *_t [HV,*]. Returns updated S, o [HV,V]."""
    S = S * gf_t.exp()[..., None, None]
    kv = torch.einsum('hvk,hk->hv', S, ke_t)
    delta = (vf_t - kv) * bf_t[..., None]
    S = S + torch.einsum('hv,hk->hvk', delta, ke_t)
    o = torch.einsum('hvk,hk->hv', S, qe_t)
    return S, o


def fused_recurrent_gated_delta_rule_compat(
    q, k, v, g, beta=None, scale=None, initial_state=None,
    inplace_final_state=True, cu_seqlens=None, ssm_state_indices=None,
    num_accepted_tokens=None, use_qk_l2norm_in_kernel=False,
):
    # ---- reject unverified speculative modes BEFORE any computation/write ----
    if num_accepted_tokens is not None:
        raise NotImplementedError(
            "speculative num_accepted_tokens mode is not verified; refusing rather "
            "than producing silent wrong values")

    if scale is None:
        scale = q.shape[-1] ** -0.5
    Hg, K = q.shape[-2], q.shape[-1]
    HV, V = v.shape[-2], v.shape[-1]
    rep = HV // Hg
    dev = q.device

    varlen = cu_seqlens is not None
    if varlen:
        if q.shape[0] != 1:
            raise ValueError("varlen (cu_seqlens) requires batch size 1")
        cu = cu_seqlens.detach().to("cpu").tolist()
        nseq = len(cu) - 1
    else:
        nseq = q.shape[0]
    # rejects speculative multi-column SSI early (before any write)
    slots = _normalize_slots(ssm_state_indices, nseq)

    # ---- fp32 work ----
    qf = q.float(); kf = k.float(); vf = v.float(); gf = g.float()
    bf = beta.float() if beta is not None else torch.ones_like(gf)
    hm = torch.arange(HV, device=dev) // rep
    if gf.shape[-1] == Hg and Hg != HV:
        gf = gf[..., hm]
        bf = bf[..., hm]
    if use_qk_l2norm_in_kernel:
        qf = _l2norm(qf)
        kf = _l2norm(kf)
    qf = qf * scale
    qe = qf[..., hm, :] if qf.shape[-2] == Hg else qf
    ke = kf[..., hm, :] if kf.shape[-2] == Hg else kf

    def read_state(slot):
        if initial_state is not None and slot >= 0:
            return initial_state[slot].float().clone()
        return torch.zeros(HV, V, K, device=dev, dtype=torch.float32)

    def write_state(slot, S):
        if inplace_final_state and initial_state is not None and slot >= 0:
            initial_state[slot] = S.to(initial_state.dtype)

    if varlen:
        Ttot = qe.shape[1]
        o = torch.zeros(1, Ttot, HV, V, device=dev, dtype=torch.float32)
        finals = torch.zeros(nseq, HV, V, K, device=dev, dtype=torch.float32)
        # sequential per-sequence: duplicate slots chain through the live cache
        for n in range(nseq):
            bos, eos = cu[n], cu[n + 1]
            slot = slots[n]
            if eos <= bos:
                # empty sequence: preserve incoming state, no output
                if initial_state is not None and slot >= 0:
                    finals[n] = initial_state[slot].float()
                continue
            if slot < 0:
                # pad slot: zero output, no state read/write
                continue
            S = read_state(slot)
            for t in range(bos, eos):
                S, o_t = _recurrence(S, qe[0, t], ke[0, t], vf[0, t], gf[0, t], bf[0, t])
                o[0, t] = o_t
            finals[n] = S
            write_state(slot, S)
        return o, finals

    # ---- non-varlen batched decode ----
    B, T = qe.shape[0], qe.shape[1]
    o = torch.zeros(B, T, HV, V, device=dev, dtype=torch.float32)
    finals = torch.zeros(B, HV, V, K, device=dev, dtype=torch.float32)
    for n in range(B):
        slot = slots[n]
        if slot < 0:
            continue
        S = read_state(slot)
        for t in range(T):
            S, o_t = _recurrence(S, qe[n, t], ke[n, t], vf[n, t], gf[n, t], bf[n, t])
            o[n, t] = o_t
        finals[n] = S
        write_state(slot, S)
    return o, finals