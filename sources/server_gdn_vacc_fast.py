"""Vectorized (chunk-parallel) pure-PyTorch GDN kernels for Vastai SV100.

Replaces the per-token Python loops in gdn_vacc_ref.py with the FLA chunk
algorithm expressed as batched matrix ops, so torch_vacc dispatches a handful
of large kernels instead of thousands of tiny ones. Math is identical to the
Triton FLA ops (chunk_scaled_dot_kkt / solve_tril / wy_fast / chunk_delta_h /
chunk_o), verified numerically against the slow reference.

Shapes (GVA): q,k [B,T,Hg,K]; v [B,T,HV,V]; g,beta [B,T,HV]; state [N,HV,V,K].
Hg = key heads, HV = value heads, rep = HV//Hg.
"""
import torch
import torch.nn.functional as F
import os as _os

# --- TP4 deadlock fix attempt: bypass ODSP fast all-reduce (busy-wait P2P that
# stalls device_synchronize during GDN forward) and force the vccl path. ---
if _os.environ.get("GDN_FORCE_VCCL", "1") == "1":
    try:
        import torch_vacc.vacc as _tv

        def _odsp_disabled(*a, **k):
            raise RuntimeError("ODSP fast all-reduce disabled; force vccl")

        _tv.all_reduce = _odsp_disabled
    except Exception:
        pass

# --- TP4 deadlock fix: device-wide synchronize (vaccrtDeviceSynchronize) waits
# on ALL streams incl. the ODSP/vccl communicator background stream (busy-wait),
# racing across the 4 visible devices. Replace with current-compute-stream sync. ---
if _os.environ.get("GDN_STREAM_SYNC", "1") == "1":
    try:
        import torch_vacc.vacc as _tv2

        def _stream_sync(device=None):
            _tv2.current_stream().synchronize()

        _tv2.synchronize = _stream_sync
        torch.vacc.synchronize = _stream_sync
    except Exception:
        pass

_DBG = _os.environ.get("GDN_DEBUG")
def _sync(n):
    if _DBG:
        with open("/work/gdn_debug.log", "a") as _f:
            _f.write(n + "\n"); _f.flush()
        if _DBG == "1":
            torch.vacc.synchronize()

_EPS = 1e-6
_C = 64  # chunk size, matches FLA


def _cpu(t):
    """Hoist a metadata tensor to a nested Python list ONCE (single device->CPU
    transfer at op entry), so per-sequence loops never trigger a device sync.
    In-loop int()/bool() on device tensors deadlocks VACC TP4 (stream sync while
    TP collectives are pending)."""
    if t is None:
        return None
    if isinstance(t, torch.Tensor):
        return t.detach().to("cpu").tolist()
    return t


def _l2norm(x):
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + _EPS)


def l2norm_fwd_fast(x, eps=_EPS, output_dtype=None):
    shape = x.shape
    xf = x.reshape(-1, shape[-1]).float()
    y = xf / torch.sqrt((xf * xf).sum(-1, keepdim=True) + eps)
    y = y.to(output_dtype if output_dtype is not None else x.dtype)
    return y.reshape(shape)


def fused_gdn_gating_fast(A_log, a, b, dt_bias, beta=1.0, threshold=20.0):
    _sync("gate_enter")
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias.float(), beta=beta, threshold=threshold)
    _sync("gate_g")
    beta_out = b.sigmoid()
    _sync("gate_beta")
    return g.unsqueeze(0), beta_out.unsqueeze(0)


# ---- unit-lower-triangular inverse: (I + L)^-1 for strictly-lower L ----
_INV_MODE = None


def _inv_unit_lower(L):
    """Return (I + L)^-1 where L is strictly lower triangular [..., C, C].

    Newton-Schulz iteration X <- X(2I - M X), M = I+L. For nilpotent L this
    yields sum_{i=0}^{2^k-1} (-L)^i exactly after k steps; ceil(log2(C)) steps
    suffice. Pure batched matmul (no solve_triangular dependency on VACC)."""
    C = L.shape[-1]
    _sync("inv_enter")
    # Build identity ON DEVICE via arange+eq (avoids vastai::EyeMOut, which does
    # a host->device vacc_copy_from + vaccrtStreamSynchronize that deadlocks TP4).
    idx = torch.arange(C, device=L.device)
    eye2 = (idx[:, None] == idx[None, :]).to(L.dtype)
    M = L + eye2
    X = eye2.expand_as(L)  # broadcast view, no materialization
    _sync("inv_eye")
    steps = max(1, (C - 1).bit_length())
    for _i in range(steps):
        MX = torch.einsum('...ij,...jk->...ik', M, X)
        X = torch.einsum('...ij,...jk->...ik', X, 2 * eye2 - MX)
        _sync(f"inv_ns{_i}")
    return X


def _bmm(a, b):
    # batched matmul via einsum (ref-proven VACC-TP-safe path; torch.bmm deadlocks in TP4)
    return torch.einsum('...ij,...jk->...ik', a, b)


def _chunk_core(q, k, v, g, beta, scale, initial_state, output_final_state, l2norm):
    """Equal-length batched chunk forward. q,k [B,T,Hg,K]; v [B,T,HV,V];
    g,beta [B,T,HV]; initial_state [B,HV,V,K] or None. Returns o [B,T,HV,V],
    final_state [B,HV,V,K]."""
    B, T, Hg, K = q.shape
    HV, V = v.shape[-2], v.shape[-1]
    rep = HV // Hg
    dev = q.device
    if _DBG:
        with open("/work/gdn_debug.log", "a") as _f:
            _f.write(f"SHAPES B={B} T={T} Hg={Hg} K={K} HV={HV} V={V} rep={rep} l2={l2norm}\n"); _f.flush()
    _sync("core_enter")
    h_map = torch.arange(HV, device=dev) // rep
    q = q[:, :, h_map, :].float()
    k = k[:, :, h_map, :].float()
    _sync("core_gather")
    v = v.float()
    g = g.float()
    beta = beta.float()
    if l2norm:
        q = _l2norm(q)
        k = _l2norm(k)
    q = q * scale
    _sync("core_l2")

    C = _C
    NT = (T + C - 1) // C
    Tp = NT * C
    pad = Tp - T
    if pad:
        q = F.pad(q, (0, 0, 0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, 0, 0, pad))
        g = F.pad(g, (0, 0, 0, pad))
        beta = F.pad(beta, (0, 0, 0, pad))
    _sync("core_pad")
    # [B,NT,HV,C,*]
    qc = q.reshape(B, NT, C, HV, K).permute(0, 1, 3, 2, 4).contiguous()
    kc = k.reshape(B, NT, C, HV, K).permute(0, 1, 3, 2, 4).contiguous()
    vc = v.reshape(B, NT, C, HV, V).permute(0, 1, 3, 2, 4).contiguous()
    gc = g.reshape(B, NT, C, HV).permute(0, 1, 3, 2).contiguous()
    bc = beta.reshape(B, NT, C, HV).permute(0, 1, 3, 2).contiguous()
    _sync("core_reshape")
    g_cs = gc.cumsum(dim=-1)  # [B,NT,HV,C]
    _sync("core_cumsum")
    eg = g_cs.exp()
    _sync("core_eg")

    # L[i,j] = beta_i (k_i.k_j) exp(g_cs_i - g_cs_j), j<i
    _sync("core_pre_kkt")
    kkt = _bmm(kc, kc.transpose(-1, -2))  # [B,NT,HV,C,C]
    _sync("core_kkt")
    decay = (g_cs[..., :, None] - g_cs[..., None, :]).exp()
    _sync("core_decay")
    L = torch.tril(kkt * decay * bc[..., :, None], diagonal=-1)
    _sync("core_tril")
    A = _inv_unit_lower(L)  # (I+L)^-1
    _sync("core_inv")

    # w = A @ (k*beta*exp(g_cs)); u = A @ (v*beta)
    w = _bmm(A, kc * bc[..., None] * eg[..., None])  # [B,NT,HV,C,K]
    _sync("core_w")
    u = _bmm(A, vc * bc[..., None])                  # [B,NT,HV,C,V]
    _sync("core_u")

    # inter-chunk state recurrence
    if initial_state is not None:
        h = initial_state.float().clone()
    else:
        h = torch.zeros(B, HV, V, K, device=dev)
    h_start = torch.empty(B, NT, HV, V, K, device=dev)
    v_new = torch.empty(B, NT, HV, C, V, device=dev)
    for n in range(NT):
        h_start[:, n] = h
        wn = w[:, n]                       # [B,HV,C,K]
        un = u[:, n]                       # [B,HV,C,V]
        vn = un - _bmm(wn, h.transpose(-1, -2))  # u - w@h^T
        v_new[:, n] = vn
        g_last = g_cs[:, n, :, -1]         # [B,HV]
        gn = g_cs[:, n]                    # [B,HV,C]
        vn_dec = vn * (g_last[..., None] - gn).exp()[..., None]
        kn = kc[:, n]                      # [B,HV,C,K]
        h = h * g_last.exp()[..., None, None] + _bmm(vn_dec.transpose(-1, -2), kn)
    final_state = h
    _sync("core_hloop")

    # output: o = (q*exp(g_cs))@h_start^T + tril(q@k^T*decay,0)@v_new
    o_inter = _bmm(qc * eg[..., None], h_start.transpose(-1, -2))  # [B,NT,HV,C,V]
    _sync("core_ointer")
    qk = _bmm(qc, kc.transpose(-1, -2))
    A_intra = torch.tril(qk * decay, diagonal=0)
    o_intra = _bmm(A_intra, v_new)
    _sync("core_ointra")
    o = (o_inter + o_intra).permute(0, 1, 3, 2, 4).reshape(B, Tp, HV, V)[:, :T]
    _sync("core_ofinal")
    return o, final_state


def chunk_gated_delta_rule_fast(q, k, v, g, beta, scale=None, initial_state=None,
                                output_final_state=False, cu_seqlens=None,
                                use_qk_l2norm_in_kernel=False):
    if scale is None:
        scale = q.shape[-1] ** -0.5
    dt = q.dtype
    Hg, HV = q.shape[-2], v.shape[-2]
    # g,beta may carry Hg or HV heads; normalize to HV
    if g.shape[-1] == Hg and Hg != HV:
        rep = HV // Hg
        hm = torch.arange(HV, device=q.device) // rep
        g = g[..., hm]
        beta = beta[..., hm]
    if cu_seqlens is not None:
        assert q.shape[0] == 1
        q, k, v, g, beta = q[0], k[0], v[0], g[0], beta[0]
        cu = _cpu(cu_seqlens)
        N = len(cu) - 1
        Ttot = q.shape[0]
        o = torch.zeros(Ttot, HV, v.shape[-1], device=q.device, dtype=torch.float32)
        fs = torch.zeros(N, HV, v.shape[-1], q.shape[-1], device=q.device) if output_final_state else None
        for n in range(N):
            bos, eos = cu[n], cu[n + 1]
            if eos <= bos:
                # empty sequence: preserve incoming state instead of zeroing
                if output_final_state and initial_state is not None:
                    fs[n] = initial_state[n].float()
                continue
            s0 = initial_state[n:n + 1].float() if initial_state is not None else None
            on, sn = _chunk_core(q[bos:eos].unsqueeze(0), k[bos:eos].unsqueeze(0),
                                 v[bos:eos].unsqueeze(0), g[bos:eos].unsqueeze(0),
                                 beta[bos:eos].unsqueeze(0), scale, s0,
                                 output_final_state, use_qk_l2norm_in_kernel)
            o[bos:eos] = on[0]
            if output_final_state:
                fs[n] = sn[0]
        return o.unsqueeze(0).to(dt), (fs.to(dt) if fs is not None else None)
    o, fs = _chunk_core(q, k, v, g, beta, scale, initial_state,
                        output_final_state, use_qk_l2norm_in_kernel)
    return o.to(dt), (fs.to(dt) if output_final_state else None)


def fused_recurrent_gated_delta_rule_fast(q, k, v, g, beta=None, scale=None,
                                          initial_state=None, inplace_final_state=True,
                                          cu_seqlens=None, ssm_state_indices=None,
                                          num_accepted_tokens=None,
                                          use_qk_l2norm_in_kernel=False):
    # decode: T small (often 1). Batched over sequences, loop over T only.
    if scale is None:
        scale = q.shape[-1] ** -0.5
    dt = q.dtype
    Hg, K = q.shape[-2], q.shape[-1]
    HV, V = v.shape[-2], v.shape[-1]
    rep = HV // Hg
    dev = q.device
    h_map = torch.arange(HV, device=dev) // rep
    q = q.float(); k = k.float(); v = v.float(); g = g.float()
    beta = beta.float() if beta is not None else torch.ones_like(g)
    if g.shape[-1] == Hg and Hg != HV:
        g = g[..., h_map]; beta = beta[..., h_map]
    if use_qk_l2norm_in_kernel:
        q = _l2norm(q); k = _l2norm(k)
    q = q * scale
    qe = q[..., h_map, :] if q.shape[-2] == Hg else q
    ke = k[..., h_map, :] if k.shape[-2] == Hg else k

    cu = _cpu(cu_seqlens)
    ssi = _cpu(ssm_state_indices)
    nat = _cpu(num_accepted_tokens)

    def state_slot(n, t_off):
        if ssi is None:
            return n
        if nat is not None:
            return ssi[n][nat[n] - 1]
        return ssi[n][t_off]

    if cu is not None:
        assert q.shape[0] == 1
        qe, ke, vv, gg, bb = qe[0], ke[0], v[0], g[0], beta[0]
        N = len(cu) - 1
        Ttot = qe.shape[0]
        o = torch.zeros(Ttot, HV, V, device=dev)
        for n in range(N):
            bos, eos = cu[n], cu[n + 1]
            if eos <= bos:
                continue
            si = state_slot(n, 0)
            if si < 0:
                continue
            S = initial_state[si].float().clone() if initial_state is not None else torch.zeros(HV, V, K, device=dev)
            for j, t in enumerate(range(bos, eos)):
                S = S * gg[t].exp()[..., None, None]
                kv = torch.einsum('hvk,hk->hv', S, ke[t])
                delta = (vv[t] - kv) * bb[t][..., None]
                S = S + torch.einsum('hv,hk->hvk', delta, ke[t])
                o[t] = torch.einsum('hvk,hk->hv', S, qe[t])
                if inplace_final_state and initial_state is not None and ssi is not None:
                    fi = ssi[n][j]
                    if fi >= 0:
                        initial_state[fi] = S.to(initial_state.dtype)
            if inplace_final_state and initial_state is not None and ssi is None:
                initial_state[si] = S.to(initial_state.dtype)
        return o.unsqueeze(0).to(dt), initial_state
    B, T = qe.shape[0], qe.shape[1]
    o = torch.zeros(B, T, HV, V, device=dev)
    # non-varlen batched decode: honor ssm_state_indices slot mapping
    if ssi is not None:
        slots = [state_slot(n, 0) for n in range(B)]
        S = torch.stack([initial_state[s].float() if (initial_state is not None and s >= 0)
                         else torch.zeros(HV, V, K, device=dev) for s in slots])
    else:
        slots = list(range(B))
        S = initial_state.float().clone() if initial_state is not None else torch.zeros(B, HV, V, K, device=dev)
    for t in range(T):
        S = S * g[:, t].exp()[..., None, None]
        kv = torch.einsum('bhvk,bhk->bhv', S, ke[:, t])
        delta = (v[:, t] - kv) * beta[:, t][..., None]
        S = S + torch.einsum('bhv,bhk->bhvk', delta, ke[:, t])
        o[:, t] = torch.einsum('bhvk,bhk->bhv', S, qe[:, t])
    if initial_state is not None:
        if inplace_final_state:
            for n, s in enumerate(slots):
                if s >= 0:
                    initial_state[s] = S[n].to(initial_state.dtype)
            return o.to(dt), initial_state
        return o.to(dt), S.to(dt)
    return o.to(dt), S.to(dt)


def causal_conv1d_fn_fast(x, weight, bias, conv_states, query_start_loc,
                          cache_indices=None, has_initial_state=None,
                          activation="silu", pad_slot_id=-1, **kw):
    dim, total = x.shape
    width = weight.shape[1]
    dt = x.dtype
    xf = x.float(); wf = weight.float()
    out = torch.zeros_like(xf)
    qsl = _cpu(query_start_loc)
    ci = _cpu(cache_indices)
    his = _cpu(has_initial_state)
    N = len(qsl) - 1
    for n in range(N):
        bos, eos = qsl[n], qsl[n + 1]
        if eos <= bos:
            continue
        seq = xf[:, bos:eos]; L = seq.shape[1]
        slot = ci[n] if ci is not None else n
        if slot == pad_slot_id:
            prev = torch.zeros(dim, width - 1, device=x.device)
        elif his is not None and his[n] and conv_states is not None:
            prev = conv_states[slot].float()
        else:
            prev = torch.zeros(dim, width - 1, device=x.device)
        full = torch.cat([prev, seq], dim=1)
        _sync("conv_cat")
        if width > 1:
            # causal conv as sum of width shifted slice-multiplies (no unfold)
            o = torch.zeros_like(seq)
            for _i in range(width):
                o = o + wf[:, _i:_i + 1] * full[:, _i:_i + L]
            _sync("conv_unfold")
        else:
            o = seq * wf[:, 0:1]
        if bias is not None:
            o = o + bias.float().unsqueeze(1)
        if activation in ("silu", "swish"):
            o = F.silu(o)
            _sync("conv_silu")
        out[:, bos:eos] = o
        _sync("conv_write")
        if conv_states is not None and slot != pad_slot_id and width > 1:
            conv_states[slot] = full[:, -(width - 1):].to(conv_states.dtype)
    return out.to(dt)


def causal_conv1d_update_fast(x, conv_state, weight, bias=None, activation=None,
                              conv_state_indices=None, num_accepted_tokens=None,
                              query_start_loc=None, max_query_len=-1, pad_slot_id=-1,
                              **kw):
    dt = x.dtype
    wf = weight.float(); width = weight.shape[1]
    if x.dim() == 3:
        b, d, sl = x.shape
        outs = [causal_conv1d_update_fast(x[:, :, t], conv_state, weight, bias, activation,
                conv_state_indices, None, None, -1, pad_slot_id) for t in range(sl)]
        return torch.stack(outs, dim=-1)
    xf = x.float()
    if xf.dim() == 1:
        xf = xf.unsqueeze(0)
    nt, dim = xf.shape
    out = torch.zeros(nt, dim, device=x.device)
    idx = conv_state_indices if conv_state_indices is not None else torch.arange(nt, device=x.device)
    idx = idx.long()
    valid = idx != pad_slot_id
    vi = idx[valid]
    if vi.numel() == 0:
        return out.to(dt)
    st = conv_state[vi].float()                       # [m,dim,width-1]
    full = torch.cat([st, xf[valid].unsqueeze(-1)], dim=-1)  # [m,dim,width]
    o = (full * wf.unsqueeze(0)).sum(-1)              # [m,dim]
    if bias is not None:
        o = o + bias.float().unsqueeze(0)
    if activation in ("silu", "swish", True):
        o = F.silu(o)
    out[valid] = o
    conv_state[vi] = full[:, :, 1:].to(conv_state.dtype)  # keep last width-1
    return out.to(dt)