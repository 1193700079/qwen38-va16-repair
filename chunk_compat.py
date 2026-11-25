"""Self-contained prefill chunk GDN compatibility module for Qwen3.8 / Qwen3-Next.

Clean replacement for the chunk path in server_gdn_vacc_fast.py. Key changes:
  * STABLE unit-lower-triangular inverse via block-16 forward substitution with a
    solve-free row recurrence (no flat Newton-Schulz). The flat NS iteration
    X <- X(2I - MX) suffers catastrophic cancellation for aligned / alternating
    l2norm keys at weak decay / high beta; block forward substitution only ever
    materializes bounded inverse entries, so it stays accurate in fp32.
  * Uses only basic matmul (einsum) + indexing + clamp, the VACC-safe op set;
    identity is built device-side via arange+eq (avoids vastai::EyeMOut host copy).
  * Decay exp argument is clamped to <= 0 so the masked upper triangle never
    evaluates a positive exp (production g <= 0).
  * Returned final_state is fp32 (FLA contract); output o carries q.dtype.
  * Empty varlen sequences preserve the incoming initial state.
  * GVA: g/beta carrying Hg heads are expanded to HV via h//rep.

No imports from the shared fast/ref modules; no monkeypatching. CPU-verified
against an independent fp64 sequential oracle; NOT hardware-validated on VACC.
"""
import torch
import torch.nn.functional as F

_EPS = 1e-6
_C = 64        # chunk size (matches FLA)
_BLOCK = 16    # inversion block size


def _mm(a, b):
    # VACC-safe batched matmul (einsum; the path proven not to deadlock in TP4).
    return torch.einsum('...ij,...jk->...ik', a, b)


def _eye(n, device, dtype):
    # device-side identity via arange+eq (avoids torch.eye host->device copy).
    idx = torch.arange(n, device=device)
    return (idx[:, None] == idx[None, :]).to(dtype)


def _l2norm(x, eps=_EPS):
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + eps)


def _inv_unit_lower_block(D):
    """Inverse of a unit-lower-triangular [..., B, B] block by solve-free row
    recurrence (forward substitution): S[r] = e_r - D[r, :r] @ S[:r, :].
    Intermediates are bounded by the true inverse entries -> fp32-stable."""
    B = D.shape[-1]
    dev, dt = D.device, D.dtype
    eye = _eye(B, dev, dt)
    S = torch.zeros_like(D) + eye          # broadcast add, no expand+contiguous
    for r in range(1, B):
        prod = _mm(D[..., r:r + 1, :r], S[..., :r, :])   # [...,1,B]
        S[..., r:r + 1, :] = eye[r:r + 1, :] - prod
    return S


def _inv_unit_lower(L):
    """Return (I + L)^-1 for strictly-lower-triangular L [..., C, C], C % 16 == 0.

    Block-16 forward substitution:
      A[i,i] = inv(M[i,i])                       (row recurrence, stable)
      A[i,j] = -A[i,i] @ sum_{k=j}^{i-1} M[i,k] @ A[k,j]   (j < i, matmul)
    where M = I + L. Only basic matmul + slicing; no solve_triangular dependency.
    """
    C = L.shape[-1]
    nb = C // _BLOCK
    dev, dt = L.device, L.dtype
    M = L + _eye(C, dev, dt)
    A = torch.zeros_like(L)
    # diagonal blocks (independent, each inverted by stable row recurrence)
    for i in range(nb):
        sl = slice(i * _BLOCK, (i + 1) * _BLOCK)
        A[..., sl, sl] = _inv_unit_lower_block(M[..., sl, sl])
    # strictly-lower off-diagonal blocks
    for i in range(nb):
        si = slice(i * _BLOCK, (i + 1) * _BLOCK)
        for j in range(i):
            sj = slice(j * _BLOCK, (j + 1) * _BLOCK)
            acc = torch.zeros_like(A[..., si, sj])
            for kk in range(j, i):
                sk = slice(kk * _BLOCK, (kk + 1) * _BLOCK)
                acc = acc + _mm(M[..., si, sk], A[..., sk, sj])
            A[..., si, sj] = -_mm(A[..., si, si], acc)
    return A


def _chunk_core(q, k, v, g, beta, scale, initial_state, l2norm):
    """Equal-length batched chunk forward. q,k [B,T,Hg,K]; v [B,T,HV,V];
    g,beta [B,T,HV]; initial_state [B,HV,V,K] or None.
    Returns o [B,T,HV,V] (fp32), final_state [B,HV,V,K] (fp32)."""
    B, T, Hg, K = q.shape
    HV, V = v.shape[-2], v.shape[-1]
    rep = HV // Hg
    dev = q.device
    hm = torch.arange(HV, device=dev) // rep
    q = q[:, :, hm, :].float()
    k = k[:, :, hm, :].float()
    v = v.float(); g = g.float(); beta = beta.float()
    if l2norm:
        q = _l2norm(q); k = _l2norm(k)
    q = q * scale

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
    qc = q.reshape(B, NT, C, HV, K).permute(0, 1, 3, 2, 4).contiguous()
    kc = k.reshape(B, NT, C, HV, K).permute(0, 1, 3, 2, 4).contiguous()
    vc = v.reshape(B, NT, C, HV, V).permute(0, 1, 3, 2, 4).contiguous()
    gc = g.reshape(B, NT, C, HV).permute(0, 1, 3, 2).contiguous()
    bc = beta.reshape(B, NT, C, HV).permute(0, 1, 3, 2).contiguous()
    g_cs = gc.cumsum(dim=-1)                 # [B,NT,HV,C], <= 0
    eg = g_cs.exp()                          # <= 1

    kkt = _mm(kc, kc.transpose(-1, -2))      # [B,NT,HV,C,C]
    # decay: clamp exp argument to <= 0 so the masked upper triangle never
    # evaluates a positive exp (avoids overflow / 0*inf NaN on VACC tril).
    diff = (g_cs[..., :, None] - g_cs[..., None, :]).clamp(max=0.0)
    decay = diff.exp()                       # <= 1 everywhere
    L = torch.tril(kkt * decay * bc[..., :, None], diagonal=-1)
    A = _inv_unit_lower(L)                   # stable (I+L)^-1

    w = _mm(A, kc * bc[..., None] * eg[..., None])   # [B,NT,HV,C,K]
    u = _mm(A, vc * bc[..., None])                   # [B,NT,HV,C,V]

    if initial_state is not None:
        h = initial_state.float().clone()
    else:
        h = torch.zeros(B, HV, V, K, device=dev, dtype=torch.float32)
    h_start = torch.empty(B, NT, HV, V, K, device=dev, dtype=torch.float32)
    v_new = torch.empty(B, NT, HV, C, V, device=dev, dtype=torch.float32)
    for n in range(NT):
        h_start[:, n] = h
        wn = w[:, n]; un = u[:, n]
        vn = un - _mm(wn, h.transpose(-1, -2))
        v_new[:, n] = vn
        g_last = g_cs[:, n, :, -1]
        gn = g_cs[:, n]
        vn_dec = vn * (g_last[..., None] - gn).exp()[..., None]   # arg <= 0
        kn = kc[:, n]
        h = h * g_last.exp()[..., None, None] + _mm(vn_dec.transpose(-1, -2), kn)
    final_state = h                          # fp32

    o_inter = _mm(qc * eg[..., None], h_start.transpose(-1, -2))
    qk = _mm(qc, kc.transpose(-1, -2))
    A_intra = torch.tril(qk * decay, diagonal=0)
    o_intra = _mm(A_intra, v_new)
    o = (o_inter + o_intra).permute(0, 1, 3, 2, 4).reshape(B, Tp, HV, V)[:, :T]
    return o, final_state


def chunk_gated_delta_rule_compat(q, k, v, g, beta=None, scale=None,
                                  initial_state=None, output_final_state=False,
                                  cu_seqlens=None, use_qk_l2norm_in_kernel=False):
    if scale is None:
        scale = q.shape[-1] ** -0.5
    dt = q.dtype
    Hg, HV = q.shape[-2], v.shape[-2]
    if beta is None:
        beta = torch.ones_like(g)
    # GVA: normalize g/beta to HV heads
    if g.shape[-1] == Hg and Hg != HV:
        rep = HV // Hg
        hm = torch.arange(HV, device=q.device) // rep
        g = g[..., hm]
        beta = beta[..., hm]

    if cu_seqlens is not None:
        assert q.shape[0] == 1
        q0, k0, v0, g0, b0 = q[0], k[0], v[0], g[0], beta[0]
        cu = cu_seqlens.detach().to("cpu").tolist()
        N = len(cu) - 1
        Ttot = q0.shape[0]
        o = torch.zeros(Ttot, HV, v.shape[-1], device=q.device, dtype=torch.float32)
        fs = torch.zeros(N, HV, v.shape[-1], q.shape[-1], device=q.device,
                         dtype=torch.float32) if output_final_state else None
        for n in range(N):
            bos, eos = cu[n], cu[n + 1]
            if eos <= bos:
                # empty sequence: preserve incoming state instead of zeroing
                if output_final_state and initial_state is not None:
                    fs[n] = initial_state[n].float()
                continue
            s0 = initial_state[n:n + 1].float() if initial_state is not None else None
            on, sn = _chunk_core(q0[bos:eos].unsqueeze(0), k0[bos:eos].unsqueeze(0),
                                 v0[bos:eos].unsqueeze(0), g0[bos:eos].unsqueeze(0),
                                 b0[bos:eos].unsqueeze(0), scale, s0, use_qk_l2norm_in_kernel)
            o[bos:eos] = on[0]
            if output_final_state:
                fs[n] = sn[0]
        # output o in q.dtype; final_state stays fp32 (FLA contract)
        return o.unsqueeze(0).to(dt), fs

    o, fs = _chunk_core(q, k, v, g, beta, scale, initial_state, use_qk_l2norm_in_kernel)
    return o.to(dt), (fs if output_final_state else None)