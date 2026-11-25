"""Contract regression suite for the no-spec recurrent GDN replacement.

Independent ground truth: a scalar/numpy fp64 recurrence oracle derived from the
gated-delta-rule definition, NOT from gdn_vacc_ref.py or server_gdn_vacc_fast.py.

Gated delta rule (per value head h, state S[h] in R^{V x K}):
    alpha_t = exp(g[t, h])
    S[h]   <- alpha_t * S[h]                      # decay
    kv      = S[h] @ k[t, h//rep]                 # [V]
    delta   = (v[t, h] - kv) * beta[t, h]         # [V]
    S[h]   <- S[h] + outer(delta, k[t, h//rep])   # rank-1 update
    o[t, h] = S[h] @ q[t, h//rep]                 # [V]
q, k are l2-normalized (eps=1e-6) when use_qk_l2norm_in_kernel=True; q is then
scaled by `scale` (default K**-0.5). Work is fp32; the returned final state is
fp32 (FLA contract), and any in-place cache write uses the cache's own dtype.

Run:
    /opt/homebrew/Caskroom/miniconda/base/bin/python3 -m pytest \
        test_recurrent_contracts.py -v
"""
import numpy as np
import pytest
import torch

from recurrent_compat import fused_recurrent_gated_delta_rule_compat

EPS = 1e-6


# --------------------------------------------------------------------------- #
# Independent fp64 oracle (numpy, explicit per-head / per-token recurrence).
# --------------------------------------------------------------------------- #
def oracle_sequence(q, k, v, g, beta, S0, scale, l2norm, eps=EPS):
    """q,k [T,Hg,K]; v [T,HV,V]; g,beta [T,HV]; S0 [HV,V,K] or None.
    Returns o [T,HV,V], S_final [HV,V,K] in fp64."""
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    T, Hg, K = q.shape
    HV, V = v.shape[1], v.shape[2]
    rep = HV // Hg
    if l2norm:
        q = q / np.sqrt((q * q).sum(-1, keepdims=True) + eps)
        k = k / np.sqrt((k * k).sum(-1, keepdims=True) + eps)
    q = q * scale
    S = np.zeros((HV, V, K), np.float64) if S0 is None else np.asarray(S0, np.float64).copy()
    o = np.zeros((T, HV, V), np.float64)
    for t in range(T):
        for h in range(HV):
            hg = h // rep
            S[h] *= np.exp(g[t, h])
            kt = k[t, hg]
            kv = S[h] @ kt
            delta = (v[t, h] - kv) * beta[t, h]
            S[h] += np.outer(delta, kt)
            o[t, h] = S[h] @ q[t, hg]
    return o, S


def _rand_inputs(rng, T, Hg, HV, K, V, dtype=torch.float32):
    """Production-realistic: g <= 0 (decay), beta in (0,1)."""
    q = torch.randn(1, T, Hg, K, dtype=torch.float64).to(dtype)
    k = torch.randn(1, T, Hg, K, dtype=torch.float64).to(dtype)
    v = torch.randn(1, T, HV, V, dtype=torch.float64).to(dtype)
    g = -torch.rand(1, T, HV, dtype=torch.float64).to(dtype) * 2.0  # <= 0
    beta = torch.rand(1, T, HV, dtype=torch.float64).to(dtype) * 0.9 + 0.05
    return q, k, v, g, beta


def _to_np(t):
    return t.detach().to(torch.float64).cpu().numpy()


def _run_compat(q, k, v, g, beta, initial_state, **kw):
    return fused_recurrent_gated_delta_rule_compat(
        q=q, k=k, v=v, g=g, beta=beta, initial_state=initial_state,
        use_qk_l2norm_in_kernel=True, **kw)


# --------------------------------------------------------------------------- #
# 1. Normal no-spec decode: varlen, one token per sequence, 1D SSI, fp32 cache.
# --------------------------------------------------------------------------- #
def test_no_spec_decode_1d_ssi_matches_oracle():
    rng = np.random.default_rng(0)
    N, Hg, HV, K, V = 3, 2, 4, 8, 8
    nslots = 5
    q, k, v, g, beta = _rand_inputs(rng, N, Hg, HV, K, V)  # T=N, one token/seq
    slots = [2, 0, 4]
    cache = torch.randn(nslots, HV, V, K, dtype=torch.float32) * 0.1
    cache0 = cache.clone()
    cu = torch.arange(N + 1, dtype=torch.int32)  # [0,1,2,3]
    ssi = torch.tensor(slots, dtype=torch.int32)
    scale = K ** -0.5

    o, fs = _run_compat(q, k, v, g, beta, cache, cu_seqlens=cu,
                        ssm_state_indices=ssi, inplace_final_state=True)

    assert o.shape == (1, N, HV, V)
    for n in range(N):
        o_ref, S_ref = oracle_sequence(
            _to_np(q[0, n:n + 1]), _to_np(k[0, n:n + 1]), _to_np(v[0, n:n + 1]),
            _to_np(g[0, n:n + 1]), _to_np(beta[0, n:n + 1]),
            _to_np(cache0[slots[n]]), scale, True)
        np.testing.assert_allclose(_to_np(o[0, n]), o_ref[0], rtol=2e-4, atol=2e-4)
        # cache slot updated to the oracle final state (fp32 cache)
        np.testing.assert_allclose(_to_np(cache[slots[n]]), S_ref, rtol=2e-4, atol=2e-4)
    # untouched slots preserved
    for s in range(nslots):
        if s not in slots:
            np.testing.assert_array_equal(_to_np(cache[s]), _to_np(cache0[s]))


# --------------------------------------------------------------------------- #
# 2. SSI 2D len-1 must behave identically to 1D.
# --------------------------------------------------------------------------- #
def test_ssi_2d_len1_equivalent_to_1d():
    rng = np.random.default_rng(1)
    N, Hg, HV, K, V = 2, 2, 4, 8, 8
    q, k, v, g, beta = _rand_inputs(rng, N, Hg, HV, K, V)
    slots = [1, 3]
    cu = torch.arange(N + 1, dtype=torch.int32)
    cache_a = torch.randn(5, HV, V, K, dtype=torch.float32) * 0.1
    cache_b = cache_a.clone()
    ssi1 = torch.tensor(slots, dtype=torch.int32)
    ssi2 = torch.tensor(slots, dtype=torch.int32).unsqueeze(-1)  # [N,1]

    oa, _ = _run_compat(q, k, v, g, beta, cache_a, cu_seqlens=cu, ssm_state_indices=ssi1)
    ob, _ = _run_compat(q, k, v, g, beta, cache_b, cu_seqlens=cu, ssm_state_indices=ssi2)
    np.testing.assert_allclose(_to_np(oa), _to_np(ob), rtol=0, atol=0)
    np.testing.assert_allclose(_to_np(cache_a), _to_np(cache_b), rtol=0, atol=0)


# --------------------------------------------------------------------------- #
# 3. FP32 work + correct (fp32) final state with bf16 inputs and fp32 cache.
# --------------------------------------------------------------------------- #
def test_fp32_final_state_not_downcast_to_bf16():
    rng = np.random.default_rng(2)
    N, Hg, HV, K, V = 2, 2, 4, 8, 8
    q, k, v, g, beta = _rand_inputs(rng, N, Hg, HV, K, V, dtype=torch.bfloat16)
    slots = [0, 1]
    cu = torch.arange(N + 1, dtype=torch.int32)
    cache = torch.randn(3, HV, V, K, dtype=torch.float32) * 0.1
    cache0 = cache.clone()
    ssi = torch.tensor(slots, dtype=torch.int32)
    scale = K ** -0.5

    o, fs = _run_compat(q, k, v, g, beta, cache, cu_seqlens=cu,
                        ssm_state_indices=ssi, inplace_final_state=True)

    # returned final state must be fp32 (FLA contract), not bf16
    assert fs.dtype == torch.float32
    for n in range(N):
        _, S_ref = oracle_sequence(
            _to_np(q[0, n:n + 1]), _to_np(k[0, n:n + 1]), _to_np(v[0, n:n + 1]),
            _to_np(g[0, n:n + 1]), _to_np(beta[0, n:n + 1]),
            _to_np(cache0[slots[n]]), scale, True)
        # fp32 returned state matches fp64 oracle of the SAME bf16 inputs closely
        np.testing.assert_allclose(_to_np(fs[n]), S_ref, rtol=2e-4, atol=2e-4)
        # bf16 rounding of the state would be ~1e-2 relative; ensure NOT bf16-rounded
        bf16_rounded = torch.tensor(S_ref, dtype=torch.float32).to(torch.bfloat16).to(torch.float32).numpy()
        err_fp32 = np.abs(_to_np(fs[n]) - S_ref).max()
        err_bf16 = np.abs(bf16_rounded - S_ref).max()
        assert err_fp32 < err_bf16 + 1e-6  # fp32 result strictly better than bf16 rounding


# --------------------------------------------------------------------------- #
# 4. Non-varlen batched decode honors ssm_state_indices (permuted slots).
# --------------------------------------------------------------------------- #
def test_non_varlen_honors_indices():
    rng = np.random.default_rng(3)
    B, T, Hg, HV, K, V = 3, 1, 2, 4, 8, 8
    q, k, v, g, beta = _rand_inputs(rng, T, Hg, HV, K, V)
    q = q.expand(B, T, Hg, K).contiguous()
    k = k.expand(B, T, Hg, K).contiguous()
    v = v.expand(B, T, HV, V).contiguous()
    g = g.expand(B, T, HV).contiguous()
    beta = beta.expand(B, T, HV).contiguous()
    # give each batch row distinct data
    q = torch.randn(B, T, Hg, K, dtype=torch.float32)
    k = torch.randn(B, T, Hg, K, dtype=torch.float32)
    v = torch.randn(B, T, HV, V, dtype=torch.float32)
    g = -torch.rand(B, T, HV, dtype=torch.float32) * 2.0
    beta = torch.rand(B, T, HV, dtype=torch.float32) * 0.9 + 0.05
    slots = [4, 1, 2]
    cache = torch.randn(6, HV, V, K, dtype=torch.float32) * 0.1
    cache0 = cache.clone()
    ssi = torch.tensor(slots, dtype=torch.int32)
    scale = K ** -0.5

    o, fs = _run_compat(q, k, v, g, beta, cache, ssm_state_indices=ssi,
                        inplace_final_state=True)  # cu_seqlens=None
    assert o.shape == (B, T, HV, V)
    for n in range(B):
        o_ref, S_ref = oracle_sequence(
            _to_np(q[n]), _to_np(k[n]), _to_np(v[n]), _to_np(g[n]), _to_np(beta[n]),
            _to_np(cache0[slots[n]]), scale, True)
        np.testing.assert_allclose(_to_np(o[n]), o_ref, rtol=2e-4, atol=2e-4)
        np.testing.assert_allclose(_to_np(cache[slots[n]]), S_ref, rtol=2e-4, atol=2e-4)


# --------------------------------------------------------------------------- #
# 5. Duplicate slot aliases: sequential fallback (seq n+1 sees seq n's write).
# --------------------------------------------------------------------------- #
def test_duplicate_slots_sequential_fallback():
    rng = np.random.default_rng(4)
    N, Hg, HV, K, V = 2, 2, 4, 8, 8
    q, k, v, g, beta = _rand_inputs(rng, N, Hg, HV, K, V)
    cu = torch.arange(N + 1, dtype=torch.int32)
    cache = torch.randn(3, HV, V, K, dtype=torch.float32) * 0.1
    cache0 = cache.clone()
    ssi = torch.tensor([1, 1], dtype=torch.int32)  # both sequences -> slot 1
    scale = K ** -0.5

    o, fs = _run_compat(q, k, v, g, beta, cache, cu_seqlens=cu,
                        ssm_state_indices=ssi, inplace_final_state=True)
    # seq 0 starts from cache0[1]
    o0, S0 = oracle_sequence(_to_np(q[0, 0:1]), _to_np(k[0, 0:1]), _to_np(v[0, 0:1]),
                             _to_np(g[0, 0:1]), _to_np(beta[0, 0:1]), _to_np(cache0[1]), scale, True)
    # seq 1 starts from seq 0's FINAL state (chained), not the original cache
    o1, S1 = oracle_sequence(_to_np(q[0, 1:2]), _to_np(k[0, 1:2]), _to_np(v[0, 1:2]),
                             _to_np(g[0, 1:2]), _to_np(beta[0, 1:2]), S0, scale, True)
    np.testing.assert_allclose(_to_np(o[0, 0]), o0[0], rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(_to_np(o[0, 1]), o1[0], rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(_to_np(cache[1]), S1, rtol=2e-4, atol=2e-4)


# --------------------------------------------------------------------------- #
# 6. Reject speculative num_accepted_tokens mode (not verified) loudly.
# --------------------------------------------------------------------------- #
def test_reject_num_accepted_tokens():
    rng = np.random.default_rng(5)
    N, Hg, HV, K, V = 2, 2, 4, 8, 8
    q, k, v, g, beta = _rand_inputs(rng, N, Hg, HV, K, V)
    cu = torch.arange(N + 1, dtype=torch.int32)
    cache = torch.zeros(3, HV, V, K, dtype=torch.float32)
    ssi = torch.tensor([[0, 1], [2, 0]], dtype=torch.int32)
    nat = torch.tensor([1, 2], dtype=torch.int32)
    with pytest.raises(NotImplementedError):
        _run_compat(q, k, v, g, beta, cache, cu_seqlens=cu,
                    ssm_state_indices=ssi, num_accepted_tokens=nat)


# --------------------------------------------------------------------------- #
# 7. Reject speculative multi-column SSI (2D with width > 1).
# --------------------------------------------------------------------------- #
def test_reject_spec_2d_multi_column_ssi():
    rng = np.random.default_rng(6)
    N, Hg, HV, K, V = 2, 2, 4, 8, 8
    q, k, v, g, beta = _rand_inputs(rng, N, Hg, HV, K, V)
    cu = torch.arange(N + 1, dtype=torch.int32)
    cache = torch.zeros(4, HV, V, K, dtype=torch.float32)
    ssi = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)  # [N,2] spec multi-query
    with pytest.raises(NotImplementedError):
        _run_compat(q, k, v, g, beta, cache, cu_seqlens=cu, ssm_state_indices=ssi)


# --------------------------------------------------------------------------- #
# 8. initial_state=None starts from zeros.
# --------------------------------------------------------------------------- #
def test_initial_state_none_starts_zero():
    rng = np.random.default_rng(7)
    N, Hg, HV, K, V = 2, 2, 4, 8, 8
    q, k, v, g, beta = _rand_inputs(rng, N, Hg, HV, K, V)
    cu = torch.arange(N + 1, dtype=torch.int32)
    scale = K ** -0.5
    o, fs = _run_compat(q, k, v, g, beta, None, cu_seqlens=cu, inplace_final_state=False)
    for n in range(N):
        o_ref, S_ref = oracle_sequence(
            _to_np(q[0, n:n + 1]), _to_np(k[0, n:n + 1]), _to_np(v[0, n:n + 1]),
            _to_np(g[0, n:n + 1]), _to_np(beta[0, n:n + 1]), None, scale, True)
        np.testing.assert_allclose(_to_np(o[0, n]), o_ref[0], rtol=2e-4, atol=2e-4)
        np.testing.assert_allclose(_to_np(fs[n]), S_ref, rtol=2e-4, atol=2e-4)


# --------------------------------------------------------------------------- #
# 9. Negative (pad) slot: zero output, no cache write.
# --------------------------------------------------------------------------- #
def test_negative_slot_skipped():
    rng = np.random.default_rng(8)
    N, Hg, HV, K, V = 2, 2, 4, 8, 8
    q, k, v, g, beta = _rand_inputs(rng, N, Hg, HV, K, V)
    cu = torch.arange(N + 1, dtype=torch.int32)
    cache = torch.randn(3, HV, V, K, dtype=torch.float32) * 0.1
    cache0 = cache.clone()
    ssi = torch.tensor([-1, 0], dtype=torch.int32)
    o, fs = _run_compat(q, k, v, g, beta, cache, cu_seqlens=cu, ssm_state_indices=ssi)
    np.testing.assert_array_equal(_to_np(o[0, 0]), np.zeros((HV, V)))
    np.testing.assert_array_equal(_to_np(cache[1]), _to_np(cache0[1]))
    np.testing.assert_array_equal(_to_np(cache[2]), _to_np(cache0[2]))


# --------------------------------------------------------------------------- #
# 10. Empty varlen sequence preserves incoming state.
# --------------------------------------------------------------------------- #
def test_empty_sequence_preserves_state():
    rng = np.random.default_rng(9)
    Hg, HV, K, V = 2, 4, 8, 8
    # seq0 has 1 token, seq1 empty, seq2 has 1 token
    q, k, v, g, beta = _rand_inputs(rng, 2, Hg, HV, K, V)
    cu = torch.tensor([0, 1, 1, 2], dtype=torch.int32)
    cache = torch.randn(3, HV, V, K, dtype=torch.float32) * 0.1
    cache0 = cache.clone()
    ssi = torch.tensor([0, 1, 2], dtype=torch.int32)
    o, fs = _run_compat(q, k, v, g, beta, cache, cu_seqlens=cu, ssm_state_indices=ssi)
    # empty seq1 -> slot1 unchanged
    np.testing.assert_array_equal(_to_np(cache[1]), _to_np(cache0[1]))


# --------------------------------------------------------------------------- #
# 11. Multi-token varlen sequence (T>1) matches oracle.
# --------------------------------------------------------------------------- #
def test_multi_token_varlen_matches_oracle():
    rng = np.random.default_rng(10)
    Hg, HV, K, V = 2, 4, 8, 8
    T0, T1 = 3, 2
    Ttot = T0 + T1
    q, k, v, g, beta = _rand_inputs(rng, Ttot, Hg, HV, K, V)
    cu = torch.tensor([0, T0, Ttot], dtype=torch.int32)
    cache = torch.randn(2, HV, V, K, dtype=torch.float32) * 0.1
    cache0 = cache.clone()
    ssi = torch.tensor([0, 1], dtype=torch.int32)
    scale = K ** -0.5
    o, fs = _run_compat(q, k, v, g, beta, cache, cu_seqlens=cu, ssm_state_indices=ssi)
    bounds = [(0, T0), (T0, Ttot)]
    for n, (bos, eos) in enumerate(bounds):
        o_ref, S_ref = oracle_sequence(
            _to_np(q[0, bos:eos]), _to_np(k[0, bos:eos]), _to_np(v[0, bos:eos]),
            _to_np(g[0, bos:eos]), _to_np(beta[0, bos:eos]), _to_np(cache0[n]), scale, True)
        np.testing.assert_allclose(_to_np(o[0, bos:eos]), o_ref, rtol=2e-4, atol=2e-4)
        np.testing.assert_allclose(_to_np(cache[n]), S_ref, rtol=2e-4, atol=2e-4)


# --------------------------------------------------------------------------- #
# 12. inplace_final_state=False returns fp32 states and leaves cache unchanged.
# --------------------------------------------------------------------------- #
def test_inplace_false_returns_state_cache_unchanged():
    rng = np.random.default_rng(11)
    N, Hg, HV, K, V = 2, 2, 4, 8, 8
    q, k, v, g, beta = _rand_inputs(rng, N, Hg, HV, K, V)
    cu = torch.arange(N + 1, dtype=torch.int32)
    cache = torch.randn(3, HV, V, K, dtype=torch.float32) * 0.1
    cache0 = cache.clone()
    ssi = torch.tensor([0, 2], dtype=torch.int32)
    scale = K ** -0.5
    o, fs = _run_compat(q, k, v, g, beta, cache, cu_seqlens=cu,
                        ssm_state_indices=ssi, inplace_final_state=False)
    assert fs.dtype == torch.float32
    np.testing.assert_array_equal(_to_np(cache), _to_np(cache0))  # cache untouched
    for n in range(N):
        _, S_ref = oracle_sequence(
            _to_np(q[0, n:n + 1]), _to_np(k[0, n:n + 1]), _to_np(v[0, n:n + 1]),
            _to_np(g[0, n:n + 1]), _to_np(beta[0, n:n + 1]),
            _to_np(cache0[ssi[n].item()]), scale, True)
        np.testing.assert_allclose(_to_np(fs[n]), S_ref, rtol=2e-4, atol=2e-4)


# --------------------------------------------------------------------------- #
# 13. GVA head expansion: g/beta carrying Hg heads expand to HV correctly.
# --------------------------------------------------------------------------- #
def test_gva_head_expansion_matches_oracle():
    rng = np.random.default_rng(12)
    N, Hg, HV, K, V = 2, 2, 4, 8, 8  # rep=2
    q, k, v, g, beta = _rand_inputs(rng, N, Hg, HV, K, V)
    # provide g/beta with Hg heads (must be expanded to HV by h//rep)
    g_hg = g[..., :Hg].contiguous()
    beta_hg = beta[..., :Hg].contiguous()
    cu = torch.arange(N + 1, dtype=torch.int32)
    cache = torch.zeros(2, HV, V, K, dtype=torch.float32)
    ssi = torch.tensor([0, 1], dtype=torch.int32)
    scale = K ** -0.5
    o, fs = _run_compat(q, k, v, g_hg, beta_hg, cache, cu_seqlens=cu, ssm_state_indices=ssi)
    # oracle uses g/beta expanded to HV
    rep = HV // Hg
    g_exp = g_hg.repeat_interleave(rep, dim=-1)
    beta_exp = beta_hg.repeat_interleave(rep, dim=-1)
    for n in range(N):
        o_ref, _ = oracle_sequence(
            _to_np(q[0, n:n + 1]), _to_np(k[0, n:n + 1]), _to_np(v[0, n:n + 1]),
            _to_np(g_exp[0, n:n + 1]), _to_np(beta_exp[0, n:n + 1]), None, scale, True)
        np.testing.assert_allclose(_to_np(o[0, n]), o_ref[0], rtol=2e-4, atol=2e-4)