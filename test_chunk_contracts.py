"""Contract regression suite for the prefill chunk GDN compatibility module.

Independent ground truth: the chunk gated-delta-rule is mathematically identical
to the token-by-token sequential recurrence, so the oracle here is a numpy fp64
SEQUENTIAL recurrence (not the chunk algorithm, and not gdn_vacc_ref/fast). The
chunk module's o and final_state must match the sequential oracle.

Adversarial cases (aligned / alternating l2norm keys, weak decay, high beta)
target the catastrophic-cancellation defect of the flat Newton-Schulz inverse;
the stable block inversion must match the fp64 oracle where flat NS diverges.

Run:
    /opt/homebrew/Caskroom/miniconda/base/bin/python3 -m pytest \
        test_chunk_contracts.py -v
"""
import numpy as np
import pytest
import torch

from chunk_compat import chunk_gated_delta_rule_compat

EPS = 1e-6


# --------------------------------------------------------------------------- #
# Independent fp64 SEQUENTIAL oracle (per-head / per-token recurrence).
# --------------------------------------------------------------------------- #
def oracle_sequence(q, k, v, g, beta, S0, scale, l2norm, eps=EPS):
    q = np.asarray(q, np.float64); k = np.asarray(k, np.float64)
    v = np.asarray(v, np.float64); g = np.asarray(g, np.float64)
    beta = np.asarray(beta, np.float64)
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


def _to_np(t):
    return t.detach().to(torch.float64).cpu().numpy()


def _rand(rng, B, T, Hg, HV, K, V, dtype=torch.float32):
    q = torch.randn(B, T, Hg, K, dtype=torch.float64).to(dtype)
    k = torch.randn(B, T, Hg, K, dtype=torch.float64).to(dtype)
    v = torch.randn(B, T, HV, V, dtype=torch.float64).to(dtype)
    g = -torch.rand(B, T, HV, dtype=torch.float64).to(dtype) * 2.0  # production g<=0
    beta = torch.rand(B, T, HV, dtype=torch.float64).to(dtype) * 0.9 + 0.05
    return q, k, v, g, beta


def _run(q, k, v, g, beta, initial_state, **kw):
    return chunk_gated_delta_rule_compat(
        q=q, k=k, v=v, g=g, beta=beta, initial_state=initial_state,
        output_final_state=True, use_qk_l2norm_in_kernel=True, **kw)


# --------------------------------------------------------------------------- #
# 1. Batched chunk (T not a multiple of 64) matches sequential oracle.
# --------------------------------------------------------------------------- #
def test_chunk_batched_matches_oracle():
    rng = np.random.default_rng(0)
    B, T, Hg, HV, K, V = 2, 70, 2, 4, 16, 16
    q, k, v, g, beta = _rand(rng, B, T, Hg, HV, K, V)
    init = torch.randn(B, HV, V, K, dtype=torch.float32) * 0.1
    scale = K ** -0.5
    o, fs = _run(q, k, v, g, beta, init)
    assert o.shape == (B, T, HV, V)
    assert fs.shape == (B, HV, V, K)
    for b in range(B):
        o_ref, S_ref = oracle_sequence(_to_np(q[b]), _to_np(k[b]), _to_np(v[b]),
                                       _to_np(g[b]), _to_np(beta[b]), _to_np(init[b]), scale, True)
        np.testing.assert_allclose(_to_np(o[b]), o_ref, rtol=2e-4, atol=2e-4)
        np.testing.assert_allclose(_to_np(fs[b]), S_ref, rtol=2e-4, atol=2e-4)


# --------------------------------------------------------------------------- #
# 2. Varlen chunk matches sequential oracle per sequence.
# --------------------------------------------------------------------------- #
def test_chunk_varlen_matches_oracle():
    rng = np.random.default_rng(1)
    Hg, HV, K, V = 2, 4, 16, 16
    lens = [13, 64, 70, 1]
    Ttot = sum(lens)
    q, k, v, g, beta = _rand(rng, 1, Ttot, Hg, HV, K, V)
    cu = torch.tensor([0] + list(np.cumsum(lens)), dtype=torch.int32)
    init = torch.randn(len(lens), HV, V, K, dtype=torch.float32) * 0.1
    scale = K ** -0.5
    o, fs = _run(q, k, v, g, beta, init, cu_seqlens=cu)
    bos = 0
    for n, L in enumerate(lens):
        eos = bos + L
        o_ref, S_ref = oracle_sequence(_to_np(q[0, bos:eos]), _to_np(k[0, bos:eos]),
                                       _to_np(v[0, bos:eos]), _to_np(g[0, bos:eos]),
                                       _to_np(beta[0, bos:eos]), _to_np(init[n]), scale, True)
        np.testing.assert_allclose(_to_np(o[0, bos:eos]), o_ref, rtol=2e-4, atol=2e-4)
        np.testing.assert_allclose(_to_np(fs[n]), S_ref, rtol=2e-4, atol=2e-4)
        bos = eos


# --------------------------------------------------------------------------- #
# 3. Empty varlen sequence preserves incoming initial state.
# --------------------------------------------------------------------------- #
def test_chunk_empty_sequence_preserves_state():
    rng = np.random.default_rng(2)
    Hg, HV, K, V = 2, 4, 16, 16
    q, k, v, g, beta = _rand(rng, 1, 5, Hg, HV, K, V)
    cu = torch.tensor([0, 5, 5, 5], dtype=torch.int32)  # seq1, seq2 empty
    init = torch.randn(3, HV, V, K, dtype=torch.float32) * 0.1
    o, fs = _run(q, k, v, g, beta, init, cu_seqlens=cu)
    np.testing.assert_allclose(_to_np(fs[1]), _to_np(init[1]), rtol=0, atol=0)
    np.testing.assert_allclose(_to_np(fs[2]), _to_np(init[2]), rtol=0, atol=0)


# --------------------------------------------------------------------------- #
# 4. Returned final_state is fp32 even with bf16 inputs (FLA contract).
# --------------------------------------------------------------------------- #
def test_chunk_final_state_fp32():
    rng = np.random.default_rng(3)
    B, T, Hg, HV, K, V = 1, 64, 2, 4, 16, 16
    q, k, v, g, beta = _rand(rng, B, T, Hg, HV, K, V, dtype=torch.bfloat16)
    init = torch.randn(B, HV, V, K, dtype=torch.float32) * 0.1
    o, fs = _run(q, k, v, g, beta, init)
    assert fs.dtype == torch.float32
    scale = K ** -0.5
    _, S_ref = oracle_sequence(_to_np(q[0]), _to_np(k[0]), _to_np(v[0]),
                               _to_np(g[0]), _to_np(beta[0]), _to_np(init[0]), scale, True)
    np.testing.assert_allclose(_to_np(fs[0]), S_ref, rtol=2e-3, atol=2e-3)


# --------------------------------------------------------------------------- #
# 5. Output o carries q.dtype (bf16), final_state stays fp32.
# --------------------------------------------------------------------------- #
def test_chunk_output_dtype_is_q_dtype():
    rng = np.random.default_rng(4)
    B, T, Hg, HV, K, V = 1, 64, 2, 4, 16, 16
    q, k, v, g, beta = _rand(rng, B, T, Hg, HV, K, V, dtype=torch.bfloat16)
    o, fs = _run(q, k, v, g, beta, None)
    assert o.dtype == torch.bfloat16
    assert fs.dtype == torch.float32


# --------------------------------------------------------------------------- #
# 6. ADVERSARIAL: alternating l2norm keys, weak decay, high beta.
#    Flat Newton-Schulz diverges here; stable block inversion must match oracle.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("T", [64, 128])
def test_adversarial_alternating_weakdecay_highbeta(T):
    rng = np.random.default_rng(5)
    Hg, HV, K, V = 2, 4, 16, 16
    k0 = torch.randn(Hg, K, dtype=torch.float64)
    k0 = k0 / torch.sqrt((k0 * k0).sum(-1, keepdim=True) + EPS)
    signs = torch.tensor([(-1.0) ** t for t in range(T)], dtype=torch.float64)
    k = (signs[None, :, None, None] * k0[None, None, :, :]).to(torch.float32)  # [1,T,Hg,K]
    q = torch.randn(1, T, Hg, K, dtype=torch.float32)
    v = torch.randn(1, T, HV, V, dtype=torch.float32)
    g = torch.full((1, T, HV), -1e-6, dtype=torch.float32)        # weak decay
    beta = torch.full((1, T, HV), float(torch.sigmoid(torch.tensor(10.0))), dtype=torch.float32)
    init = torch.randn(1, HV, V, K, dtype=torch.float32) * 0.1
    scale = K ** -0.5
    o, fs = _run(q, k, v, g, beta, init)
    o_ref, S_ref = oracle_sequence(_to_np(q[0]), _to_np(k[0]), _to_np(v[0]),
                                   _to_np(g[0]), _to_np(beta[0]), _to_np(init[0]), scale, True)
    assert torch.isfinite(o).all() and torch.isfinite(fs).all()
    np.testing.assert_allclose(_to_np(o[0]), o_ref, rtol=3e-3, atol=3e-3)
    np.testing.assert_allclose(_to_np(fs[0]), S_ref, rtol=3e-3, atol=3e-3)


# --------------------------------------------------------------------------- #
# 7. ADVERSARIAL: aligned (identical) l2norm keys, weak decay, high beta.
# --------------------------------------------------------------------------- #
def test_adversarial_aligned_identical_keys():
    rng = np.random.default_rng(6)
    T, Hg, HV, K, V = 64, 2, 4, 16, 16
    k0 = torch.randn(Hg, K, dtype=torch.float64)
    k0 = k0 / torch.sqrt((k0 * k0).sum(-1, keepdim=True) + EPS)
    k = k0[None, None, :, :].expand(1, T, Hg, K).contiguous().to(torch.float32)
    q = torch.randn(1, T, Hg, K, dtype=torch.float32)
    v = torch.randn(1, T, HV, V, dtype=torch.float32)
    g = torch.full((1, T, HV), -1e-6, dtype=torch.float32)
    beta = torch.full((1, T, HV), float(torch.sigmoid(torch.tensor(10.0))), dtype=torch.float32)
    scale = K ** -0.5
    o, fs = _run(q, k, v, g, beta, None)
    o_ref, S_ref = oracle_sequence(_to_np(q[0]), _to_np(k[0]), _to_np(v[0]),
                                   _to_np(g[0]), _to_np(beta[0]), None, scale, True)
    assert torch.isfinite(o).all() and torch.isfinite(fs).all()
    np.testing.assert_allclose(_to_np(o[0]), o_ref, rtol=3e-3, atol=3e-3)
    np.testing.assert_allclose(_to_np(fs[0]), S_ref, rtol=3e-3, atol=3e-3)


# --------------------------------------------------------------------------- #
# 8. Random bf16 inputs match fp64 oracle of the same bf16 values.
# --------------------------------------------------------------------------- #
def test_random_bf16_matches_oracle():
    rng = np.random.default_rng(7)
    B, T, Hg, HV, K, V = 2, 130, 2, 4, 16, 16
    q, k, v, g, beta = _rand(rng, B, T, Hg, HV, K, V, dtype=torch.bfloat16)
    init = torch.randn(B, HV, V, K, dtype=torch.float32) * 0.1
    scale = K ** -0.5
    o, fs = _run(q, k, v, g, beta, init)
    for b in range(B):
        o_ref, S_ref = oracle_sequence(_to_np(q[b]), _to_np(k[b]), _to_np(v[b]),
                                       _to_np(g[b]), _to_np(beta[b]), _to_np(init[b]), scale, True)
        np.testing.assert_allclose(_to_np(o[b].float()), o_ref, rtol=3e-2, atol=3e-2)
        np.testing.assert_allclose(_to_np(fs[b]), S_ref, rtol=3e-2, atol=3e-2)


# --------------------------------------------------------------------------- #
# 9. Long sequence (many chunks) matches oracle.
# --------------------------------------------------------------------------- #
def test_long_sequence_matches_oracle():
    rng = np.random.default_rng(8)
    B, T, Hg, HV, K, V = 1, 512, 2, 4, 16, 16
    q, k, v, g, beta = _rand(rng, B, T, Hg, HV, K, V)
    init = torch.randn(B, HV, V, K, dtype=torch.float32) * 0.1
    scale = K ** -0.5
    o, fs = _run(q, k, v, g, beta, init)
    o_ref, S_ref = oracle_sequence(_to_np(q[0]), _to_np(k[0]), _to_np(v[0]),
                                   _to_np(g[0]), _to_np(beta[0]), _to_np(init[0]), scale, True)
    np.testing.assert_allclose(_to_np(o[0]), o_ref, rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(_to_np(fs[0]), S_ref, rtol=2e-4, atol=2e-4)


# --------------------------------------------------------------------------- #
# 10. GVA: g/beta with Hg heads expand to HV.
# --------------------------------------------------------------------------- #
def test_gva_head_expansion():
    rng = np.random.default_rng(9)
    B, T, Hg, HV, K, V = 1, 70, 2, 4, 16, 16
    q, k, v, g, beta = _rand(rng, B, T, Hg, HV, K, V)
    g_hg = g[..., :Hg].contiguous(); beta_hg = beta[..., :Hg].contiguous()
    scale = K ** -0.5
    o, fs = _run(q, k, v, g_hg, beta_hg, None)
    rep = HV // Hg
    g_exp = g_hg.repeat_interleave(rep, dim=-1)
    beta_exp = beta_hg.repeat_interleave(rep, dim=-1)
    o_ref, S_ref = oracle_sequence(_to_np(q[0]), _to_np(k[0]), _to_np(v[0]),
                                   _to_np(g_exp[0]), _to_np(beta_exp[0]), None, scale, True)
    np.testing.assert_allclose(_to_np(o[0]), o_ref, rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(_to_np(fs[0]), S_ref, rtol=2e-4, atol=2e-4)


# --------------------------------------------------------------------------- #
# 11. Strong decay: no NaN/inf from upper-triangle positive exp (clamp fix).
# --------------------------------------------------------------------------- #
def test_strong_decay_no_nan_inf():
    rng = np.random.default_rng(10)
    B, T, Hg, HV, K, V = 1, 128, 2, 4, 16, 16
    q, k, v, g, beta = _rand(rng, B, T, Hg, HV, K, V)
    g = g * 50.0  # very negative -> strong decay, large |g_cs| differences
    o, fs = _run(q, k, v, g, beta, None)
    assert torch.isfinite(o).all()
    assert torch.isfinite(fs).all()
    scale = K ** -0.5
    o_ref, S_ref = oracle_sequence(_to_np(q[0]), _to_np(k[0]), _to_np(v[0]),
                                   _to_np(g[0]), _to_np(beta[0]), None, scale, True)
    np.testing.assert_allclose(_to_np(o[0]), o_ref, rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(_to_np(fs[0]), S_ref, rtol=2e-3, atol=2e-3)


# --------------------------------------------------------------------------- #
# 12. initial_state=None starts from zeros (batched + varlen).
# --------------------------------------------------------------------------- #
def test_initial_state_none_zeros():
    rng = np.random.default_rng(11)
    B, T, Hg, HV, K, V = 2, 64, 2, 4, 16, 16
    q, k, v, g, beta = _rand(rng, B, T, Hg, HV, K, V)
    scale = K ** -0.5
    o, fs = _run(q, k, v, g, beta, None)
    for b in range(B):
        o_ref, S_ref = oracle_sequence(_to_np(q[b]), _to_np(k[b]), _to_np(v[b]),
                                       _to_np(g[b]), _to_np(beta[b]), None, scale, True)
        np.testing.assert_allclose(_to_np(o[b]), o_ref, rtol=2e-4, atol=2e-4)
        np.testing.assert_allclose(_to_np(fs[b]), S_ref, rtol=2e-4, atol=2e-4)


# --------------------------------------------------------------------------- #
# 13. Many production-g random draws: finite + accurate (no overflow).
# --------------------------------------------------------------------------- #
def test_production_random_draws_finite():
    Hg, HV, K, V = 2, 4, 16, 16
    T = 96
    scale = K ** -0.5
    worst = 0.0
    for seed in range(20):
        rng = np.random.default_rng(100 + seed)
        q, k, v, g, beta = _rand(rng, 1, T, Hg, HV, K, V)
        o, fs = _run(q, k, v, g, beta, None)
        assert torch.isfinite(o).all() and torch.isfinite(fs).all()
        o_ref, S_ref = oracle_sequence(_to_np(q[0]), _to_np(k[0]), _to_np(v[0]),
                                       _to_np(g[0]), _to_np(beta[0]), None, scale, True)
        rel = np.abs(_to_np(fs[0]) - S_ref).max() / (np.abs(S_ref).max() + 1e-9)
        worst = max(worst, rel)
    assert worst < 1e-3