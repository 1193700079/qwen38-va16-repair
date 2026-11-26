"""VACC-safe vision (ViT) attention.

The vendor MMEncoderAttention falls back to forward_native -> torch SDPA on
PrivateUse1/VACC, which segfaults the worker (no Python traceback). Replace it
with an explicit batched-matmul full (non-causal) attention, the same primitive
family proven stable on VACC by the GDN/attention repairs.
"""
import os
import torch


def _cpu(t):
    if t is None:
        return None
    if isinstance(t, (list, tuple)):
        return list(t)
    return t.detach().to("cpu").tolist()


def _bmm(a, b):
    lead = a.shape[:-2]
    n, m = a.shape[-2], a.shape[-1]
    p = b.shape[-1]
    return torch.bmm(a.reshape(-1, n, m), b.reshape(-1, m, p)).reshape(*lead, n, p)


def _full_attn(q, k, v, scale):
    # q,k,v: [B,H,L,D] float32
    scores = _bmm(q, k.transpose(-1, -2)) * scale
    probs = torch.softmax(scores, dim=-1)
    return _bmm(probs, v)


def forward_mm_compat(self, query, key, value, cu_seqlens=None, **kw):
    bsz, q_len = query.size()[:2]
    kv_len = key.size(1)
    is_reshaped = query.dim() != 4
    q, k, v = self.view_qkv_to_4d(query, key, value, bsz, q_len, kv_len)
    q = q.permute(0, 2, 1, 3).float()
    k = k.permute(0, 2, 1, 3).float()
    v = v.permute(0, 2, 1, 3).float()
    if self.num_heads > self.num_kv_heads:
        rep = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    cs = _cpu(cu_seqlens)
    if cs is not None and bsz == 1 and len(cs) > 2:
        outs = []
        for i in range(len(cs) - 1):
            a, b = int(cs[i]), int(cs[i + 1])
            if b > a:
                outs.append(_full_attn(q[:, :, a:b], k[:, :, a:b], v[:, :, a:b], self.scale))
        out = torch.cat(outs, dim=2) if outs else q.new_zeros(q.shape)
    else:
        out = _full_attn(q, k, v, self.scale)
    out = out.permute(0, 2, 1, 3).to(query.dtype)
    if is_reshaped:
        out = out.reshape(bsz, q_len, -1)
    return out


def _lin(layer, x):
    out = layer(x)
    if isinstance(out, tuple):
        out = out[0]
    return out


def forward_merger_compat(self, x):
    if self.use_postshuffle_norm:
        x = self.norm(x.view(-1, self.hidden_size))
    else:
        x = self.norm(x).view(-1, self.hidden_size)
    x = _lin(self.linear_fc1, x)
    x = self.act_fn(x)
    return _lin(self.linear_fc2, x)


def forward_vmlp_compat(self, x):
    shape = x.shape
    x = _lin(self.linear_fc1, x)
    x = self.act_fn(x)
    out = _lin(self.linear_fc2, x)
    return out.view(shape)


def install():
    from vllm.model_executor.layers.attention.mm_encoder_attention import MMEncoderAttention
    MMEncoderAttention.forward = forward_mm_compat
    MMEncoderAttention.forward_native = forward_mm_compat
    # Vendor fused vision ops (patch_merger_vision / fuse_mlp_vision) segfault on
    # this VACC build; use the pure-torch equivalents (TP linears all-reduce themselves).
    try:
        from vllm_vacc.vllm.model_executor.models import qwen3_vl as _vvl
        _vvl.Qwen3_VisionPatchMerger.forward = forward_merger_compat
        _vvl.Qwen3_VisionMLP.forward = forward_vmlp_compat
    except Exception as e:  # pragma: no cover
        import logging
        logging.getLogger(__name__).warning("mm vision mlp/merger patch skipped: %s", e)
