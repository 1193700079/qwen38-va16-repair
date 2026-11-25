"""Diagnostic-only fail-fast synchronization at actual GDN boundaries."""
from pathlib import Path
import ast
p=Path('/opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_next.py')
s=p.read_text()
assert '# REPAIR_BOUNDARIES_20261125' not in s
helper='''
# REPAIR_BOUNDARIES_20261125
import os as _repair_os
import json as _repair_json
import time as _repair_time

def _rb(label, **tensors):
    info = {}
    for name, x in tensors.items():
        if isinstance(x, torch.Tensor):
            info[name] = dict(shape=list(x.shape), dtype=str(x.dtype), device=str(x.device), stride=list(x.stride()), offset=x.storage_offset(), contiguous=x.is_contiguous())
            if x.device.type == "cpu" and x.numel() < 100:
                info[name]["values"] = x.tolist()
        else:
            info[name] = str(x)
    print("REPAIR_BOUNDARY " + _repair_json.dumps(dict(pid=_repair_os.getpid(), label=label, phase="before", tensors=info)), flush=True)
    start = _repair_time.monotonic()
    torch.vacc.synchronize()
    print("REPAIR_BOUNDARY " + _repair_json.dumps(dict(pid=_repair_os.getpid(), label=label, phase="after", seconds=_repair_time.monotonic()-start)), flush=True)

'''
s+='\n'+helper
repls=[
('''        self_kv_cache = self.kv_cache[forward_context.virtual_engine]''','''        _rb("gdn_entry", mixed=mixed_qkv, a=a, b=b)
        self_kv_cache = self.kv_cache[forward_context.virtual_engine]'''),
('''        num_actual_tokens = attn_metadata.num_actual_tokens''','''        _rb("cache_views", conv=conv_state, ssm=ssm_state, indices=non_spec_state_indices_tensor, has_initial=has_initial_state)
        num_actual_tokens = attn_metadata.num_actual_tokens'''),
('''        query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)''','''        _rb("after_conv", mixed=mixed_qkv_non_spec)
        query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)'''),
('''        g, beta = fused_gdn_gating(self.A_log, a, b, self.dt_bias)''','''        _rb("after_rearrange", q=query_non_spec, k=key_non_spec, v=value_non_spec)
        g, beta = fused_gdn_gating(self.A_log, a, b, self.dt_bias)
        _rb("after_gating", g=g, beta=beta)'''),
('''            initial_state = ssm_state[non_spec_state_indices_tensor].contiguous()
            initial_state[~has_initial_state, ...] = 0''','''            initial_state = ssm_state[non_spec_state_indices_tensor].contiguous()
            _rb("after_state_gather", initial=initial_state, has_initial=has_initial_state)
            initial_state[~has_initial_state, ...] = 0
            _rb("after_state_zero", initial=initial_state)'''),
('''            # Init cache
            ssm_state''','''            _rb("after_chunk", output=core_attn_out_non_spec, state=last_recurrent_state)
            # Init cache
            ssm_state'''),
('''        elif attn_metadata.num_decodes > 0:
            core_attn_out_non_spec''','''            _rb("after_cache_write", ssm=ssm_state)
        elif attn_metadata.num_decodes > 0:
            core_attn_out_non_spec'''),
]
for old,new in repls:
    if s.count(old)!=1: raise RuntimeError(f'Expected unique anchor: {old[:80]} count={s.count(old)}')
    s=s.replace(old,new,1)
ast.parse(s)
p.write_text(s)
print('Instrumented actual GDN boundaries', flush=True)
