"""Regression-backed patch: CPU bool mask index-put poisons subsequent VACC ops.
No model math, data type or checkpoint changes; clear exactly the same slots.
"""
from pathlib import Path
import ast
p=Path('/opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_next.py')
s=p.read_text()
old='''            initial_state[~has_initial_state, ...] = 0'''
new='''            # Avoid CPU-bool-mask index_put_ on PrivateUse1: reproduced hang.
            # Metadata comes from the CPU scheduler; transfer at most once.
            _state_present = has_initial_state.detach().cpu().tolist()
            for _n, _has_state in enumerate(_state_present):
                if not _has_state:
                    initial_state[_n].zero_()'''
assert s.count(old)==1
s=s.replace(old,new)
ast.parse(s);p.write_text(s)
print('PATCH_CPU_MASK_APPLIED',flush=True)
