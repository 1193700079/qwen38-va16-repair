from pathlib import Path
import ast
p=Path('/opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_next.py')
s=p.read_text()
old='''            initial_state[~has_initial_state, ...] = 0
            ('''
new='''            initial_state[~has_initial_state, ...] = 0
            # Capture real first-layer input, synchronizing BEFORE our kernel.
            import os as _os
            torch.vacc.synchronize()
            _capture = dict(q=query_non_spec.cpu(), k=key_non_spec.cpu(),
                v=value_non_spec.cpu(), g=g_non_spec.cpu(), beta=beta_non_spec.cpu(),
                initial_state=initial_state.cpu(), cu_seqlens=non_spec_query_start_loc.cpu(),
                cache_shape=tuple(ssm_state.shape), cache_stride=tuple(ssm_state.stride()),
                cache_dtype=ssm_state.dtype, cache_offset=ssm_state.storage_offset(),
                slots=non_spec_state_indices_tensor.cpu(), prefix=self.prefix)
            _filename = f"/work/capture-{_os.getpid()}.pt"
            torch.save(_capture, _filename)
            print("REPAIR_CAPTURE_SAVED", _filename, self.prefix, flush=True)
            raise RuntimeError("REPAIR_CAPTURE_COMPLETE: intentionally stop before GDN")
            ('''
assert s.count(old)==1
s=s.replace(old,new)
ast.parse(s);p.write_text(s)
print('Capture only; no compute patch',flush=True)
