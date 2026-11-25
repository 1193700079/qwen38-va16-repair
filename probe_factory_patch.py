from pathlib import Path
p=Path('/opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_next.py')
s=p.read_text()
old='''            initial_state = ssm_state[non_spec_state_indices_tensor].contiguous()'''
probe='''            import os as _os
            _probe_where = _os.environ.get("REPAIR_PROBE_WHERE", "before")
            if _probe_where == "before":
                _factory_probe(query_non_spec)
            initial_state = ssm_state[non_spec_state_indices_tensor].contiguous()'''
assert s.count(old)==1;s=s.replace(old,probe)
old='''            initial_state[~has_initial_state, ...] = 0'''
new='''            if _probe_where == "after_gather":
                _factory_probe(query_non_spec)
'''+old+'''
            if _probe_where == "after":
                _factory_probe(query_non_spec)'''
assert s.count(old)==1;s=s.replace(old,new)
s+='''

def _factory_probe(q):
    import os, json
    print("FACTORY context",os.getpid(),q.device,torch.vacc.current_stream(),torch.is_inference_mode_enabled(),torch.is_grad_enabled(),torch.get_default_dtype(),flush=True)
    print("FACTORY allocate_empty before",flush=True)
    y=torch.empty((1,13,12,128),device=q.device,dtype=torch.float32)
    torch.vacc.synchronize()
    print("FACTORY allocate_empty after",flush=True)
    print("FACTORY fill_zero before",flush=True)
    y.zero_()
    torch.vacc.synchronize()
    print("FACTORY fill_zero after",flush=True)
    print("FACTORY zeros before",flush=True)
    z=torch.zeros((13,12,128),device=q.device,dtype=torch.float32)
    torch.vacc.synchronize()
    print("FACTORY zeros after",flush=True)
    print("FACTORY add before",flush=True)
    w=z+1
    torch.vacc.synchronize()
    print("FACTORY add after",flush=True)
    raise RuntimeError("FACTORY_PROBE_COMPLETE")
'''
p.write_text(s)
