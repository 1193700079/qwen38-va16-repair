from pathlib import Path
import ast
P=Path('/opt/venv/lib/python3.12/site-packages')
# Record CPU-readable cache map and active linear recycler only, no numerical mutations.
f=P/'attention_compat.py';s=f.read_text()
old='''    tables=_host(raw)
    lengths=_host(md.seq_lens_tensor)
    starts=_host(md.query_start_loc)'''
new=old+'''
    if not getattr(self,"_repair_mapping_logged",False):
        import json,os
        print("REPAIR_KV_MAP",json.dumps(dict(pid=os.getpid(),cache_shape=list(kc.shape),cache_stride=list(kc.stride()),blocksize=blocksize,tables=[r[:4] for r in tables],lengths=lengths,starts=starts,slots=_host(md.slot_mapping))),flush=True)
        self._repair_mapping_logged=True'''
assert s.count(old)==1;s=s.replace(old,new);ast.parse(s);f.write_text(s)
f=P/'vllm_vacc/vllm/model_executor/layers/linear.py';s=f.read_text()
old='''        parallel_embedding_output = None
        if memory_recycler is not None:'''
new='''        parallel_embedding_output = None
        if not getattr(self,"_repair_recycler_logged",False):
            print("REPAIR_RECYCLER",type(memory_recycler).__name__,flush=True)
            self._repair_recycler_logged=True
        if memory_recycler is not None:'''
assert s.count(old)==1;s=s.replace(old,new);ast.parse(s);f.write_text(s)
