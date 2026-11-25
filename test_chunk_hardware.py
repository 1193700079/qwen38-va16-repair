"""Device regression: actual capture and structured keys vs independent FP64 oracle."""
import argparse,importlib.util,time,sys
import numpy as np
import torch,torch_vacc
ap=argparse.ArgumentParser();ap.add_argument('--old',action='store_true');a=ap.parse_args()
torch.vacc.set_device(0)
# Extract only oracle function; do not execute pytest module or previous patches.
import ast
from pathlib import Path
src=ast.parse(Path('/work/test_chunk_contracts.py').read_text())
node=next(n for n in src.body if isinstance(n,ast.FunctionDef) and n.name=='oracle_sequence')
ns={'np':np,'EPS':1e-6};exec(compile(ast.Module(body=[node],type_ignores=[]),'<independent-oracle>','exec'),ns)
oracle=ns['oracle_sequence']
if a.old:
 spec=importlib.util.spec_from_file_location('legacy','/work/sources/server_gdn_vacc_fast.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);fn=mod.chunk_gated_delta_rule_fast
else:
 from chunk_compat import chunk_gated_delta_rule_compat as fn

def check(label,q,k,v,g,beta,s0):
 T=q.shape[1];K=q.shape[-1]
 seq=lambda x:x[0].double().numpy()
 o64,s64=oracle(seq(q),seq(k),seq(v),seq(g),seq(beta),s0[0].double().numpy(),K**-.5,True)
 cu=torch.tensor([0,T],device='cpu',dtype=torch.int32)
 start=time.monotonic()
 with torch.inference_mode():
  o,s=fn(*[t.to('vacc') for t in (q,k,v,g,beta)],initial_state=s0.to('vacc'),output_final_state=True,cu_seqlens=cu,use_qk_l2norm_in_kernel=True)
  torch.vacc.synchronize()
 oc=o.float().cpu().numpy()[0];sc=s.float().cpu().numpy()[0]
 assert s.dtype==torch.float32, f'final state dtype contract: {s.dtype}'
 np.testing.assert_allclose(oc,o64,rtol=.03,atol=.015)
 np.testing.assert_allclose(sc,s64,rtol=.03,atol=.015)
 print('PASS',label,'seconds',time.monotonic()-start,'output_max_error',np.max(np.abs(oc-o64)),'state_max_error',np.max(np.abs(sc-s64)),flush=True)

c=torch.load('/work/capture-754.pt',map_location='cpu',weights_only=True)
check('captured-layer0',*[c[k] for k in ['q','k','v','g','beta','initial_state']])
torch.manual_seed(110)
for label,strong,aligned in [('alternating-weak-decay',False,True),('strong-decay',True,False),('multi-chunk',False,False)]:
 T=129 if label=='multi-chunk' else 64
 Hg,HV,K,V=4,12,128,128
 q=torch.randn(1,T,Hg,K).to(torch.bfloat16)
 k=torch.randn(1,T,Hg,K).to(torch.bfloat16)
 if aligned:k=k[:,:1].expand(1,T,Hg,K).clone()*((-1.)**torch.arange(T)).reshape(1,T,1,1)
 k=k.to(torch.bfloat16)
 v=torch.randn(1,T,HV,V).to(torch.bfloat16)
 g=torch.full((1,T,HV),-20. if strong else -1e-6)
 beta=torch.full((1,T,HV),.99995).to(torch.bfloat16)
 s0=torch.randn(1,HV,V,K)*.1
 check(label,q,k,v,g,beta,s0)
