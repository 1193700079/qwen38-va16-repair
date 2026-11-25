import torch,torch_vacc, importlib.util
from pathlib import Path
from recurrent_compat import fused_recurrent_gated_delta_rule_compat as fixed
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--old',action='store_true');a=ap.parse_args()
torch.vacc.set_device(0)
if a.old:
    spec=importlib.util.spec_from_file_location('old','/work/sources/server_gdn_vacc_fast.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);fn=m.fused_recurrent_gated_delta_rule_fast
else:fn=fixed
q,k=[torch.randn(1,2,4,128).to(torch.bfloat16) for _ in range(2)]
v=torch.randn(1,2,12,128).to(torch.bfloat16);g=-torch.rand(1,2,12);beta=torch.rand(1,2,12).to(torch.bfloat16)
state=torch.randn(5,12,128,128)*.1
cu=torch.tensor([0,1,2],dtype=torch.int32);slots=torch.tensor([3,1],dtype=torch.int32)
args=dict(cu_seqlens=cu,ssm_state_indices=slots,initial_state=state.clone(),use_qk_l2norm_in_kernel=True)
expected,es=fixed(q,k,v,g,beta,**args)
actual_cache=state.to('vacc')
with torch.inference_mode():
    o,s=fn(*[t.to('vacc') for t in (q,k,v,g,beta)],cu_seqlens=cu,ssm_state_indices=slots,initial_state=actual_cache,use_qk_l2norm_in_kernel=True)
    torch.vacc.synchronize()
torch.testing.assert_close(o.float().cpu(),expected,rtol=.025,atol=.025)
torch.testing.assert_close(actual_cache.cpu(),args['initial_state'],rtol=.025,atol=.025)
assert s.dtype==torch.float32
print('PASS DEVICE DECODE 1D-SLOTS BF16/FP32',flush=True)
