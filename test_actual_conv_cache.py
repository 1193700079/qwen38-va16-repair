import torch,torch_vacc,importlib.util
from vllm.platforms import current_platform
import vllm_vacc
vllm_vacc.exec_patching()
spec=importlib.util.spec_from_file_location('existing','/work/sources/server_gdn_vacc_fast.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
torch.vacc.set_device(0);torch.manual_seed(44)
D=2560;W=4;T=13
x=torch.randn(D,T+4).to(torch.bfloat16)
w=torch.randn(D,W).to(torch.bfloat16)*.2
for padded in (False,True):
    if padded:
        raw=torch.zeros(50*401408,device='vacc',dtype=torch.bfloat16)
        cs=torch.as_strided(raw,(50,D,3),(401408,1,D))
    else:cs=torch.zeros(50,D,3,device='vacc',dtype=torch.bfloat16)
    slot=torch.tensor([2],device='vacc',dtype=torch.int32)
    with torch.inference_mode():
        o=m.causal_conv1d_fn_fast(x[:,:T].to('vacc'),w.to('vacc'),None,cs,torch.tensor([0,T]),cache_indices=slot,has_initial_state=torch.tensor([False]))
        torch.vacc.synchronize()
        expected=torch.nn.functional.silu(torch.nn.functional.conv1d(torch.nn.functional.pad(x[:,:T].float(),(3,0)).unsqueeze(0),w.float().unsqueeze(1),groups=D)).squeeze(0).to(torch.bfloat16)
        torch.testing.assert_close(o.cpu(),expected,rtol=.02,atol=.02)
        torch.testing.assert_close(cs[2].cpu(),x[:,T-3:T],rtol=0,atol=0)
        for n in range(4):
            out=m.causal_conv1d_update_fast(x[:,T+n].unsqueeze(0).to('vacc'),cs,w.to('vacc'),activation='silu',conv_state_indices=slot)
            torch.vacc.synchronize()
            ref=torch.nn.functional.silu((x[:,T+n-3:T+n+1].float()*w.float()).sum(-1)).to(torch.bfloat16)
            torch.testing.assert_close(out.cpu()[0],ref,rtol=.02,atol=.02)
            torch.testing.assert_close(cs[2].cpu(),x[:,T+n-2:T+n+1],rtol=0,atol=0)
    print('PASS REAL CONV CACHE',padded,flush=True)
