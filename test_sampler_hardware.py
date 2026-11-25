import torch,torch_vacc
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--fix',action='store_true');a=ap.parse_args()
from vllm.platforms import current_platform
import vllm_vacc
vllm_vacc.exec_patching()
torch.vacc.set_device(0)
fn=torch.vacc.sampler_v1
if a.fix:
    from sampler_compat import sample_vacc_compat
    fn=sample_vacc_compat
for n in [32,248320]:
    torch.manual_seed(29)
    logits=torch.randn(1,n,dtype=torch.float32)
    for target in [1,n//2,n-1]:
        values=logits.clone(); values[0,target]=50.
        expected=values.argmax(-1)
        with torch.inference_mode():
            got,_=fn(values.to('vacc'),torch.tensor([1.],device='cpu'),torch.tensor([0],device='cpu',dtype=torch.int32),torch.tensor([0.],device='cpu'),1,0,{})
            torch.vacc.synchronize()
        actual=got.cpu().long()
        print('GREEDY',n,target,'actual',actual.tolist(),str(got.dtype),flush=True)
        assert torch.equal(actual,expected),(n,target,actual.tolist())
print('SAMPLER_GREEDY_PASS',flush=True)
