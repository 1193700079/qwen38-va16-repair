"""Verify actual physical cache contents, not only softmax output tolerance."""
import torch,torch_vacc
from vllm.platforms import current_platform
import vllm_vacc
vllm_vacc.exec_patching()
from vllm_vacc.vllm.attention.ops.vacc_paged_attn import VaccPagedAttention as P
torch.vacc.set_device(0)
B=784;H=1;D=256
for strided in (False,True):
    if strided:
        base=torch.full((2*8*(B*H*D+256),),-3.,dtype=torch.bfloat16,device='vacc')
        cache=torch.as_strided(base,(2,8,B,H,D),(8*(B*H*D+256),B*H*D+256,H*D,D,1))
    else:cache=torch.full((2,8,B,H,D),-3.,dtype=torch.bfloat16,device='vacc')
    k=torch.arange(3*H*D).reshape(3,H,D).to(torch.bfloat16)
    v=-k
    slots=torch.tensor([2*B+13,1*B+300,4*B+783],dtype=torch.int32,device='vacc')
    kc,vc=P.split_kv_cache(cache,H,D)
    P.write_to_paged_cache(k.to('vacc'),v.to('vacc'),kc,vc,slots,'auto',1.,1.)
    torch.vacc.synchronize()
    host=cache.cpu()
    for i,s in enumerate(slots.cpu().tolist()):
        assert torch.equal(host[0,s//B,s%B],k[i]), f'key differs slot{s} strided{strided}'
        assert torch.equal(host[1,s//B,s%B],v[i]), f'value differs slot{s} strided{strided}'
    print('PASS exact cache write',strided,flush=True)
