"""Regression for vendor RotaryEmbeddingBase ctor skipping CustomOp init."""
import argparse,math,sys
import torch,torch_vacc
ap=argparse.ArgumentParser();ap.add_argument('--fix',action='store_true');a=ap.parse_args()
from vllm.platforms import current_platform
import vllm_vacc
vllm_vacc.exec_patching()
from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding
from vllm.config import VllmConfig, set_current_vllm_config
if a.fix:
    from mrope_compat import forward_mrope_compat
    MRotaryEmbedding.forward=forward_mrope_compat

def oracle(x,pos,dim=64,hs=256):
    inp=x.float().cpu().reshape(x.shape[0],-1,hs).double()
    inv=1/(1e7**(torch.arange(0,dim,2,dtype=torch.float64)/dim))
    if pos.ndim==1:freq=pos.double().cpu()[:,None]*inv
    else:
        pi=pos.cpu().double()
        axes=[0]*32
        for j in range(1,33,3):axes[j]=1
        for j in range(2,30,3):axes[j]=2
        freq=torch.stack([pi[axes[j]]*inv[j] for j in range(32)],-1)
    co=freq.cos()[:,None,:];si=freq.sin()[:,None,:]
    # Compare mathematical RoPE (cache uses BF16 so allow that rounding).
    co=co.to(torch.bfloat16).double();si=si.to(torch.bfloat16).double()
    x1=inp[...,:32];x2=inp[...,32:64]
    rot=torch.cat((x1*co-x2*si,x2*co+x1*si,inp[...,64:]),-1)
    return rot.reshape(x.shape).to(x.dtype)

torch.vacc.set_device(0)
with set_current_vllm_config(VllmConfig()):
    with torch.device('vacc:0'):
        r=MRotaryEmbedding(256,64,128,1e7,True,torch.bfloat16,[11,11,10],True)
    for pos in [torch.tensor([0,1,17],dtype=torch.long),torch.tensor([[0,1,17],[0,4,9],[3,2,11]],dtype=torch.long)]:
        q=torch.randn(3,6*256).to(torch.bfloat16);k=torch.randn(3,256).to(torch.bfloat16)
        with torch.inference_mode():got_q,got_k=r(pos.to('vacc'),q.to('vacc'),k.to('vacc'))
        torch.vacc.synchronize()
        for inp,got in [(q,got_q),(k,got_k)]:
            torch.testing.assert_close(got.cpu(),oracle(inp,pos),rtol=.016,atol=.016)
            assert torch.equal(got.cpu().reshape(3,-1,256)[...,64:],inp.reshape(3,-1,256)[...,64:])
        print('PASS mrope positions',list(pos.shape),flush=True)
