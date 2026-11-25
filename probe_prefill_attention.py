import torch,torch_vacc
from vllm.platforms import current_platform
import vllm_vacc
vllm_vacc.exec_patching()
from vllm_vacc.vllm.v1.attention.backends.vacc_attn import VACCAttentionBackendImpl,VACCAttentionMetadata
from types import SimpleNamespace
torch.vacc.set_device(0)
torch.manual_seed(12)
T,H,KV,D=21,6,1,256
q=torch.randn(T,H,D).to(torch.bfloat16);k=torch.randn(T,KV,D).to(torch.bfloat16);v=torch.randn_like(k)
md=VACCAttentionMetadata(1,T,0,torch.arange(T,dtype=torch.int32,device='vacc'),[T],0,torch.empty((0,0),dtype=torch.int32,device='vacc'),False,seq_lens=[T],query_start_loc=torch.tensor([0,T],dtype=torch.int32))
impl=VACCAttentionBackendImpl(H,D,D**-.5,KV,None,None,'auto')
layer=SimpleNamespace(_k_scale=torch.tensor(1.),_v_scale=torch.tensor(1.))
cache=torch.zeros(2,5,784,KV,D,device='vacc',dtype=torch.bfloat16)
with torch.inference_mode():out=impl.forward(layer,q.to('vacc'),k.to('vacc'),v.to('vacc'),cache,md)
torch.vacc.synchronize()
kh=k.float().repeat_interleave(H//KV,dim=1).permute(1,0,2);vh=v.float().repeat_interleave(H//KV,dim=1).permute(1,0,2)
scores=q.float().permute(1,0,2)@kh.transpose(-1,-2)*D**-.5
mask=torch.arange(T)[None,:]>torch.arange(T)[:,None]
scores=scores.masked_fill(mask,-torch.inf)
ref=(scores.softmax(-1)@vh).permute(1,0,2).reshape(T,-1)
got=out.float().cpu()
print('MAX_ABS',float((got-ref).abs().max()),'REL_L2',float((got-ref).norm()/ref.norm()),flush=True)
torch.testing.assert_close(got,ref,rtol=.03,atol=.03)
print('PREFILL_ATTENTION_PASS',flush=True)
