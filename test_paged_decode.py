"""Regression: hybrid attention pages are not fixed 8192-token groups."""
import argparse,types
import torch,torch_vacc
ap=argparse.ArgumentParser();ap.add_argument('--fix',action='store_true');a=ap.parse_args()
from vllm.platforms import current_platform
import vllm_vacc
vllm_vacc.exec_patching()
from vllm_vacc.vllm.v1.attention.backends.vacc_attn import VACCAttentionBackendImpl,VACCAttentionMetadata
if a.fix:
 import attention_compat
 attention_compat.install()
torch.vacc.set_device(0)
torch.manual_seed(22)
impl=VACCAttentionBackendImpl(6,256,256**-.5,1,None,None,'auto')
layer=types.SimpleNamespace(_k_scale=torch.tensor(1.),_v_scale=torch.tensor(1.))
for length in (15,790):
 B=784; block_ids=[3,1]; cache=torch.zeros(2,5,B,1,256,device='vacc',dtype=torch.bfloat16)
 kh=torch.randn(length,1,256).to(torch.bfloat16);vh=torch.randn_like(kh)
 for t in range(length-1):
  # Prepare host buffers then a single transfer, not one H2D/token.
  pass
 host=torch.zeros(2,5,B,1,256,dtype=torch.bfloat16)
 for t in range(length-1):
  host[0,block_ids[t//B],t%B]=kh[t];host[1,block_ids[t//B],t%B]=vh[t]
 cache.copy_(host)
 q=torch.randn(1,6,256).to(torch.bfloat16)
 slots=torch.tensor([block_ids[(length-1)//B]*B+(length-1)%B],dtype=torch.int32,device='vacc')
 table=torch.tensor([block_ids],dtype=torch.int32,device='vacc')
 md=VACCAttentionMetadata(0,0,1,slots,[length],length,table,False,seq_lens=[length],query_start_loc=torch.tensor([0,1],dtype=torch.int32))
 md._repair_raw_block_tables=table
 with torch.inference_mode():out=impl.forward(layer,q.to('vacc'),kh[-1:].to('vacc'),vh[-1:].to('vacc'),cache,md)
 torch.vacc.synchronize()
 keys=kh.float().repeat_interleave(6,dim=1).permute(1,0,2)
 vals=vh.float().repeat_interleave(6,dim=1).permute(1,0,2)
 logits=q.float().permute(1,0,2)@keys.transpose(-1,-2)*(256**-.5)
 expected=(logits.softmax(-1)@vals).permute(1,0,2).to(torch.bfloat16)
 torch.testing.assert_close(out.cpu().reshape_as(expected),expected,rtol=.025,atol=.025)
 print('PASS PAGED DECODE',length,'page',B,'out_shape',list(out.shape),flush=True)
