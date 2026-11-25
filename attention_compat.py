"""Attention decode using actual physical page IDs (not fixed 8K groups).
Correctness-first no-prefix/no-spec decoder path; computation stays on VACC.
"""
import math
import torch

def _host(t):
    return t.detach().cpu().tolist() if isinstance(t,torch.Tensor) else list(t)

QCH=128  # query chunk to bound FP32 score memory for prefill/mixed batches

def forward_decode_compat(self,layer,query,key,value,kv_cache,md,output=None):
    if self.alibi_slopes is not None or self.sliding_window is not None:
        raise NotImplementedError('Only unwindowed decoder attention verified')
    from vllm_vacc.vllm.attention.ops.vacc_paged_attn import VaccPagedAttention
    q=query.view(-1,self.num_heads,self.head_size)
    kc,vc=VaccPagedAttention.split_kv_cache(kv_cache,self.num_kv_heads,self.head_size)
    if key is not None:
        VaccPagedAttention.write_to_paged_cache(key.view(-1,self.num_kv_heads,self.head_size),value.view(-1,self.num_kv_heads,self.head_size),kc,vc,md.slot_mapping,self.kv_cache_dtype,layer._k_scale,layer._v_scale)
    blocksize=kc.shape[1]
    raw=getattr(md,'_repair_raw_block_tables',None)
    if raw is None:raise ValueError('Raw physical block IDs required for hybrid decode')
    tables=_host(raw)
    lengths=_host(md.seq_lens_tensor)
    starts=_host(md.query_start_loc)
    hm=torch.arange(self.num_heads,device=q.device)//self.num_queries_per_kv
    outs=[]
    for i,row in enumerate(tables):
        n=int(lengths[i]); nb=math.ceil(n/blocksize)
        if nb>len(row):raise ValueError('Insufficient physical blocks')
        # Single gather of the physical pages preserves exact block order.
        idx=torch.tensor(row[:nb],device=q.device,dtype=torch.long)
        ks=kc.index_select(0,idx).reshape(-1,self.num_kv_heads,self.head_size)[:n]
        vs=vc.index_select(0,idx).reshape(-1,self.num_kv_heads,self.head_size)[:n]
        kg=ks[:,hm].float().transpose(0,1)   # [heads, n, head]
        vg=vs[:,hm].float().transpose(0,1)
        q0=starts[i]; q1=starts[i+1]; qlen=q1-q0
        base=n-qlen  # first query token's absolute position
        for c0 in range(0,qlen,QCH):
            c1=min(qlen,c0+QCH)
            qi=q[q0+c0:q0+c1].float().transpose(0,1)  # [heads, chunk, head]
            scores=torch.bmm(qi,kg.transpose(-1,-2))*self.scale  # [heads,chunk,n]
            pos=base+torch.arange(c0,c1,device=q.device)
            keypos=torch.arange(n,device=q.device)
            causal=keypos[None,:]<=pos[:,None]  # [chunk,n]
            scores=scores.masked_fill(~causal[None,:,:],float('-inf'))
            attn=torch.bmm(scores.softmax(-1),vg).transpose(0,1).to(q.dtype)
            outs.append(attn)
    return torch.cat(outs,dim=0).reshape(-1,self.num_heads*self.head_size)

def install():
    from vllm_vacc.vllm.v1.attention.backends.vacc_attn import VACCAttentionBackendImpl as Impl,VACCAttentionMetadataBuilderV1 as Builder
    if getattr(Impl,'_repair_paged_decode',False):return
    orig_forward=Impl.forward;orig_build=Builder.build
    def build(self,common_prefix_len,common_attn_metadata,fast_build=False):
        md=orig_build(self,common_prefix_len,common_attn_metadata,fast_build)
        md._repair_raw_block_tables=common_attn_metadata.block_table_tensor[:common_attn_metadata.num_reqs]
        return md
    def forward(self,layer,query,key,value,kv_cache,attn_metadata,output=None):
        if attn_metadata is not None and attn_metadata.num_decode_tokens:
            return forward_decode_compat(self,layer,query,key,value,kv_cache,attn_metadata,output)
        return orig_forward(self,layer,query,key,value,kv_cache,attn_metadata,output)
    Builder.build=build;Impl.forward=forward;Impl._repair_paged_decode=True
