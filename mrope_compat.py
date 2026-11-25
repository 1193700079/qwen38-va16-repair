"""MRoPE adapter for vendor split sin/cos caches; no Triton or CustomOp dispatch.
Keeps original positional frequencies, three-axis mapping and partial rotation.
"""
import torch

def forward_mrope_compat(self, positions, query, key=None, offsets=None):
    if key is None: raise ValueError('MRoPE requires key')
    if not self.is_neox_style: raise NotImplementedError('Only checkpoint neox-style MRoPE verified')
    if positions.ndim not in (1,2): raise ValueError('Expected [T] or [3,T] positions')
    if offsets is not None: positions=positions+offsets
    pos=positions.long()
    cos=self.cos_cache[pos].to(query.dtype)
    sin=self.sin_cache[pos].to(query.dtype)
    if positions.ndim==2:
        if positions.shape[0]!=3: raise ValueError('MRoPE has three axes')
        section=self.mrope_section
        if not section or sum(section)!=self.rotary_dim//2: raise ValueError('Invalid MRoPE sections')
        if self.mrope_interleaved:
            # Same slice bounds as upstream apply_interleaved_rope.
            axes=[0]*(self.rotary_dim//2)
            for j in range(1,section[1]*3,3): axes[j]=1
            for j in range(2,section[2]*3,3): axes[j]=2
        else:
            axes=[i for i,n in enumerate(section) for _ in range(n)]
        idx=torch.tensor(axes,device=query.device,dtype=torch.long).view(1,1,-1).expand(1,pos.shape[-1],-1)
        cos=cos.gather(0,idx).squeeze(0)
        sin=sin.gather(0,idx).squeeze(0)
    co=cos.float().unsqueeze(1)
    si=sin.float().unsqueeze(1)
    half=self.rotary_dim//2
    def rotate(x):
        y=x.reshape(pos.shape[-1],-1,self.head_size)
        left=y[...,:half].float();right=y[...,half:self.rotary_dim].float()
        rot=torch.cat((left*co-right*si,right*co+left*si),-1).to(x.dtype)
        return torch.cat((rot,y[...,self.rotary_dim:]),-1).reshape(x.shape)
    return rotate(query),rotate(key)
