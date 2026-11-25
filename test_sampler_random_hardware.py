import torch, torch_vacc
from sampler_compat import sample_vacc_compat
torch.utils.rename_privateuse1_backend("vacc"); torch.vacc.set_device(0)
V=248320
torch.manual_seed(3)
base=torch.randn(2,V,dtype=torch.float32); base[0,123456]=40.; base[1,777]=40.
d=base.to('vacc')
def call(temp,tk,tp,seeds=None):
    gens={}
    if seeds:
        for i,s in enumerate(seeds):
            g=torch.Generator(device='vacc'); g.manual_seed(s); gens[i]=g
    ids,_=sample_vacc_compat(d, torch.tensor(tp), torch.tensor(tk), torch.tensor(temp), False, True, gens)
    torch.vacc.synchronize()
    return ids
ids=call([0.,0.],[0,0],[1.,1.]); assert ids.cpu().tolist()==[123456,777]
assert ids.dtype==torch.int32 and ids.device.type=='vacc'
ids=call([1.,1.],[1,1],[1.,1.]); assert ids.cpu().tolist()==[123456,777]
a=call([1.,1.],[0,0],[1.,1.],seeds=[5,6]); b=call([1.,1.],[0,0],[1.,1.],seeds=[5,6])
assert torch.equal(a.cpu(),b.cpu())
print('SAMPLER_RANDOM_DEVICE_PASS',flush=True)
