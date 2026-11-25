"""Regression: CPU boolean advanced indexing must not stall the next VACC op.
Run each mode in a separate bounded container. Original mode is reproduction;
slot mode is equivalent explicit per-slot clearing to validate the hypothesis.
"""
import argparse,time
import torch,torch_vacc
ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['original','slot'],required=True);a=ap.parse_args()
torch.vacc.set_device(0)
dev='vacc:0'
with torch.inference_mode():
    width=12*128*128
    pool=torch.zeros(50*(width+4096),device=dev,dtype=torch.float32)
    state=torch.as_strided(pool,(50,12,128,128),(width+4096,16384,128,1),storage_offset=3840)
    idx=torch.tensor([0,2],device=dev,dtype=torch.int32)
    initial=state[idx].contiguous()
    initial.fill_(3.)
    mask=torch.tensor([False,True],device='cpu',dtype=torch.bool)
    torch.vacc.synchronize()
    print('BEFORE CLEAR',flush=True)
    if a.mode=='original':initial[~mask,...]=0
    else:
        for n,has in enumerate(mask.tolist()):
            if not has: initial[n].zero_()
    torch.vacc.synchronize()
    print('AFTER CLEAR',flush=True)
    out=torch.zeros((13,12,128),device=dev,dtype=torch.float32)
    torch.vacc.synchronize()
    print('AFTER NEXT FILL',flush=True)
    torch.testing.assert_close(initial[0].cpu(),torch.zeros((12,128,128)))
    torch.testing.assert_close(initial[1].cpu(),torch.full((12,128,128),3.))
    print('PASS',a.mode,flush=True)
