"""Bounded diagnostic, not deployment. Reproduce BF16/state-layout variants.
Each case runs in its own named disposable container, without TP or a model.
"""
import argparse, importlib.util, json, time, os
import torch
import torch_vacc
ap=argparse.ArgumentParser()
ap.add_argument('--case',choices=['fp32','bf16','bf16-strided','bf16-inference','bf16-default','cache-only'],required=True)
a=ap.parse_args()
torch.vacc.set_device(0)
dev=torch.device('vacc:0')
print(json.dumps({'event':'init','case':a.case,'device':str(dev),'torch':torch.__version__}),flush=True)
spec=importlib.util.spec_from_file_location('gdn_under_test','/work/sources/server_gdn_vacc_fast.py')
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
torch.set_num_threads(1)

def mark(stage):
    print(json.dumps({'event':'before-sync','stage':stage}),flush=True)
    torch.vacc.synchronize()
    print(json.dumps({'event':'after-sync','stage':stage}),flush=True)

def run():
    T,Hg,HV,K,V=53,4,12,128,128
    dtype=torch.float32 if a.case=='fp32' else torch.bfloat16
    torch.manual_seed(9)
    # Generate common data on CPU; avoid random generation differences across devices.
    q0=torch.randn(1,T,Hg,K,device='cpu').to(dtype)
    k0=torch.randn(1,T,Hg,K,device='cpu').to(dtype)
    v0=torch.randn(1,T,HV,V,device='cpu').to(dtype)
    q,k,v=(z.to(dev) for z in (q0,k0,v0))
    g=(-torch.rand(1,T,HV,device='cpu')*.05).to(dev)
    beta=torch.rand(1,T,HV,device='cpu').to(dtype).to(dev)
    cu=torch.tensor([0,T],device='cpu',dtype=torch.int32)
    # Match padded, aliasing cache layout; unused trailing bytes remain sentinels.
    width=HV*V*K
    buf=torch.zeros(8*(width+256),device=dev,dtype=dtype)
    cache=torch.as_strided(buf,(8,HV,V,K),(width+256,V*K,K,1))
    ids=torch.tensor([2],device='cpu',dtype=torch.int32)
    s0=cache[ids].contiguous()
    mark('inputs')
    if a.case=='cache-only':
        state=torch.ones_like(s0)
        mark('state-created')
        cache[ids]=state
        mark('cache-write')
        torch.testing.assert_close(cache[2].cpu(),torch.ones_like(cache[2],device='cpu'))
        print('PASS cache-only',flush=True); return
    if a.case=='bf16-strided':
        # Real q/k/v are slices from a combined QKV projection tensor.
        packed=torch.cat((q.flatten(2),k.flatten(2),v.flatten(2)),dim=-1)
        q=packed[..., :Hg*K].view(1,T,Hg,K)
        k=packed[..., Hg*K:2*Hg*K].view(1,T,Hg,K)
        v=packed[..., 2*Hg*K:].view(1,T,HV,V)
    print(json.dumps({'event':'shapes','q':list(q.shape),'strides':list(q.stride()),'dtype':str(dtype),'state_dtype':str(s0.dtype)}),flush=True)
    # Synchronize after EVERY recorded stage in this isolated process.
    mod._sync=mark
    start=time.monotonic()
    o,s=mod.chunk_gated_delta_rule_fast(q,k,v,g,beta,initial_state=s0,output_final_state=True,cu_seqlens=cu,use_qk_l2norm_in_kernel=True)
    mark('chunk-return')
    cache[ids]=s.to(cache.dtype)
    mark('cache-written')
    got=o.float().cpu()
    assert torch.isfinite(got).all(), 'nonfinite output'
    print(json.dumps({'event':'PASS','case':a.case,'elapsed':time.monotonic()-start,'maxabs':float(got.abs().max())}),flush=True)

if a.case=='bf16-inference':
    with torch.inference_mode(): run()
elif a.case=='bf16-default':
    with torch.inference_mode(), torch.device(dev): run()
else: run()
