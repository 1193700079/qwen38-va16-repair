import argparse, ast, types, time, json
from pathlib import Path
import torch
ap=argparse.ArgumentParser();ap.add_argument('--device',default='cpu');ap.add_argument('--source',default='/work/sources/server_gdn_vacc_fast.py');ap.add_argument('--capture',default='/work/capture-754.pt');ap.add_argument('--trace',action='store_true');ap.add_argument('--bf16-default',action='store_true');ap.add_argument('--custom-op',action='store_true');a=ap.parse_args()
if a.device!='cpu':
    import torch_vacc
    torch.vacc.set_device(0)
data=torch.load(a.capture,map_location='cpu',weights_only=True)
for k,v in data.items():
    if isinstance(v,torch.Tensor):
        f=v.float();print('INPUT',k,str(v.dtype),list(v.shape),'range',float(f.min()) if f.numel() else None,float(f.max()) if f.numel() else None,'finite',bool(torch.isfinite(f).all()),flush=True)

def checkpoint(label):
    if not a.trace:return
    print('SYNC BEFORE',label,flush=True)
    if a.device!='cpu':torch.vacc.synchronize()
    print('SYNC AFTER',label,flush=True)

src=Path(a.source).read_text()
tree=ast.parse(src)
class Instrument(ast.NodeTransformer):
    def visit_FunctionDef(self,node):
        if node.name not in ('_chunk_core','_inv_unit_lower'):return node
        def block(body):
            out=[]
            for s in body:
                if isinstance(s,(ast.If,ast.For,ast.While)):
                    s.body=block(s.body);s.orelse=block(s.orelse)
                out.append(s)
                if isinstance(s,(ast.Assign,ast.AugAssign)):
                    call=ast.parse(f'_checkpoint({node.name+":"+str(s.lineno)!r})').body[0]
                    out.append(ast.copy_location(call,s))
            return out
        node.body=block(node.body);return node
if a.trace:tree=Instrument().visit(tree)
ast.fix_missing_locations(tree)
mod=types.ModuleType('captured_gdn');exec(compile(tree,a.source,'exec'),mod.__dict__)
mod._sync=lambda _:None
mod._checkpoint=checkpoint
args={k:v.to(a.device) for k,v in data.items() if k in ['q','k','v','g','beta','initial_state']}
args.update(cu_seqlens=data['cu_seqlens'],output_final_state=True,use_qk_l2norm_in_kernel=True)
if a.bf16_default: torch.set_default_dtype(torch.bfloat16)
print('DEFAULT_DTYPE',str(torch.get_default_dtype()),flush=True)
if a.custom_op:
    @torch.library.custom_op('repair::gdn',mutates_args=())
    def custom(q:torch.Tensor,k:torch.Tensor,v:torch.Tensor,g:torch.Tensor,beta:torch.Tensor,initial_state:torch.Tensor,cu_seqlens:torch.Tensor)->tuple[torch.Tensor,torch.Tensor]:
        return mod.chunk_gated_delta_rule_fast(q,k,v,g,beta,initial_state=initial_state,cu_seqlens=cu_seqlens,output_final_state=True,use_qk_l2norm_in_kernel=True)
    fn=lambda:custom(**{k:v for k,v in args.items() if isinstance(v,torch.Tensor)})
else:
    fn=lambda:mod.chunk_gated_delta_rule_fast(**args)
start=time.monotonic()
with torch.inference_mode(): o,s=fn()
checkpoint('return')
for label,t in [('output',o),('state',s)]:
    c=t.float().cpu()
    print('RESULT',label,'dtype',str(t.dtype),'finite',bool(torch.isfinite(c).all()),'maxabs',float(c.abs().max()),flush=True)
    assert torch.isfinite(c).all(),f'{label} nonfinite on captured real input'
print('REPLAY_PASS',time.monotonic()-start,flush=True)
