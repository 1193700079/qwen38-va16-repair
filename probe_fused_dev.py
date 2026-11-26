import time, torch, torch_vacc
torch.utils.rename_privateuse1_backend("vacc")
torch.vacc.set_device(0)
import torch_vacc.vacc as v

def t(fn, n=200):
    fn(); torch.vacc.synchronize()
    a = time.time()
    for _ in range(n): fn()
    torch.vacc.synchronize()
    return (time.time() - a) / n * 1e3  # ms/op

B, H = 1, 5120
x = torch.randn(B, H, device="vacc", dtype=torch.bfloat16)
w = torch.randn(H, device="vacc", dtype=torch.bfloat16)
res = torch.randn(B, H, device="vacc", dtype=torch.bfloat16)

# correctness: fused_residual_rmsnorm vs manual
out = torch.empty_like(x); resout = torch.empty_like(x)
v.fused_residual_rmsnorm(x, w, residual=res, output=out, residual_out=resout)
torch.vacc.synchronize()
man = (res + x)
norm = (man / torch.sqrt((man.float()**2).mean(-1, keepdim=True) + 1e-6)).to(torch.bfloat16) * w
print("fused_residual_rmsnorm maxerr", (out - norm).abs().max().item())
print("residual_out maxerr", (resout - man).abs().max().item())
print("fused_residual_rmsnorm ms/op", round(t(lambda: v.fused_residual_rmsnorm(x, w, residual=res, output=out, residual_out=resout)), 4))
print("manual add+rmsnorm ms/op", round(t(lambda: ((res + x) / torch.sqrt(((res + x).float()**2).mean(-1, keepdim=True) + 1e-6)).to(torch.bfloat16) * w), 4))

# swiglu
y = torch.randn(1, 256, 2 * 4096, device="vacc", dtype=torch.bfloat16)
sg = v.swiglu(y)
import torch.nn.functional as F
a, b = torch.chunk(y, 2, dim=-1)
man_sg = F.silu(a) * b
torch.vacc.synchronize()
print("swiglu maxerr", (sg - man_sg).abs().max().item())
print("swiglu ms/op", round(t(lambda: v.swiglu(y)), 4))
print("manual swiglu ms/op", round(t(lambda: F.silu(a) * b), 4))
