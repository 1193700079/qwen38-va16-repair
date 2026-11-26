import inspect
import torch, torch_vacc
torch.utils.rename_privateuse1_backend("vacc")
torch.vacc.set_device(0)
import torch_vacc.vacc as v
for name in ["fused_residual_rmsnorm", "fused_mlp_with_rmsnorm", "swiglu",
             "fuse_atten_qwen3", "conv1d", "conv1d_glu", "topk_topp", "fused_mla",
             "custom_qwen3_ops"]:
    f = getattr(v, name, None)
    if f is None:
        print(name, "MISSING")
        continue
    if isinstance(f, dict):
        for k, vv in f.items():
            try:
                print("dict.", k, str(inspect.signature(vv))[:200])
            except Exception:
                print("dict.", k, "sig-err")
        continue
    try:
        print(name, str(inspect.signature(f))[:220])
    except Exception as e:
        print(name, "sig-err", e)
    doc = (f.__doc__ or "")[:400]
    if doc.strip():
        print("   DOC:", doc.replace("\n", " | ")[:400])
