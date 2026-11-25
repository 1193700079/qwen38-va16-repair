import torch, torch_vacc, time, json, sys
def mark(s):
    print("STAGE", s, flush=True)
mark("import")
torch.utils.rename_privateuse1_backend("vacc")
torch.vacc.set_device(0)
mark("device")
out = {}
mark("api"); out["vacc_api"] = [n for n in dir(torch.vacc) if not n.startswith("_")]
mark("cq"); cq = getattr(torch.vacc, "custom_qwen3_ops", None)
out["custom_qwen3_ops"] = [n for n in dir(cq) if not n.startswith("_")] if cq is not None else None
mark("pre-argmax")
# device-side argmax / topk at full vocab
mark("argmax"); torch.manual_seed(7)
n = 248320
logits = torch.randn(1, n, dtype=torch.float32)
logits[0, n // 2] = 50.0
d = logits.to("vacc")
t = time.monotonic()
for _ in range(5):
    a = d.argmax(-1)
    torch.vacc.synchronize()
out["argmax_device_s_per_call"] = (time.monotonic() - t) / 5
out["argmax_device_correct"] = int(a.cpu()[0]) == n // 2
t = time.monotonic()
for _ in range(5):
    v, i = torch.topk(d, 5, dim=-1)
    torch.vacc.synchronize()
out["topk_device_s_per_call"] = (time.monotonic() - t) / 5
out["topk_device_correct"] = int(i.cpu()[0, 0]) == n // 2
mark("d2h")
# logits device->cpu transfer cost (current greedy path)
t = time.monotonic()
for _ in range(5):
    c = d.cpu()
    torch.vacc.synchronize()
out["logits_d2h_s_per_call"] = (time.monotonic() - t) / 5
mark("einsum")
# small-op launch overhead: 64 tiny einsums vs 1 batched
q = torch.randn(1, 12, 1, 128, device="vacc")
k = torch.randn(1, 12, 64, 128, device="vacc")
torch.vacc.synchronize()
t = time.monotonic()
for _ in range(64):
    o = torch.einsum("bhtk,bhsk->bhts", q, k[:, :, :1, :])
torch.vacc.synchronize()
out["tiny_einsum_64_s"] = time.monotonic() - t
t = time.monotonic()
o = torch.einsum("bhtk,bhsk->bhts", q, k)
torch.vacc.synchronize()
out["batched_einsum_64_s"] = time.monotonic() - t
mark("graph")
# graph capture surface
out["has_graph"] = [n for n in dir(torch.vacc) if "graph" in n.lower()]
out["has_stream_api"] = [n for n in dir(torch.vacc) if "stream" in n.lower()]
print(json.dumps(out, ensure_ascii=False, indent=1), flush=True)
mark("multinomial")
# multinomial on device (random sampling candidate) - segfault risk, run last
p = torch.softmax(torch.randn(1, 1000, device="vacc"), -1)
try:
    g = torch.Generator(device="vacc"); g.manual_seed(3)
    s = torch.multinomial(p, 4, replacement=False, generator=g)
    torch.vacc.synchronize()
    out["multinomial_device"] = s.cpu().tolist()
except Exception as e:
    out["multinomial_device"] = f"ERR {type(e).__name__}: {e}"
print(json.dumps(out, ensure_ascii=False, indent=1), flush=True)
