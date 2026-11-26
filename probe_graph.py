import torch, torch_vacc
torch.utils.rename_privateuse1_backend("vacc")
ga = [a for a in dir(torch.vacc) if any(k in a.lower() for k in ("graph", "capture", "replay"))]
tg = [a for a in dir(torch) if "Graph" in a]
print("vacc_graph_attrs", ga)
print("torch_graph_attrs", tg)
try:
    import torch_vacc.vacc as v
    print("vacc_mod_graph", [a for a in dir(v) if any(k in a.lower() for k in ("graph", "capture", "replay"))])
except Exception as e:
    print("err", e)
