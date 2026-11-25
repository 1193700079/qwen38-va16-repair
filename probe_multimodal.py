import importlib, inspect, json
out = {}
try:
    from vllm.model_executor.models.registry import ModelRegistry
    archs = ["Qwen3_5ForConditionalGeneration"]
    for a in archs:
        try:
            cls = ModelRegistry.resolve_model_cls(a)
            if isinstance(cls, tuple): cls = cls[0]
            out[a] = {
                "supports_mm": getattr(cls, "supports_multimodal", None),
                "has_visual": any(k in dict(inspect.getmembers(cls)) for k in ("visual", "vision_tower")),
                "module": cls.__module__,
            }
        except Exception as e:
            out[a] = {"err": str(e)}
except Exception as e:
    out["registry_err"] = str(e)
# vision ops present?
try:
    import torch, torch_vacc
    ops = [o for o in dir(torch.vacc) if any(k in o for k in ("vit", "vision", "img", "qwenvl"))]
    out["vacc_vision_ops"] = ops
    cq = [o for o in dir(torch.vacc.custom_qwen3_ops) if any(k in o for k in ("vit", "vision", "img", "qwenvl"))]
    out["custom_qwen3_vision_ops"] = cq
except Exception as e:
    out["ops_err"] = str(e)
print(json.dumps(out, ensure_ascii=False, default=str))
