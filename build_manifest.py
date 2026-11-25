from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parent
files=['chunk_compat.py','recurrent_compat.py','mrope_compat.py','attention_compat.py','sampler_compat.py','fix_cpu_mask.py','fix_mrope.py','fix_recurrent.py','fix_chunk_attention.py','fix_sampler.py','concurrency_gate.py','generation_config.json','start_qwen38.sh','sources/server_gdn_vacc_fast.py','Dockerfile']
out={'baseline_image':'qwen38_gdnfast:latest','baseline_image_id':'sha256:a1fc47c3bb81da03817f467d2b9a4f379379583d5600d1e160d3c07b64000009','files':{n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in files},'repaired_image':'qwen38-va16-repaired:20261125','repaired_image_id':'sha256:a1d8b0e5f6fc93978a85989fc433dc67ba838d64d9213e780ab6f05e60bf187b','status':'verified text/greedy compatibility deployment; restart, sequential requests and 1203-token recall passed; not full-feature certification','context_limit':4096,'not_supported':['stochastic sampling','multimodal input','speculative decode','mixed prefill/decode execution','structured/tool-constrained decoding']}
(root/'manifest.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
print(json.dumps(out,ensure_ascii=False,indent=2))
