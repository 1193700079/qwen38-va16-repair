from pathlib import Path
import shutil
p = Path('/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/attention/mm_encoder_attention.py')
s = p.read_text()
assert 'REPAIR_MM_20261125' not in s
shutil.copyfile('/work/mm_attention_compat.py', '/opt/venv/lib/python3.12/site-packages/mm_attention_compat.py')
p.write_text(s + '\n# REPAIR_MM_20261125\nfrom mm_attention_compat import forward_mm_compat\nMMEncoderAttention.forward = forward_mm_compat\nMMEncoderAttention.forward_native = forward_mm_compat\n')
q = Path('/opt/venv/lib/python3.12/site-packages/vllm_vacc/vllm/model_executor/models/qwen3_vl.py')
t = q.read_text()
assert 'REPAIR_MM_MERGER_20261125' not in t
q.write_text(t + '\n# REPAIR_MM_MERGER_20261125\nfrom mm_attention_compat import forward_merger_compat, forward_vmlp_compat\nQwen3_VisionPatchMerger.forward = forward_merger_compat\nQwen3_VisionMLP.forward = forward_vmlp_compat\n')
print('PATCH_MM_APPLIED', flush=True)
