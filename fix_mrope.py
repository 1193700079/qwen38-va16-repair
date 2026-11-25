from pathlib import Path
import shutil
p=Path('/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/rotary_embedding/mrope.py')
s=p.read_text()
assert 'REPAIR_MROPE_20261125' not in s
shutil.copyfile('/work/mrope_compat.py','/opt/venv/lib/python3.12/site-packages/mrope_compat.py')
p.write_text(s+'\n# REPAIR_MROPE_20261125\nfrom mrope_compat import forward_mrope_compat\nMRotaryEmbedding.forward = forward_mrope_compat\n')
print('PATCH_MROPE_APPLIED',flush=True)
