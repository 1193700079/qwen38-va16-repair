from pathlib import Path
import shutil
P=Path('/opt/venv/lib/python3.12/site-packages')
shutil.copyfile('/work/recurrent_compat.py',P/'recurrent_compat.py')
p=P/'vllm/model_executor/models/qwen3_next.py'
s=p.read_text();assert 'REPAIR_RECURRENT_20261125' not in s
p.write_text(s+'\n# REPAIR_RECURRENT_20261125\nfrom recurrent_compat import fused_recurrent_gated_delta_rule_compat as fused_recurrent_gated_delta_rule\n')
print('PATCH_RECURRENT_APPLIED',flush=True)
