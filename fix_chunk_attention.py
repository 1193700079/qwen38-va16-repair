from pathlib import Path
import shutil
P=Path('/opt/venv/lib/python3.12/site-packages')
for name in ('chunk_compat.py','attention_compat.py'):
    shutil.copyfile('/work/'+name,P/name)
p=P/'vllm/model_executor/models/qwen3_next.py'
s=p.read_text();assert 'REPAIR_CHUNK_20261125' not in s
p.write_text(s+'\n# REPAIR_CHUNK_20261125\nfrom chunk_compat import chunk_gated_delta_rule_compat as fla_chunk_gated_delta_rule\n')
p=P/'vllm_vacc/vllm/v1/attention/backends/vacc_attn.py'
s=p.read_text();assert 'REPAIR_ATTN_20261125' not in s
p.write_text(s+'\n# REPAIR_ATTN_20261125\nfrom attention_compat import install as _repair_install_attention\n_repair_install_attention()\n')
print('PATCH_CHUNK_ATTN_APPLIED',flush=True)
