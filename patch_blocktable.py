"""Fix vendor ZeroDivision for state/mamba KV groups whose page size
(= max_model_len) exceeds BLOCK_GROUP_SIZE (default 8192).

Root cause (measured with instrumented block_table): MultiGroupBlockTable builds
one BlockTable per KV group.
  attention group: block_size=784            -> env(8192)//784 = 10  (ok)
  mamba/state grp: block_size=max_model_len  -> 8192//16384 = 0 -> cdiv by 0.
The state group holds exactly one state block per request, so a group factor of 1
is correct. We clamp every `env_blk_grp_size // <block_size>` group factor to >=1
in the v1 KV path. BLOCK_GROUP_SIZE env is left untouched so vendor prefill
attention grouping (env//16 etc.) is unchanged.
"""
import re, sys

TARGETS = [
    "/opt/venv/lib/python3.12/site-packages/vllm_vacc/vllm/v1/worker/block_table.py",
    "/opt/venv/lib/python3.12/site-packages/vllm_vacc/vllm/v1/core/kv_cache_utils.py",
    "/opt/venv/lib/python3.12/site-packages/vllm_vacc/vllm/v1/core/single_type_kv_cache_manager.py",
    "/opt/venv/lib/python3.12/site-packages/vllm_vacc/vllm/v1/worker/vacc_model_runner.py",
]
# match any group-factor division:  env_blk_grp_size // <expr>  (assignment OR cdiv arg)
PAT = re.compile(r"(?<!max\(1, )env_blk_grp_size\s*//\s*([A-Za-z_][A-Za-z0-9_.]*)")
changed = 0
for p in TARGETS:
    try:
        s = open(p).read()
    except FileNotFoundError:
        continue
    if "max(1, env_blk_grp_size //" in s:
        continue
    new, n = PAT.subn(lambda m: "max(1, env_blk_grp_size // %s)" % m.group(1), s)
    if n:
        open(p, "w").write(new)
        changed += n
        print(f"patched {p}: {n} site(s)")
print("block_table/kv group-factor clamp done, sites=", changed)
