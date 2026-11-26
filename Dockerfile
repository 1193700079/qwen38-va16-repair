# Baseline digest is captured in baseline.json; this tag contains the previously
# verified model-registration/loader/hybrid-cache and reference-kernel adapters.
FROM qwen38_gdnfast:latest
COPY sources/server_gdn_vacc_fast.py /opt/venv/lib/python3.12/site-packages/gdn_vacc_fast.py
COPY fix_cpu_mask.py fix_mrope.py fix_recurrent.py fix_chunk_attention.py fix_sampler.py fix_mm.py mrope_compat.py recurrent_compat.py chunk_compat.py attention_compat.py sampler_compat.py mm_attention_compat.py /work/
RUN python3 /work/fix_cpu_mask.py \
 && python3 /work/fix_mrope.py \
 && python3 /work/fix_recurrent.py \
 && python3 /work/fix_chunk_attention.py \
 && python3 /work/fix_sampler.py \
 && python3 /work/fix_mm.py \
 && python3 -m compileall -q /opt/venv/lib/python3.12/site-packages/mrope_compat.py /opt/venv/lib/python3.12/site-packages/recurrent_compat.py /opt/venv/lib/python3.12/site-packages/chunk_compat.py /opt/venv/lib/python3.12/site-packages/attention_compat.py
COPY generation_config.json /work/generation/generation_config.json
COPY concurrency_gate.py /opt/venv/lib/python3.12/site-packages/concurrency_gate.py
COPY patch_blocktable.py /work/patch_blocktable.py
RUN python3 /work/patch_blocktable.py
ENV GDN_FORCE_VCCL=0 GDN_STREAM_SYNC=0
ENTRYPOINT ["/vacc/entrypoint.sh"]
CMD ["/bin/bash"]
