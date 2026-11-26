#!/usr/bin/env bash
# Verified text inference candidate. Keeps original model weights read-only.
# Loopback only; use ssh -L 18080:127.0.0.1:18080 test@10.1.2.136.
set -euo pipefail
name=qwen38-va16-repaired
image=qwen38-va16-repaired:20261125-r3
if docker container inspect "$name" >/dev/null 2>&1; then
    echo "Container $name already exists; inspect it before changing it." >&2
    exit 1
fi
exec docker run -d --name "$name" --restart unless-stopped \
  --privileged -v /dev:/dev -v /home/test/models:/models:ro \
  --shm-size=32g -e VACC_VISIBLE_DEVICES=0,1,2,3 \
  -e GDN_FORCE_VCCL=0 -e GDN_STREAM_SYNC=0 -e GDN_MAX_CONCURRENCY=2 \
  -e PYTHONFAULTHANDLER=1 \
  -p 127.0.0.1:18080:18080 \
  "$image" vllm serve /models/Qwen3.8-27B \
  --served-model-name qwen3.8-27b --tensor-parallel-size 4 \
  --max-model-len 32768 --enforce-eager --no-disable-hybrid-kv-cache-manager \
  --generation-config /work/generation --middleware concurrency_gate.SerialInferenceGate \
  --host 0.0.0.0 --port 18080
