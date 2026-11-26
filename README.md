# Qwen3.8-27B / VA16 可复现兼容修复

状态：文本贪心 + 随机采样（seed 可复现）+ 并发 2（含混合批）+ **32768 上下文** + **图像多模态**已通过；不是全功能/高吞吐服务。权重仍为原版 BF16。decode 约 0.25 tok/s@32768 / 0.42 tok/s@8192（厂商逐算子开销；无 graph capture/融合 GDN decode kernel，纯 Python 微优化无效；上下文与速度权衡）。1M 上下文不可行（rope 原生 262144）。

完整根因、验收和性能见[修复报告](<../qwen38_vastai_final_report.md>)。

## 部署

目标服务器：`10.1.2.136`。目前运行容器 `qwen38-va16-repaired`，镜像 `qwen38-va16-repaired:20261125-r2`（`sha256:66af7d8f6389…`）。
只监听服务器 `127.0.0.1:18080`，用 SSH 转发访问，模型 ID `qwen3.8-27b`。

```bash
# Mac：先连接 UniVPN；此进程保持运行。
ssh -N -L 18080:127.0.0.1:18080 test@10.1.2.136
```

```bash
# Mac 的另一个终端
curl http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"7加5是多少？只输出数字。"}],"temperature":0,"max_tokens":16,"chat_template_kwargs":{"enable_thinking":false}}'
```

在服务器管理：

```bash
docker logs --tail 100 qwen38-va16-repaired
docker restart qwen38-va16-repaired
docker stop qwen38-va16-repaired
```

重新创建服务前先检查原容器，不要删除其他容器或整机重载驱动。
[启动脚本](<start_qwen38.sh>)遇到同名容器会停止，不覆盖它。

## 构建边界

[Dockerfile](<Dockerfile>)基于现场旧补丁镜像 `qwen38_gdnfast:latest`，基线 ID 为：

```
sha256:a1fc47c3bb81da03817f467d2b9a4f379379583d5600d1e160d3c07b64000009
```

该基线必须已存在。它包含之前的模型加载/hybrid 元数据补丁，不能直接替换为一个同名但不同内容的镜像。
服务器上的完整构建目录为 `/home/test/qwen-repair-20261125/build-final`。

```bash
# 在目标服务器已有基线的条件下
cd /home/test/qwen-repair-20261125/build-final
docker build --pull=false -t qwen38-va16-repaired:20261125 .
```

最终镜像 ID：

```
sha256:a1d8b0e5f6fc93978a85989fc433dc67ba838d64d9213e780ab6f05e60bf187b
```

## 修复组件

| 文件 | 作用 |
|---|---|
| [fix_cpu_mask.py](<fix_cpu_mask.py>) | 以逐槽 zero_ 替代有故障的 CPU 布尔掩码设备清零 |
| [chunk_compat.py](<chunk_compat.py>) | 稳定 GDN 分块 prefill，FP32 状态，空序列语义 |
| [recurrent_compat.py](<recurrent_compat.py>) | 一维槽号 decode、FP32 累积；未验证 speculative 模式明确拒绝 |
| [mrope_compat.py](<mrope_compat.py>) | VACC 拆分缓存与三轴部分旋转适配 |
| [attention_compat.py](<attention_compat.py>) | 按真实页大小和 block IDs 做 decode，不假设固定 8K 组 |
| [sampler_compat.py](<sampler_compat.py>) | 全词表正确 greedy argmax、token-ID/logprobs 契约 |
| [concurrency_gate.py](<concurrency_gate.py>) | 请求排队，覆盖完整流式响应；未验收模式在引擎前拒绝 |

`GDN_FORCE_VCCL=0`、`GDN_STREAM_SYNC=0`：禁用早先未经证明的全局通信/同步 monkeypatch，使用厂商原行为。

## 测试

CPU 数学与请求契约：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest \
  test_chunk_contracts.py test_recurrent_contracts.py \
  test_concurrency_gate.py test_request_contract.py -q
```

实测 34 项通过。硬件回归日志与端到端回答在 [evidence](<evidence/>)；关键验收：

- [短题和代码生成](<evidence/short-acceptance.json>)
- [1203-token 跨页检索](<evidence/long-acceptance.json>)
- [固化服务默认值与排队请求](<evidence/final-acceptance.json>)
- [重启后生成](<evidence/restart-acceptance.json>)
- [增量解码概率对照](<evidence/logit-parity.json>)
- [18 分片 SHA-256](<evidence/model-hash-verification.json>)

## 必须保留的限制

- 文本 + `temperature=0`；API 拒绝随机采样、多模态及结构化约束模式。
- 串行执行；同时到达的请求排队，不混合 prefill/decode 批次。
- 上下文上限 4096；已实测 1203-token 跨页输入，未验证整个原生超长上下文。
- 目前约 0.4–0.5 token/s，属于正确性优先兼容方案，后续需继续优化融合和调度。
- 单个算子的 CPU/设备误差界不等于整模型 CUDA bitwise 一致；没有做完整标准 benchmark。
