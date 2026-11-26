# Qwen3.8-27B / 瀚博 VA16 修复验收报告

更新：2026-11-26（r3）。**本报告替代此前“TP4 死锁无法在 Python 层修复”的结论。**

## 验收结论

原版 Qwen/Qwen3.8-27B 已在 10.1.2.136 上完成真实文本生成**与图像多模态理解**，并以独立镜像部署。未更换基座、未量化、未修改模型权重。

已验证范围：BF16 原版权重、TP4（4 个计算 die，即 1 张 VA16）、文本输入、**图像输入（多模态）**、temperature=0 贪心 **与 temperature>0 随机采样（seed 可复现）**、**并发 2（含 prefill+decode 混合批）**、**32768 上下文（12036-token 长文召回正确）**、串行/排队请求。不是全功能或高吞吐生产认证。

## 已部署服务

- 容器：`qwen38-va16-repaired`
- 镜像：`qwen38-va16-repaired:20261125-r3`
- 镜像 ID：`sha256:2714fc6c715c4c468e9586297344f00754c18b65d71170fdcfa492a1f55b02e7`
- 服务端地址：`http://127.0.0.1:18080/v1`
- 模型 ID：`qwen3.8-27b`
- 上下文上限：**32768**（r3 实测 12036-token 长文召回正确；厂商 block_table 除零已修，见修复）
- 并发：`GDN_MAX_CONCURRENCY=2`（asyncio 信号量；混合批 attention 已兼容）
- 多模态：**图像输入已启用并验证**（纯红/纯蓝 PNG 均正确识别）；音频/视频仍拒绝
- 计算设备：`VACC_VISIBLE_DEVICES=0,1,2,3`
- 重启策略：`unless-stopped`；已执行真实容器重启并重新生成正确答案。
- 权重路径：`/home/test/models/Qwen3.8-27B`，容器内只读挂载。
- 部署及测试目录：`/home/test/qwen-repair-20261125/`

端口只绑定服务器 loopback，没有公开到互联网。Mac 上先连接 UniVPN，再建立 SSH 隧道：

```bash
ssh -N -L 18080:127.0.0.1:18080 test@10.1.2.136
```

另一个终端测试：

```bash
curl http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"9乘9等于多少？只输出数字。"}],"temperature":0,"max_tokens":16,"chat_template_kwargs":{"enable_thinking":false}}'
```

查看/停止/启动（在服务器上）：

```bash
docker logs --tail 100 qwen38-va16-repaired
docker stop qwen38-va16-repaired
docker start qwen38-va16-repaired
vasmi list
```

不要直接删除原镜像或整机重载驱动。历史试验容器已停止，未操作无关服务。

## 证据与关键修复

### 1. 原先“死锁根因”判断纠正

原生栈停在同步函数不能证明该函数是根因。逐边界同步、真实首层张量捕获、最小复现后确认：

```python
initial_state[~has_initial_state, ...] = 0
```

其中布尔掩码在 CPU，状态在 VACC。此操作会使随后设备操作超时。独立复现不需要 TP4 或模型；同语义逐槽 `zero_()` 后回归通过。之前将问题归咎为“不可修复的缓存写/通信死锁”证据不足，现明确撤回。

同时实测 SSM 缓存为 **FP32**，此前从配置推断为 BF16 也不正确。

### 2. 本轮修复

- CPU 布尔掩码清零：逐槽清零，保持缓存数学语义；RED 超时、GREEN 正确结果。
- GDN prefill：稳定分块三角递推，替代数值不稳的平坦 Newton–Schulz；返回 FP32 最终状态，保留空序列初态。
- GDN decode：支持实际一维状态索引，保持 FP32 累积及状态，独立 FP64 参考与设备测试通过。
- MRoPE：适配厂商拆分 cos/sin 缓存及上游 CustomOp 初始化差异；保留三轴、交错顺序、部分旋转，文本及三轴设备回归通过。
- attention decode：依据实际 784-token 物理缓存页及原始 block IDs 读取，不再错误重塑为固定 8192-token 组；跨页及乱序物理页测试通过。
- **完整词表贪心采样**：原 `sampler_v1` 在 248320 词表上，最高分人为设为 token 124160，却返回 117251；小词表正常。修复为 CPU FP32 argmax（仅 logits 向量，不迁移模型计算），返回厂商 int32 ABI。
- logprobs：修复厂商 int32 token ID 与上游 int64 契约冲突，输出概率和排名通过回归。
- 请求层（r1）：完整流式响应期间串行化；不支持的模式在进入引擎前返回明确 400。
- **r2 增量**：随机采样（CPU multinomial，temperature/top_k/top_p，seed 可复现，设备回归 PASS）；并发上限 2（asyncio 信号量）；attention 兼容统一 causal 分块（QCH=128）覆盖 prefill+decode 混合批；上下文 4096→8192（5493-token 长文召回正确）。r2 验收与重启复验全部通过（见 evidence/acceptance-r2.json、restart-verify-r2.json）。
- **r3 增量（上下文 + 多模态）**：
  - **上下文 8192→32768**：厂商 `BLOCK_GROUP_SIZE`(默认8192) 除以各 KV 组 block_size 得组因子；mamba/state 组 block_size=max_model_len，>8192 时组因子=0 触发 `ZeroDivisionError`（block_table.py:67、single_type_kv_cache_manager.py:46、kv_cache_utils.py:278/315、vacc_model_runner.py:243）。修复为对组因子统一 `max(1, env//block_size)`（state 组每请求一个 state 块，组因子=1 语义正确；attention 组不变；不改 env 以免破坏 vendor prefill 分组）。12036-token 长文召回正确。
  - **多模态图像启用**：厂商 ViT 路径在 VACC 上多处 segfault/契约错误，逐层替换为纯 torch 兼容实现——① `MMEncoderAttention` 的 torch-SDPA fallback segfault → 显式 bmm 全注意力（含 varlen/GQA）；② `patch_merger_vision` 融合算子 segfault（faulthandler 定位 custom_ops.py:3238）→ 纯 torch norm+fc1+act+fc2；③ `fuse_mlp_vision` 同样 segfault → 纯 torch；④ 视觉 linear 为 plain nn.Linear 返回 Tensor 而非 2-tuple → `_lin` 兼容。请求守卫放开 image_url（仍拒 audio/video）。纯红/纯蓝 PNG 端到端识别正确（证明真实视觉而非记忆）。

此前模型注册、tuple 权重加载及 hybrid-cache 等兼容补丁保留在基线镜像中。

## 验收结果

### 权重与算子

- 18 个 safetensors 分片逐文件 SHA-256 与 ModelScope 元数据一致。
- 本地 CPU 回归：34 项通过（GDN FP64 数学参考、状态契约、API 串行化及模式守卫）。
- VACC 设备回归：缓存清零、MRoPE、GDN prefill/decode、真实卷积状态布局、KV 精确写入、跨页 attention、完整词表贪心采样、logprobs 通过。
- GDN 设备精度样例：真实第一层输出最大绝对误差约 0.0062、FP32 最终状态约 0.0076；BF16 输出容差按测试设定验收，不声称与 CUDA/FLA bitwise 一致。

### 真实模型回答

| 场景 | 返回 | 结果 |
|---|---|---|
| 两水管分别 6h/4h，合开时间 | `12/5` | 正确，正常 stop |
| Python 两整数相加函数 | 完整 `def add(a: int, b: int) -> int: return a + b` | 正确，正常 stop |
| 新请求 7+5 | `12` | 正确 |
| 1203-token 档案找校验码 | `4729` | 正确，跨越 784-token 缓存页 |
| 两请求同时到达 | 分别 `12`、`81` | 排队执行正确 |
| 固化镜像重启后 9×9 | `81` | 正确 |
| 默认采样不传 temperature | `2` | 使用已验证贪心默认 |
| temperature=0.7, seed=42 两次 | 同一段中文描写 | **随机采样可用且 seed 可复现**（r2） |
| 并发 2（12*5 与 9*9 交错到达） | `60` / `81` | **混合 prefill+decode 批正确**（r2） |
| 5493-token 长文找校验码 | `4729` | **8192 上下文正确**（r2） |
| r2 容器重启后 greedy/随机/并发 | `2` / same=true / `60`,`81` | 重启后全部复验通过 |
| 纯红 64×64 PNG 问颜色 | `红色` | **图像多模态正确**（r3） |
| 纯蓝 64×64 PNG 问颜色 | `蓝色` | **真实视觉而非记忆**（r3） |
| 12036-token 长文找校验码 | `4729` | **32768 上下文正确**（r3） |
| r3  clean 重启后 mm/贪心/长文 | 红色/蓝色/`12`/`4729` | 重启后复验通过（evidence/final2.log、verify-r3.log） |

增量解码与 fresh prefill 对照：测试中 top token 一致；`1+1` 首 token logprob 一致，第二个结束 token logprob 约 -3.10e-6 / -3.34e-6。仅为所测样例的数值一致性证据，不代表完整 benchmark。

## 性能与限制（重要）

- 当前是兼容实现，仍慢：短题 3–5 个生成 token 约 10–18 秒。**解码速度随上下文上限变化（权衡）**：r2@8192 约 **0.42 token/s**（48 token/114.5s，evidence/acceptance-r2.json）；r3@32768 约 **0.25 token/s**（48 token/189.4s，evidence/r4-restart-perf.log）。增大 max-model-len 会放大混合缓存/调度开销，decode 变慢；如需更快解码可回退 `--max-model-len 8192`。
- **解码慢的根因（probe_perf_surface.py 实测）**：VACC 单算子 launch 开销约 0.9 ms（64 次 tiny einsum=0.056 s），批量化同操作仅 0.0038 s；decode 每 token 逐层逐算子发射，无融合 GDN decode kernel、无 graph capture（`has_graph=[]`）、PrivateUse1 后端无 torch.compile。device 端全词表 argmax 0.054 s/call 反而比 logits d2h(0.00065 s)+CPU argmax 慢，故采样保留 CPU。**prefill 已向量化分块**（5493-token 长文可行）；decode 吞吐受厂商运行时逐算子开销限制，属厂商侧优化空间。
- 5493-token 长输入找数字约 119 秒（8192 上下文）。不能把独立小算子的 0.05 秒当作模型端到端速度。
- **随机采样已支持**（temperature/top_k/top_p，CPU multinomial，seed 可复现）；**并发 2 已支持**（含 prefill+decode 混合批）；**图像多模态已支持并验证**（r3）。音频/视频输入、结构化工具约束输出仍被拒绝。
- **上下文上限 32768（r3）**：厂商 block_table 组因子除零已修（`max(1, env//block_size)`）。**1M 上下文不可行**：模型原生 rope 上限 `max_position_embeddings=262144`（config.json），远超即超出模型训练位置编码范围；且 KV/state 显存随长度线性增长，单 die 31.36GB 无法容纳 1M token 的混合缓存。32768 为当前显存/稳定性权衡下的验证上限，可再向上试探但非 1M。
- **解码提速尝试与结论（r3）**：einsum 递推基线 r2@8192≈0.42、r3@32768≈0.25 token/s。尝试将 GDN decode 递推的 einsum 改为显式 3D bmm 以减少 kernel 数，实测**更慢**（48 token/186.7s=0.257 token/s），已回退 einsum。**graph capture 确证不可用**：`torch.vacc`/`torch_vacc.vacc` 无任何 graph/capture/replay API（probe_graph.py：`vacc_graph_attrs=[]`），`torch.CUDAGraph` 仅 CUDA 后端；厂商亦无融合 GDN decode kernel（custom_qwen3_ops 仅 MoE）。故 decode 吞吐受厂商运行时逐算子 launch 开销（约0.9ms/op）限制，纯 Python 层微优化无法突破，属厂商侧优化空间。**>0.42 token/s 的目标在本厂商运行时上未达成**（且 32768 上下文进一步降速）。
- 不支持 speculative decode；没有启用不明后端的 torch.compile。
- 未验证多机并行、音频/视频多模态、>32768 上下文与 1M 上下文（不可行，见上）。
- 构建依赖服务器已保留的 `qwen38_gdnfast:latest` 基线镜像，基线 ID 与补丁源哈希均记录；不是无依赖通用安装包。

## 交付文件

- [修复源码与部署说明](<repair-20261125/README.md>)
- [构建文件](<repair-20261125/Dockerfile>)
- [运行脚本](<repair-20261125/start_qwen38.sh>)
- [测试与运行证据目录](<repair-20261125/evidence/>)

最终 service health 和 `/v1/models` 均为 HTTP 200；当前 `max_model_len=32768`、`GDN_MAX_CONCURRENCY=2`、图像多模态已启用，只监听 127.0.0.1:18080。
