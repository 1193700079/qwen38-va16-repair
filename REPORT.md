# Qwen3.8-27B / 瀚博 VA16 修复验收报告

更新：2026-11-25。**本报告替代此前“TP4 死锁无法在 Python 层修复”的结论。**

## 验收结论

原版 Qwen/Qwen3.8-27B 已在 10.1.2.136 上完成真实文本生成，并以独立镜像部署。未更换基座、未量化、未修改模型权重。

已验证范围：BF16 原版权重、TP4（4 个计算 die，即 1 张 VA16）、文本输入、temperature=0、串行/排队请求。不是全功能或高吞吐生产认证。

## 已部署服务

- 容器：`qwen38-va16-repaired`
- 镜像：`qwen38-va16-repaired:20261125`
- 镜像 ID：`sha256:a1d8b0e5f6fc93978a85989fc433dc67ba838d64d9213e780ab6f05e60bf187b`
- 服务端地址：`http://127.0.0.1:18080/v1`
- 模型 ID：`qwen3.8-27b`
- 上下文上限：4096（保守限制；没有宣称原生 262K 全部可用）
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
- 请求层：完整流式响应期间串行化，避免进入尚未验证的混合 prefill/decode 调度；不支持的模式在进入引擎前返回明确 400。

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
| temperature=0.7 | HTTP 400 | 在引擎前明确拒绝 |

增量解码与 fresh prefill 对照：测试中 top token 一致；`1+1` 首 token logprob 一致，第二个结束 token logprob 约 -3.10e-6 / -3.34e-6。仅为所测样例的数值一致性证据，不代表完整 benchmark。

## 性能与限制（重要）

- 当前是兼容实现，仍慢：短题 3–5 个生成 token 约 10–18 秒；25-token 代码回答约 63 秒；约 0.4–0.5 token/s 的长回答解码速度。
- 1203-token 长输入找数字约 39 秒。不能把独立小算子的 0.05 秒当作模型端到端速度。
- 当前只验收文本、贪心采样；随机采样、图片/视频/音频输入、结构化工具约束输出会被拒绝。
- 请求可同时提交，但服务串行处理，不是并发计算。
- 不支持 speculative decode；没有启用不明后端的 torch.compile。
- 未验证 4096-token 极限、原生超长上下文、多机并行或完整多模态能力。
- 构建依赖服务器已保留的 `qwen38_gdnfast:latest` 基线镜像，基线 ID 与补丁源哈希均记录；不是无依赖通用安装包。

## 交付文件

- [修复源码与部署说明](<repair-20261125/README.md>)
- [构建文件](<repair-20261125/Dockerfile>)
- [运行脚本](<repair-20261125/start_qwen38.sh>)
- [测试与运行证据目录](<repair-20261125/evidence/>)

最终 service health 和 `/v1/models` 均为 HTTP 200；当前 `max_model_len=4096`，只监听 127.0.0.1:18080。
