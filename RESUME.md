# 已验证检查点：2026-11-25

## 当前结果
- 原版 Qwen/Qwen3.8-27B BF16 在 10.1.2.136 VA16 上真实多 token 生成通过，未换权重。
- 活跃容器 `qwen38-va16-repaired`，镜像 `qwen38-va16-repaired:20261125-r2`。
- 镜像 ID `sha256:66af7d8f6389d2b62ad8f3b86f2683e3d8e1c1ca28cf808749465f27684166aa`。
- 本机只监听服务器 127.0.0.1:18080，TP4 用逻辑 die 0–3（非4张物理卡）。context cap **8192**，`GDN_MAX_CONCURRENCY=2`，restart unless-stopped。
- 已验证：18 shard SHA-256、34 CPU tests、多个硬件 RED/GREEN tests、水管题12/5、加法Python代码、全新请求12、1203-token跨页召回4729、并发到达排队的12/81、真正重启后81。
- **r2 增量已验证**：随机采样 seed=42 两次同文（CPU multinomial）；并发2交错 12*5=60 / 9*9=81（混合 prefill+decode 批）；5493-token 长文召回 4729（8192 上下文）；r2 重启后 greedy=2 / random same / 60,81 复验通过。证据 evidence/acceptance-r2.json、evidence/restart-verify-r2.json。
- 性能仍约 0.42 tok/s（decode 逐算子 launch ~0.9ms，无融合 GDN decode kernel/无 graph/无 torch.compile；prefill 已向量化）。>8192 触发厂商 block_table.py:67 ZeroDivisionError（BLOCK_GROUP_SIZE 默认8192）。多模态：模型 supports_multimodal=True + 厂商 ViT 融合算子在，但未端到端验证，API 仍拒绝图像。

## 必须纠正的历史结论
- 旧报告/全局记忆里“Python层无法修复”“TP4通信/缓存写死锁”均已被实际回归反证，不能据此停止或换模型。
- 真实故障：`initial_state[~has_initial_state,...]=0` 的 CPU bool mask 对 VACC 张量 index_put 污染后续操作；裸设备最小复现原写法timeout124，逐槽zero_通过且语义正确。**异步栈停在哪，不代表哪里是最先出错的操作。**
- 实际 SSM cache FP32，不是先前推断bf16；原fast将state返回bf16丢精度，新chunk/recurrent保留FP32。
- 厂商全词表sampler_v1 greedy可选错token（最高分124160，实际117251，词表248320）；新sampler走CPU argmax仅复制logits。错误多语言词不是模型能力差。
- MRoPE vendor Base ctor略过CustomOp导致_forward_method缺失，新adapter保留rotary数学已hardware oracle过。
- 全注意力decode按8192分组reshape不适合真实784token缓存页；新模块用raw page IDs。runtime原始map=[3,4,5,6]、slot2365=3*784+13、kc.shape[50,784,1,256]，无16/784映射冲突；recycler=None。

## 文件与运行
- 本地源码/证据位于本目录，说明见 README.md，报告在上级 qwen38_vastai_final_report.md。
- 服务器 `/home/test/qwen-repair-20261125/`，完整 build-final + Dockerfile + evidence。用户启动脚本 `/home/test/qwen-repair-20261125/start_qwen38.sh`。
- 构建依赖保留的 qwen38_gdnfast 基线镜像，不是从官方镜像零起点一条命令；不要删除旧镜像链。
- 自建所有测试容器以 `gdnrepair-` 开头，保留日志/镜像，不操作无关容器。
- SSH helper 在上级 repair_remote.exp，读取已有会话凭据，不把密码/token打印到报告。
- VPN偶发SSH超时可重试一次；路由若走en0而非utun则断隧道。UniVPN托盘AX菜单连接；若有连接断开modal，button 确定 在group1，必须关闭modal才重连。不要索要或导出浏览器cookie。

## 后续优化（非此次已完成承诺）
- 设备融合/减少同步以提高性能。
- 独立验收随机采样、真正批量并发、多模态/工具约束。
- 原生超长上下文未验证，不能只修改max-model-len宣称支持。
