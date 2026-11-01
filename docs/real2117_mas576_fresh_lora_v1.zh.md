# Real 2117 + MAS 576：从原始 ASR 新建 LoRA

运行目录：`/data/qwen3-asr/records/therapist_mas_setting_d_v1/real2117_mas576_fresh_lora_b8g4_v1`。

## 数据与初始化

- 初始化：原始 `/data/qwen3-asr/models/Qwen3-ASR-1.7B`，新建 LoRA；不加载 M0 或第六轮 adapter。
- Setting D real：2117 条，66 位患者，14660.529 秒（4.072 小时）。
- Synthetic：完整复用 `therapist_synthetic96_loop6_v1` 六轮的 576 条；36 个新文本，51 位患者，3617.72 秒（1.005 小时）。不重新合成，不按 CER 过滤。
- 合计：2693 条、5.077 小时；synthetic 占条数 21.39%。原始 real 每条保留一次，按 Trainer 的 seed 42 随机采样。
- 已核对每条合成音频与 reference 的 SHA256，以及 reference 来自 real train；新文本身份键与 real train 和 merged feedback 不重叠。
- 合并训练清单 SHA256：`a83d406cab66c0dae2ec380b778fd1bd9ef2bedce833ef94a26a981753f79015`。

## 训练及评测

- baseline 同版本 utterance-mean 梯度累积修复路径；已核对 trainer 与官方训练模块哈希和 M0 训练时一致。
- batch 8、grad_acc 4、lr 2e-4、linear schedule、warmup 0.02、seed 42。
- LoRA r16、alpha4、dropout0.05；q/k/v/o/gate/up/down projections。
- 最大 140 steps，每 10 steps 保存模型并 decoded 评测，workers 2、prefetch 2。
- 至少保留 step 50 对照；之后若连续三次 merged 句级平均 CER 高于历史最佳至少 0.02，则停止。否则跑到 140。
- 只保存模型，不保存 optimizer 状态；不影响训练中的优化器更新。
- 沿用合并的 511 条真实语音作为反馈集。解码沿用原两个分片的 batch 方式，再合并计算；不再单独选择旧 dev。
- 同时报告句级平均 CER、pooled CER、patient macro 及 easy/medium/hard。对照包含 real-only M0 step50、zero-shot 和六轮连续纯 synthetic 终点。
- `step50_comparison.json` 是固定 step 对照；最终 best 是本次 merged-feedback 选出的最优点，和历史 M0 的选择过程不等价，单独标注。
- 这批 MAS 文本来自同一反馈集驱动的历史闭环；本次不将该反馈集描述成未使用的独立测试。

## 执行

入口 `examples/train_real_mas576.py`；运行状态在 `monitor.json`，日志在 `train.log`，完成后写 `run_summary.json`。
启动前针对新初始化、step50 保留、停止规则及既有梯度路径的 8 项测试通过。
