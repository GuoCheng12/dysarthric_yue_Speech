# 学习进展驱动的患者课程 v2：准备协议

本轮只实现和验证，不启动正式 ASR 训练。入口为 `examples/real_curriculum_v2.py`；正式启动必须显式执行 `run`。

## 数据与三组

Setting D：2117 条 real、66 位患者、48 个 content。评测沿用合并后的 511 条语音，不恢复独立 dev；这是已反复使用的探索性反馈。train 与反馈的录音、音频路径、content、规范化文本/Jyutping hash 均隔离。没有 synthetic、没有 M0 权重初始化、没有新增真实数据。

三个分支 static / random / dynamic，全部从原始 Qwen3-ASR-1.7B 新建 LoRA。默认单卡执行顺序 static→random→dynamic；先完成 static 便于共用第一次处方。三组使用相同初始化配置和起步采样计划，各自训练、各自评测；不再要求不同运行的 adapter/optimizer/RNG 逐位相同，也不把相同 seed 视为评测结果相同。

2026-09-07 双卡执行安排：通过 `examples/launch_real_curriculum_v2.py --root NEW_ROOT --gpus COURSE_GPU RANDOM_GPU` 启动时，static 与 random 同时开始，static 完成后同卡启动 dynamic；另一张卡专跑 random。每个进程及其 evaluator 均通过 `CUDA_VISIBLE_DEVICES` 限定为单卡，因此仍为 batch8×累积4，不启用 DataParallel/DDP。用户已明确授权独立评测：random 不再等待 static 的 step20 评测，不读取其他分支的预测、loss 或状态凭证。此执行顺序覆盖记录在 `parallel_execution.json`。任一分支失败，主管进程停止仍运行的本实验分支；没有自动重试。

Reflection 引用接口修复后的单独重启：调用启动器的 `prepare_dynamic_restart(NEW_ROOT, PREVIOUS_ROOT)` 创建新记录目录，校验科学配置和公共输入未变，以符号链接保留已完成的 Static/Random，并原样复制第一次处方。重新完成 check/model-smoke 后，通过 `CUDA_VISIBLE_DEVICES=GPU_UUID .../launch_real_curriculum_v2.py --root NEW_ROOT --dynamic-only` 只从原始模型重跑 Dynamic。复用输入哈希记录在 `dynamic_restart.json`；不覆盖旧结果，不从缺少 optimizer/RNG 的 checkpoint 假装无损恢复。结束后沿用原报告逻辑汇总三组。

- 1–20 step：共同随机覆盖流，640 次曝光。
- 21–100：四个 20-step 阶段，各 400 次共同覆盖 + 240 次课程曝光。
- 每个有效 batch：20 coverage + 12 course，微 batch8、累积4。
- 所有 coverage 来自同一固定洗牌队列，完整走完 2117 条后才循环。共 2240 次 coverage，因此每组保证所有真实录音至少出现一次。
- 各组共 3200 次训练曝光，不宣称是 3200 条唯一录音或每条都练相同遍数。记录唯一录音数、患者/content 分布、实际音频时长和重复次数。
- 每阶段同一录音总曝光最多2次，包括 coverage 与 course。两路可能选中同一录音，计为两次；没有暗中去重造成配额变化。

random 的课程流随机选取；static 使用第一次处方冻结的候选池/配额，按累计曝光轮换，不读取新 CER 调整；dynamic 使用相同第一次处方，在 step40/60/80 自己的实际反馈后更新。static 与 dynamic 前40步的训练样本计划保持一致，但不要求模型/优化器/RNG 逐位相同。

## 固定训练配置

每组100个optimizer updates；LoRA r16、alpha4、dropout .05；q/k/v/o/gate/up/down 七种原有 projection suffix。LR峰值2e-4，warmup2步，按全局100步线性衰减。使用已修复的 utterance-mean accumulation-window loss。模型/优化器/学习率不由 Agent 控制。

一次连续 Trainer.train() 完成全程，不在阶段边界重建 optimizer 或重新开始 scheduler。使用单GPU、num_workers=0 的原生 DataLoader；不使用会提前取下一 batch 的 DataLoaderShard。当前阶段训练完才安装下一阶段的640个训练位置，读取尚未制定的未来课程即报错。

每阶段评估期间保存/恢复 Python、NumPy、Torch RNG，测 loss 时关闭 dropout，结束恢复训练模式。诊断前后核对 adapter/optimizer/scheduler/RNG 没有变化。保留每20步 adapter 与训练状态摘要，不复制基础模型或音频；不建立自动失败恢复/重试分支。

每组在 step20/40/60/80/100 均使用自己的 checkpoint 重新解码2117条训练语音及511条反馈语音，并独立测量 dropout-off loss。即使模型状态碰巧相同，也不复用另一组的评测。每份 metrics 文件记录自己的 checkpoint 路径。仍保留同一进程评测前后的状态一致性检查，确保评测不修改该模型、optimizer、scheduler 或 RNG；取消的只是跨实验状态相等要求。

## Evidence 与持久课程

每条匿名 Rxxxx 提供当前/上一阶段/base预测、CER、SDI、唯一 all-optimal 字符位置上的近期修复/退化/持续错误、曝光次数和 dropout-off loss。base→current 与 previous→current 分开命名。未测 loss 是 null，不是0。只有文字错误证据，不把 deletion 解释成某个声学 phone 缺失。

同时提供66位训练患者的聚合进展与各阶段全部content池的剩余容量，减少Agent对2117行自行心算；限制content后仍须按实际成员计算容量。

patient macro 与原 evaluator 一致：先计算每个患者的 pooled CER，再等权平均患者。prompt 数值保留6位小数，减少无意义的浮点token；本地评测产物保持全精度。

Agent 定义 Uxxx 单元：患者集合、训练 content 选择（空表示这些患者的全部训练 content）、目标错误或 maintenance、证据、预期变化。Harness 把患者×content选择落到真实录音池；至少两个content；激活单元池不重叠。定义一旦注册不可修改；暂停保留记录，再激活可复用原ID。更换定义需要新ID。

每张处方激活1–4个单元，整数配额合计240。首次处方必须满足后续全部阶段的重复上限容量；后续只核对下一阶段。编译依据曝光次数和固定seed轮换，不另加 CER 分数挑数据。配额变化（增/减/暂停/激活）由Harness计算。

Reflection仍由同一Therapist角色完成，无Skill Optimizer。每个激活单元必须引用自己的测量ID，讨论：

1. 课程中实际练过的唯一录音及同集合前后loss；同时列课程曝光、含coverage总曝光。
2. 同患者、该阶段所有训练未涉及的其他训练content；必须排除coverage，也按每个患者实际练习content计算。
3. 这些患者在511条反馈集上的未见content。部分患者没有对应反馈，报告空集合，不填0或模拟结果。

`outcome.global_observation_ids` 显式列出本轮 overall/easy/medium/hard 全局反馈ID。每个单元的 Reflection 至少引用一个本单元ID，可以附加这些全局ID来判断整体收益或保护缺口；不能只引用全局、引用其他单元的局部ID、旧轮ID或不存在的ID。局部患者改善而全局easy退化可以同时成立，但全局退化不等于该单元造成退化。提示词、字段说明和运行时校验使用同一规则。

结果是多单元和coverage共同训练后的关联反馈，不宣称隔离了每个单元的因果作用。critical状态提供真实的保持/退出/进入ID，修复旧版“数量相同就代表成员不变”的推断漏洞。

## 启动与验收

```bash
# 准备阶段的真实反向传播检查也明确使用单卡。
export CUDA_VISIBLE_DEVICES=0

PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src \
  /data/qwen3-asr/venvs/qwen3-asr/bin/python examples/real_curriculum_v2.py prepare

# 准备验证：CPU完整模拟链路，以及真实32条音频的L20前后向；optimizer更新数为0。
PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src \
  /data/qwen3-asr/venvs/qwen3-asr/bin/python examples/real_curriculum_v2.py check
PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src \
  /data/qwen3-asr/venvs/qwen3-asr/bin/python examples/real_curriculum_v2.py model-smoke

# 用户另行确认开始后才执行；密钥通过环境变量或隐藏交互输入，永不写入协议。
PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src \
  /data/qwen3-asr/venvs/qwen3-asr/bin/python examples/real_curriculum_v2.py run
```

处方/Reflection继续使用已有 gpt-5.6-sol/xhigh 网关配置、Responses严格JSON schema、store=false、API重试0。准备阶段的fixture只验证接口，不冒充新模型的step20或真实Agent输出。正式首次处方只能在真实step20诊断后产生。

评测节点固定20/40/60/80/100。同时报告完整同step曲线、固定step100和同一组节点内各组最佳checkpoint：overall mean-utterance CER、easy/medium/hard、pooled CER、patient macro、总字符错误。报告是否达到overall<20%，以及dynamic相对static/random的差异，不根据结果追加步数、换学习率或重新挑首次处方。技术错误直接停止，最多由用户授权一次针对性复核；没有自动fallback。

结构化输出接口参考：[OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)。本协议不推断第三方网关的官方模型可用性。
