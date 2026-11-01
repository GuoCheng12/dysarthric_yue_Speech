# R + Patient TTS 的 reference 策略对照

入口：`examples/tts_reference_policy.py`。

默认记录目录：`/data/qwen3-asr/records/therapist_mas_setting_d_v1/tts_reference_policy_v1`。

本次只实施代码、CPU 准备和测试，不启动新的 GPU 生成或训练。

## 比较与固定项

复用旧 R-only 和旧 Patient TTS 的三个 seed。新增 `random_multi`、`hard_multi` 两组，每组 R 1541 + 合成 576 条，ASR seed 42/43/44，共六次新训练。

两组使用原 Q 的患者、GT、槽位顺序及原始逐目标 TTS seed；reference 只来自 R 中同一患者的录音。两组的规范化文本与粤拼 hash 都不能与目标相同。旧 reference 保持原样，诊断中若与目标文本相同也不会悄悄替换。

ASR 直接调用旧实验的 `training_command` 与 `train`，固定 50 optimizer updates；linear scheduler horizon 仍为 140，不改为 50。从原始 Qwen3-ASR 初始化 LoRA，其他超参数与旧实验一致。训练前核对源码、依赖版本、数据/音频 hash；训练后核对逐步学习率。评测仍为已有 merged511，属于探索性结果。

## Reference 选择的精确定义

1. 检查 R 中相关患者的录音：单声道、非空、有限数值、时长 3–12 秒、peak 与 RMS 至少 1e-4、绝对 DC 不超过 0.2、削波比例不超过 0.02。不按 CER 剔除候选。
2. 固定 reference-selection seed 为 20260907。每位患者的候选先按 utt_id 排序再打乱；同规范化文本或同粤拼的重复录音只保留一个随机代表，避免重复文本占满 reference 池。
3. `random_multi` 在该不同文本候选池内随机选最多 3 条。候选少于 2 条时报出患者 ID，不改用 Q 或其他患者。
4. 时长分为 `[3,6)`、`[6,9)`、`[9,12]` 秒。`hard_multi` 在每个区间选 CER 最高的若干条，数量等于随机组在该区间的数量；CER 并列按 utt_id 排序。
5. 两组共有的 reference 放在相同槽位，其余槽位在相同时长区间内按最接近时长配对。每个目标在两组共用槽位，排除任一组与目标同文本的槽位，再选当前使用次数最少的槽位，固定随机数打破并列。

因此，两个策略逐目标的 reference 时长区间一致，实际秒数不保证相同。少量候选患者可能在两组使用完全相同的 reference；`reference_audit.json` 记录这一情况。使用次数在同文本排除限制内尽量均衡，完整槽位计数保存下来。Hard 指 zero-shot ASR 识别难度，不是临床严重度。

## 运行入口

所有命令在 `/data/qwen3-asr/repo/dysarthric_yue_Speech` 下执行，使用与旧实验相同的 Python：

```bash
/data/qwen3-asr/venvs/qwen3-asr/bin/python examples/tts_reference_policy.py --stage prepare
```

`prepare` 只使用 CPU，冻结 reference 配对、48 条诊断子集、六个训练命令、源码和输入 hash。已经准备过的目录不会覆盖写入。当前目录已经完成准备时无需再次运行。

后续选择空闲 GPU，通过 `CUDA_VISIBLE_DEVICES` 指定；脚本继承该变量，不自动抢占或停止其他任务。GPU 阶段按顺序执行：

```bash
/data/qwen3-asr/venvs/qwen3-asr/bin/python -u examples/tts_reference_policy.py --stage probe
/data/qwen3-asr/venvs/qwen3-asr/bin/python -u examples/tts_reference_policy.py --stage run
```

`run` 会先完成尚缺的 probe，再生成剩余音频、组装两组训练集、顺序执行六次训练，最后生成对比报告。已完成且可验证的合成配对会复用；已生成 probe 结果、assembly 或训练结果的阶段会跳过。完整运行应由终端会话管理器或已有任务调度方式保活。

也可逐阶段运行：

```bash
/data/qwen3-asr/venvs/qwen3-asr/bin/python examples/tts_reference_policy.py --stage synthesize --probe-only
/data/qwen3-asr/venvs/qwen3-asr/bin/python examples/tts_reference_policy.py --stage diagnose
/data/qwen3-asr/venvs/qwen3-asr/bin/python examples/tts_reference_policy.py --stage synthesize
/data/qwen3-asr/venvs/qwen3-asr/bin/python examples/tts_reference_policy.py --stage assemble
/data/qwen3-asr/venvs/qwen3-asr/bin/python examples/tts_reference_policy.py --stage train --arm random_multi --seed 42
/data/qwen3-asr/venvs/qwen3-asr/bin/python examples/tts_reference_policy.py --stage report
```

所有阶段接受 `--root /绝对路径`。训练必须同时指定 `--arm` 和 `--seed`。`report` 要求两个新组的三个 seed 全部完成。

## 配对生成与诊断

复用旧 48 条诊断抽样算法：Q 内 easy/medium/hard 各 16 条，按患者轮转，抽样 seed 为 20260906。每个目标重新生成旧 reference、random、hard 三个版本，使用同一成功 seed；诊断还包含原 Q 真实录音。

采用固定 epoch 8 Patient TTS 加载器及原技术有效性检查。技术失败时整对重试，seed 为 `原 seed + 10000 × attempt`，最多十次；48 条 probe 按三个版本一起重试。其余目标仅生成两个新版本。因此，完整语料只有两组各 576 条，额外的 48 条旧 reference 音频只用于诊断。

只捕获生成器明确返回的音频有效性错误；CUDA、加载、文件写入错误直接终止。每次尝试和失败都保留。已写音频但未写 receipt 的中断尝试需要检查，程序不会覆盖它；未完成的 ASR 训练目录也不会自动重置。

诊断使用固定 R-only seed42 ASR，保存逐句识别、CER、时长、字符/秒，以及 10ms 帧相对能量的静音比例。静音与字符速度均为代理指标，不据此自动判定障碍程度。`probe/listening.zh.md` 提供四版本及其 reference 的音频链接、目标文本和转录，供检查漏字、重复、内容混入和发音差异。CER 不用于筛选或从两种策略中挑赢家。

## 产物与核算

- `protocol.json`：冻结配置、来源与训练命令。
- `pairs.json`、`reference_audit.json`、`preparation.json`：全部目标配对、候选/排除原因、槽位使用次数与难度/时长统计。
- `run_status.json`、各阶段日志、`selected/*.json`：当前阶段、运行状态、生成回执。
- `probe/diagnostics.json`、`probe/listening.zh.md`：配对诊断与试听。
- 两组 `train.jsonl`、`assembly.json`、各 seed 的 checkpoint、预测与指标。
- `results.json`、`report.zh.md`、`acceptance.json`：两个新策略对 R-only、旧合成，以及 hard 对 random 的逐 seed、逐患者差值；独立重算 12 个新旧 run 的四组 CER，并核对 patient macro 和 pooled CER。

首先判断是否超过 R-only；只超过旧 Patient TTS 不能单独说明数据增强有效。三个 ASR seed 共用固定 reference 清单，不是三套 reference 重复实验；旧固定策略对多 reference 是整套策略的比较，也不单独识别 reference 数量效应。

## 验证

```bash
/data/qwen3-asr/venvs/qwen3-asr/bin/python -m pytest tests/test_tts_reference_policy.py tests/test_tts_increment_control.py tests/test_qwen3_asr_lora_gradient_accumulation.py -q
```

覆盖真实划分上的选择确定性、同患者与同文本排除、时长区间配对、小候选池、技术质量检查、三版本配对重试、CUDA 错误不重采样、完成配对复用、seed/音频篡改检测，以及报告发现错误 CER 并正确计算零差值。复用旧测试核对训练命令及梯度累积尾批处理。
