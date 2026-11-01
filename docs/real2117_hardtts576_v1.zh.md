# 全部真实数据 + hard-reference TTS

本实验在全部2117条真实训练录音上追加既有576条hard-reference TTS，检查此前R1541+TTS的结果是否随真实数据覆盖改变。

- 真实清单与`tts_increment_positive_control_v2/real_q/train.jsonl`逐条、逐序相同；其内容等于原始全部2117条真实训练数据。
- 追加`tts_reference_policy_v1/hard_multi/train.jsonl`中的576条合成录音，音频SHA256、标签与生成收据逐一核对。未重新生成、过滤、换reference或修改文本。
- 最终2693条：R1541、Q576真实录音、Q576对应合成录音。Q的真实版本和合成版本都进入训练。
- 每个seed从原始Qwen3-ASR-1.7B新建LoRA；seed42/43/44，batch8×gradacc4，lr2e-4，LoRA r16/alpha4/dropout0.05。
- 复用既有140步linear学习率序列、warmup0.02，在第50步停止并评测。与全部真实基线的命令差异仅训练清单路径和输出路径，逐步学习率事后核验。训练预算按更新次数匹配，未固定真实样本曝光量。
- 保留原先逐句平均CER主指标；同时报告Easy/Medium/Hard、vlink247的106条及其余患者405条，并保留pooled和患者macro指标。既有511条合并dev+test属于开发反馈范围。
- 对照复用R-only1541、全部真实2117和R1541+hardTTS576的三seed结果。三个seed全部执行，不按中间CER选择是否继续。

运行入口：`examples/train_real2117_hardtts576.py --stage prepare|run|train|report`。单seed训练使用`--stage train --seed 42`（或43、44）。运行记录位于：

`/data/qwen3-asr/records/therapist_mas_setting_d_v1/real2117_hardtts576_v1`

启动前完成真实/TTS清单、音频哈希、训练参数与数据身份检查。报告检查在临时目录复用既有预测，独立重算12组模型的6个分组，验证相同输入的差值为零以及106/405加权可还原Overall。最终结果需有`acceptance.json`才视为通过核验。
