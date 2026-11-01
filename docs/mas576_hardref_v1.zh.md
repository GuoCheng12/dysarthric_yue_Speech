# MAS新文本576条：更换reference后微调

沿用旧MAS六轮的36个新文本、51位患者及576条文本×患者配对，每个文本16条。保持目标文本、标点、旧生成seed和样本顺序；重新生成时只换同患者reference。此批vlink247为6条，与此前Q重建576中的122条不同。

新reference取自全部真实训练集的同患者录音，技术合格且时长3–12秒；排除原reference的文本身份和已知被污染的vlink228货币录音。优先保留原reference的3秒时长分箱，若无候选则使用最近非空分箱，在其中按zero-shot CER选择最多3个不同文本，固定seed均衡轮换。少数据患者允许1–2个reference。576条全部更换reference，使用129个不同reference；6位患者需换到邻近时长分箱。

使用原epoch8 CosyVoice3患者LoRA，保持推理参数不变，启用已验证的空token恢复处理。技术异常最多重试10次，seed为旧seed+10000×attempt；不按下游CER筛选或选种子。技术检查沿用既有0.25–30秒等条件，不等于内容全部正确。

训练两组，各seed42/43/44：

- 全部2117真实＋旧MAS576原音频。
- 全部2117真实＋换hard-reference后重新生成的MAS576。

所有真实数据沿用real_q三seed对照的顺序，追加576条，总2693条。每次从Qwen3-ASR-1.7B新建LoRA，batch8×gradacc4、lr2e-4、r16/alpha4/dropout0.05；相同140步linear scheduler与warmup0.02，固定第50步停止并评测。复用已验证的训练入口，不修改其源码。真实样本曝光未固定。

复用全部真实2117和全部真实＋Q重建hard-TTS576的三seed结果作为另外两组对照。报告Overall、Easy/Medium/Hard、vlink247106条、其余患者405条，并独立重算编辑距离。MAS与Q重建之间同时改变了文本和患者分布，不能只解释为文本效应。511条为反复使用的开发反馈集合，MAS文本也来自此前反馈过程。

入口：`examples/train_mas576_hardref.py --stage prepare|run|synthesize|assemble|train|report`。
两张GPU各生成288条，随后并行执行旧/新reference两组的同seed训练，共三批。可从已写入的生成收据恢复；不会覆盖原MAS音频或既有实验记录。

结果目录：`/data/qwen3-asr/records/therapist_mas_setting_d_v1/real2117_mas576_hardref_v1`。

启动前通过reference选择与空token处理的6个测试，并用既有预测检查报告计算、逐条身份及相同输入的零差值。最终以`acceptance.json`作为结果核验完成标志。
