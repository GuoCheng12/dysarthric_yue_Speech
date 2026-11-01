# 患者 TTS 配对诊断结果

结论：优先把VoxCPM2作为下一轮患者微调候选，Qwen3-TTS次之；本轮不支持优先换成Fish S2。三个新模型的zero-shot都没有直接解决患者特征保留问题。

## 已完成

24条Q、19位患者，easy/medium/hard各8条。三个新模型各生成24条克隆与24条声学重建，共144个新WAV；复用旧条件后共264条诊断录音。统一冻结replay ASR，144条新回译，零新增训练。

| 文本生成版本 | 逐句平均CER | 对原声CAMPPlus中位数 | 时长/原声中位数 |
| --- | ---: | ---: | ---: |
| 患者原声 | 19.95% | 1.000 | 1.000 |
| 当前Speech Generator（已微调） | 8.11% | 0.662 | 0.895 |
| 原始CosyVoice3 | 63.83% | 0.613 | 0.965 |
| VoxCPM2 zero-shot | 17.43% | 0.620 | 0.642 |
| Qwen3-TTS Base zero-shot | 16.84% | 0.589 | 0.582 |
| Fish S2 Pro zero-shot | 22.93% | 0.517 | 0.531 |

这是合成语音回译，不是augmentation后的ASR效果。24条Q均在旧Speech Generator的TTS训练wav.scp中，但未进入冻结replay ASR的训练；旧SG是诊断参照，不是同适配条件baseline。

## 为什么优先VoxCPM2

绕过文本生成、直接重建原声时：

| 声学重建路径 | CER | CAMPPlus | 粗F0轮廓偏移中位数（半音） |
| --- | ---: | ---: | ---: |
| CosyVoice token＋Flow | 29.18% | 0.735 | 1.174 |
| VoxCPM2 AudioVAE | 23.07% | 0.923 | 0.243 |
| Qwen3-TTS codec | 22.48% | 0.914 | 0.425 |
| Fish S2 codec | 27.51% | 0.881 | 0.431 |

Vox的声学重建在音色与粗F0上最接近原声。在共同无歧义的13条对齐中，原声共有9个ASR替换/删除错误：Vox保留7个、新增1个；CosyVoice token＋Flow保留4个、新增11个。这个小诊断支持优先测试Vox底座，不将9个错误扩张成病理构音泛化结论。

但进入text→speech后，Vox仍有明显距离：时长中位数只有原声的64.2%，音色相似度0.620。在共同无歧义的11条文本生成对照中，6个原声ASR错误，Vox只保留1个，Qwen/Fish/旧SG各保留0个。高生成CER主要不能解释为“保留了患者错误”。

因此下一步应检验Vox患者微调能否利用较好的声学底座，而不是直接把任何新模型的zero-shot音频灌入ASR，也不是只挑最低合成CER的模型。

## 验收与存储

- 7项测试通过，144个新WAV均核对帧数、采样率、SHA-256、reference与seed。
- 无长度上限触发；保留模型全部输出，不人工裁剪。模型是否漏读仍由GT与回译/试听展示，不能把“文件完整”等同于“内容完全正确”。
- 模型权重固定revision并核对远端LFS SHA-256；原ASR环境未改变。
- 模型约19.1GiB，独立环境约6GiB；删除已安装的3个wheel包，约0.8GB，可重新下载。
- 完成时磁盘约36.1GiB（df显示37G）可用，高于20GiB保留线。没有删除旧checkpoint或原始数据。

完整数值：`/data/qwen3-asr/records/therapist_mas_setting_d_v1/tts_model_comparison24_v1/results.json`

试听：`/data/qwen3-asr/records/therapist_mas_setting_d_v1/tts_model_comparison24_v1/listening/index.html`
