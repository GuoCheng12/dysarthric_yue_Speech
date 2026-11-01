# SG Flow-only 患者适配：最终结果

运行目录：`/data/qwen3-asr/records/cosyvoice3_flow_sft_setting_d_v1`。

## 结论

**Flow-only 患者适配没有通过预设的声学进展标准，因此没有继续做 576 条 ASR utility 训练。**

这不是训练没有运行：Flow 在 2108 条真实训练语音上完成了 3 epochs、792 个 optimizer steps，
训练过程没有 NaN/Inf，held-out conditional-flow loss 从 0.5224 降至 0.4986。但这一下降没有转化为
对未见 content 的患者语音重建改善；相反，固定真实 token 后，重建声音整体变得更偏离原声。

因此目前最直接的判断是：**问题不能只靠在现有 tokenizer 输出之后微调 Flow 解决。** 这条单配置路线
在本数据和当前 reference 条件下应停止，不应再把更多 ASR 训练成本投入进去。

## 实验条件

- 固定 epoch8 Speech Generator 的 LLM、tokenizer 和 HiFT/vocoder，只训练 Flow 全部 332,257,088 个参数。
- 2108 条 Setting D train、57 位患者；reference 为同患者不同训练录音。
- 24 位评测患者，easy/medium/hard 各 8 条；评测 target content 不在 TTS train 中。
- Flow：Adam，LR 1e-5，constant，microbatch 2、effective batch 8，3 epochs；目标只在 target mel 帧计算 loss，reference 只作为条件。
- epoch0 和每个训练 epoch 都生成完整配对音频。所有 epoch 使用相同患者、GT、reference、seed；训练 epoch 使用相同的真实 target token，文本条件则复用 epoch0 实际生成的 token 序列，从而把差异限制在 Flow。
- MCD-DTW 越低、CAMPPlus 越高表示与患者原声更接近；二者都不是病理构音指标。

## 整体结果

### 真实 token 重建

| Flow | MCD-DTW ↓ | CAMPPlus ↑ | 时长误差 % ↓ | 粗 F0 中位数差（半音） ↓ |
|---:|---:|---:|---:|---:|
| epoch 0（原 SG） | **7.0945** | **0.7369** | 0.09 | 1.81 |
| epoch 1 | 7.2415 | 0.7079 | 0.09 | 1.72 |
| epoch 2 | 7.1320 | 0.7108 | 0.09 | **1.00** |
| epoch 3 | 7.3555 | 0.7063 | 0.09 | 1.14 |

epoch2 是训练点中 MCD 最低的点，但仍比 epoch0 高 0.0375 dB，CAMPPlus 低 0.0261；没有达到“重建 MCD 改善且 CAMPPlus 不下降超过 0.01”的标准。

hard 组同样没有改善：MCD 从 epoch0 的 8.4081 变为 epoch2 的 8.4413，CAMPPlus 从 0.7250 变为 0.7150。

### 固定 SG token 的文本条件生成

| Flow | MCD-DTW ↓ | CAMPPlus ↑ | 时长误差 % ↓ |
|---:|---:|---:|---:|
| epoch 0（原 SG） | **7.8797** | **0.6842** | 17.67 |
| epoch 1 | 8.0169 | 0.6558 | 17.67 |
| epoch 2 | 7.9601 | 0.6617 | 17.67 |
| epoch 3 | 8.0983 | 0.6531 | 17.67 |

文本条件生成没有改善；epoch2 虽比 epoch1/3 接近，但仍同时劣于 epoch0 的 MCD 和 CAMPPlus。

## 为什么 held-out loss 下降却没有变好

Flow loss 只说明模型更贴合训练集上的 conditional-flow velocity 目标，不等于未见 content 的声学重建更接近患者。这里训练端的 speaker/reference 条件和真实训练语音都被充分看到，但评测要求模型把同一类条件泛化到新的句子。

更关键的是，之前的 48 条重建诊断已经显示：真实 token 经过原有 tokenizer + Flow + vocoder 时，患者原声信息就会损失。此次只放开 Flow 后，这个损失没有被修复。说明瓶颈至少不是“Flow 完全没做患者微调”这么简单，可能还涉及 tokenizer 表示、token-to-mel 条件形式，以及训练目标与患者异常发音信息之间的不匹配。

## ASR utility 决定

预设条件要求：所选 Flow checkpoint 的真实 token 重建和文本条件生成都比 epoch0 低 MCD，且 CAMPPlus 各自下降不超过 0.01。实际四项检查均为 false：

```json
{
  "selected_epoch": 2,
  "reconstruction_mcd_improves": false,
  "reconstruction_camp_preserved": false,
  "text_mcd_improves": false,
  "text_camp_preserved": false,
  "proceed_to_utility": false
}
```

所以没有重新生成 576 条合成数据，也没有启动额外的三 seed ASR 训练。这样可以把结论归因于 Flow 适配本身，而不是在声学质量没有改善时继续堆下游训练量。

## 审计与文件

- cache 中的 24 条评测语音 token 与既有官方 frontend 缓存逐条完全一致；CAMPPlus embedding 最大误差小于 1e-5。
- smoke 中自定义训练条件与官方 Flow 推理的 `mu / speaker / cond` 最大绝对误差均为 0；关键参数均有有限非零梯度，实际更新后丢弃 smoke 权重。
- 4 个评测点均有 24 条 reconstruction + 24 条 text-condition 音频，共 192 条完整试听音频；HTML 含 240 个播放器（另含真实原声和 reference）。
- 运行结束可用空间约 31 GiB，原有 SG、数据和 checkpoint 未改写。

完整试听：[listening/index.html](/data/qwen3-asr/records/cosyvoice3_flow_sft_setting_d_v1/listening/index.html)

机器可读判定：[acoustic_decision.json](/data/qwen3-asr/records/cosyvoice3_flow_sft_setting_d_v1/acoustic_decision.json)

逐 epoch 指标：[evaluation/epoch_0/metrics.json](/data/qwen3-asr/records/cosyvoice3_flow_sft_setting_d_v1/evaluation/epoch_0/metrics.json)
