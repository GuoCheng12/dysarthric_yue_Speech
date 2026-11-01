# 患者 TTS 真实增量正对照

运行目录：`/data/qwen3-asr/records/therapist_mas_setting_d_v1/tts_increment_positive_control_v2`。

目的：没有新增真实数据时，用现有真实录音构造一个可测量的真实增量，判断患者 TTS 能否转移这个增量的 ASR 价值。

## 冻结设计

从 Setting D 的 2117 条真实训练语音中分出 R=1541、Q=576。Q 按患者和已有 zero-shot 难度分层抽取，所有合成 reference 留在 R；每个患者×难度层至少保留一个真实 replay 候选。Q 覆盖 46 位患者，easy/medium/hard=300/166/110。R 覆盖全部 66 位患者，easy/medium/hard=796/442/303。R、Q 各包含 48 个原训练 content；这不是新 content 划分实验。

Q 的 46 位患者包含 merged 评测集的全部 36 位患者：511 条评测语音均属于本次真实/合成增量覆盖的患者。评测难度为 easy 265、medium 145、hard 101。

| 组别 | 共用数据 | 增量数据 | 总条数 |
| --- | --- | --- | --- |
| replay | R 1541 | 从 R 重复采样 576，与 Q 患者×难度逐条匹配 | 2117 |
| real_q | R 1541 | 真实 Q 576 | 2117 |
| patient_tts | R 1541 | Speech Generator 按 Q 患者和 GT 合成 576 | 2117 |
| base_tts | R 1541 | 官方 CosyVoice3 按相同 Q 患者和 GT 合成 576 | 2117 |

两个合成组逐条使用同一 reference 和同一成功 seed。完整保留生成音频，不用 ASR CER 筛选。仅当音频出现空输出、非有限值、静音、严重削波、DC 异常或时长越界时，两个模型同时换到下一个预设 seed（最多 10 对）；全部尝试保留记录。CUDA、模型、文件读写等程序错误直接报错，不作为重采样理由。

基础模型由加载器读入官方权重后移除未合并的 LoRA 层，并逐张量验证 LLM 与官方 checkpoint 完全相同；flow/vocoder 均来自同一官方模型。患者模型使用当前 epoch 8 Speech Generator。现有 Speech Generator 已见过原 2117 条（包括 Q），所以本实验测的是已训练生成器向 ASR 转移真实录音增量的效用，不是生成器未见 Q 的泛化。

## 训练与评测

- 四组都从原始 Qwen3-ASR-1.7B 新建 LoRA，不从 M0 继续。
- 三个训练 seed：42、43、44；同 seed 使用相同数据槽位顺序，共用 R 位于相同槽位。
- batch 8、gradient accumulation 4；已修复的 loss normalization。
- LoRA rank 16、alpha 4、dropout 0.05，七类 projection。
- LR 2e-4，linear scheduler 总 horizon 140，warmup 0.02，沿用既有 baseline。
- 每次实际训练固定 50 optimizer steps，仅评测 step 50；不选 best checkpoint，不按结果追加训练。
- 每次约 1600 次样本曝光，约 0.75 epoch；不是声称全部 2117 条都看过一次。
- 评测使用既有 merged real 511 条（原 dev+test）；没有 synthetic 评测项。
- 总计 12 次独立训练。比较三 seed 平均 utterance CER、easy/medium/hard CER，同时保存 pooled CER、patient macro CER 与逐句预测。

## 判读顺序

1. real_q 相对 replay：真实录音增量能否改善当前训练方案。
2. patient_tts 相对 replay：患者合成是否提供超出重复训练的价值。
3. patient_tts 相对 base_tts：患者生成器微调是否提供额外的 ASR 效用。
4. patient_tts 相对 real_q：它与真实增量还差多少。

结论据三 seed 的差值和一致性给出。患者 TTS 优于 replay 不等同于优于完整 2117 条真实训练；优于基础 TTS 不单独证明某一种病理构音机制。

## 实现与验证

入口：`examples/tts_increment_control.py`。`protocol.json` 冻结分组、参数和代码 SHA256；`pairs.json` 冻结合成输入；`selection.json` 保存真实划分；`assembly.json` 保存训练文件 hash；每组每 seed 保存执行参数、checkpoint、预测和结果。

启动前，划分确定性、reference 保留、患者与难度匹配、训练命令一致性及已有梯度修复回归测试：9 项通过。v1 为准备草稿，没有启动计算；v2 在启动前收窄重采样异常范围后重新冻结，划分不变。
