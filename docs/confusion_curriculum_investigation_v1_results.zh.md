# 双向混淆课程调查：结果与停止决定

2026-09-06。

## 结论

**停止，不构建新 MAS、不运行 Agent smoke。**

现有结果确实显示 real-only 适配改善 hard、退化 easy，但没有找到足够支持“修好 A→B，同时使原本正确的 B 反向误识别成 A”的跨患者、跨 content 具体混淆对。93 个主分析候选中，满足执行前继续条件的为 **0**。

这不是“所有课程学习都无效”的结论，也不等于证明每一个音都不存在这种现象。它直接决定本轮行动：**没有足够依据把反向音素混淆作为新 MAS 的核心机制，不继续搭建和跑 smoke。**

## 1. 核验的数据与模型

- 固定 Setting D train：2117 real、66 位患者、48 个唯一句子。
- 固定反馈集：原 dev258 + test253，共 511 条、12 个 content，easy/medium/hard 为 265/145/101。与 train 的 utt_id 和 content 均无交集。
- 主比较：原 zero-shot → real-only M0 batch8/acc4 checkpoint50。
- 重复检查：此前真实增量正对照的 real_q seed42/43/44 checkpoint50。其训练清单只含 audio/text/prompt；逐 audio 核对三项输入与原 2117 real 完全相同，再由原清单关联患者和 content。三次均从原始 Qwen3-ASR 新建 LoRA，不是继续 M0。
- 使用已有预测，没有新增解码、训练、TTS 或外部 API 调用。
- M0 是历史上按 dev 选择的 step50；本次不重新选 checkpoint。三个复核 seed 则使用原先预定的固定 step50。

主指标为 **mean utterance CER（%）**，不是 pooled CER：

| 模型 | Overall | Easy | Medium | Hard |
|---|---:|---:|---:|---:|
| Zero-shot | 25.6977 | 6.4093 | 28.7753 | 71.8874 |
| M0 step50 | 21.9613 | 7.3967 | 24.7609 | 56.1560 |
| Real-only seed42 | 21.6623 | 7.4426 | 24.3454 | 55.1195 |
| Real-only seed43 | 21.9389 | 6.9884 | 24.7083 | 57.1896 |
| Real-only seed44 | 21.4433 | 7.1124 | 23.9923 | 55.3847 |

需要纠正此前概括：**在这组固定比较中，medium 总体也在改善。** 真正稳定复现的组间现象是 hard 改善、easy 退化，不是 easy 和 medium 两组都整体退化。某些 medium 录音仍然可以变差，不能把个体变化和组均值混为一谈。

## 2. 判定标准没有在看完结果后修改

执行前计划见 [调查计划](/data/qwen3-asr/repo/dysarthric_yue_Speech/docs/confusion_curriculum_investigation_v1.zh.md)。计划与输入文件 SHA256 已在结果运行前写入 sources.json。

一个主候选须同时具有：hard 方向至少 5 次真正恢复、3 位患者、2 个 content、2 个 GT 字；easy/medium 反方向新增错误也有同等支持；两个方向的净变化符合假设；至少两个既有 seed 重复双向净变化并各有至少 3 次对应事件；真实训练池两端都有可用录音。

| 条件 | 通过的主候选数 / 93 |
|---|---:|
| hard 恢复达到 5/3/2/2 支持 | 0 |
| easy/medium 反向退化达到 5/3/2/2 支持 | 1 |
| 主比较同时具有两侧正净变化 | 2 |
| 至少两个 seed 达到规定的重复支持 | 0 |
| 训练池两端均有可用录音 | 93 |
| 全部条件同时成立 | **0** |

所以本次失败的原因不是找不到真实训练素材，也不是超过训练时间或存储预算，而是支持核心机制的证据不足。

## 3. 哪些零星现象确实出现了？

主分析仅统计：原始文本读音可解析且上下文稳定、原文和 CER 对齐都确定、两个字的 Jyutping **只差一个分量** 的替换。字已被真正识别正确才算恢复；转为其他错字或删除不算。

| 候选纠错方向 | M0 hard 真恢复 | M0 easy/medium 反向新错 | 为什么不继续 |
|---|---:|---:|---|
| coda i→空 | 1 | 1 | 只涉及一条 content；三个复核 seed 的反向新错均为 0 |
| nucleus yu→i | 1 | 2 | hard 始终只是一位患者的一处 `住`；seed42/43/44 为恢复1、反向新错1/2/2，仍缺少跨患者纠错支持 |
| tone 3→1 | 3 | 0 | 没有对应的反向新增错误 |
| coda k→t（此前假设例子） | 0 | 0 | 没有观察到所举例子的主证据 |

因此不是绝对“找不到任何一次共同出现”，而是**零星共同出现不足以支撑一个值得实现的新机制**。尤其不能把一个患者、一个字在多个 seed 重现，当作多个独立患者/content 的证据。

## 4. 为什么粗粒度统计看起来可能支持故事？

另保留了 298 个允许多分量同时变化的探索性候选。它们不用于改变主判定：

| 粗粒度候选 | hard 恢复 | easy/medium 反向新错 | 只差该分量的主证据 |
|---|---:|---:|---|
| tone 1→3，反向3→1 | 5 | 6 | 恢复1，反向新错0 |
| tone 6→3，反向3→6 | 5 | 8 | 恢复1，反向新错0 |

具体例子解释了差别：

- `堆`（deoi1）从 `對`（deoi3）改回 `堆`，可以作为一个单声调分量差异的真实整字恢复例。
- 另一条原本正确的 `對抗人生曲折` 被改成 `追尋人生曲折`。其中 `對`→`追` 不只有 tone3→1，声母也变了；它还是整段词组替换。仅把它计入“反向声调错误”，无法据此定位声调判断边界。
- `蝴蝶` 中 `蝶` 从 `接` 改回，与 `書冊` 被改成 `書籍` 同时存在。两者能被聚合成相反的声调方向，但同时改变了其他语音分量／词汇选择；不能把共同的声调标签当作因果联系。

这些是实际转写现象，不是人为构造的示例。完整 utt_id、患者、GT 位置和 before/after 文本见 audit.json。它们说明**忽略多分量差异会制造容易讲述、却没有被隔离验证的音素机制**；本次没有将探索性聚合替代主证据。

## 5. 覆盖与边界

511 条共有 6656 个原始 GT 字位置。其中 170 个位置因 G2P 无法解析或上下文读音敏感，不参与分量诊断。

M0 主比较中，可确认的整字错误→正确有 273 个位置、正确→错误有 206 个位置；其余还包含正确保持、错误保持和无法确定的对齐变化。它们不是纯音素事件计数，也不能忽略插入错误直接换算为 CER。

只差单个语音分量、且有明确整字变化的主证据较少：hard 恢复 15 个、新错 8 个；easy 恢复 2 个、新错 19 个；medium 恢复 15 个、新错 16 个。这些事件分散在多个不同方向，并没有形成足够支持的反向混淆对。

本次结论针对转写中的可归因证据，不把字典 G2P 当成患者的真实声学发音测量。合并 511 条早已被用于反馈，也不能被称为未经查看的最终 test。

## 6. 验收与交付

- 新增调查测试 7 项；与原 Harness 的 phonology、目标恢复、归一化和 grounding 相关测试合计 **42 passed**。
- 另一个完整矩阵编辑距离实现重新核对 5 个模型 × 511 条 = **2555 条预测**，所有 overall/分组 mean utterance CER 和 pooled CER 均一致。
- 核对输入/计划/分析脚本 hash、real_q 训练池身份、三个 step50 的 global_step、初始化路径、batch、grad_acc、LR 和 seed。
- 没有放宽阈值、换 step 找阳性、改动旧 MAS、调用 Agent、生成音频或启动训练。
- 新结果约 2 MB，不保存新模型权重。

记录目录：[/data/qwen3-asr/records/therapist_mas_setting_d_v1/confusion_curriculum_investigation_v1](/data/qwen3-asr/records/therapist_mas_setting_d_v1/confusion_curriculum_investigation_v1)。

- [机器可读结果与逐例证据](/data/qwen3-asr/records/therapist_mas_setting_d_v1/confusion_curriculum_investigation_v1/audit.json)
- [来源与 SHA256](/data/qwen3-asr/records/therapist_mas_setting_d_v1/confusion_curriculum_investigation_v1/sources.json)
- [验收记录](/data/qwen3-asr/records/therapist_mas_setting_d_v1/confusion_curriculum_investigation_v1/acceptance.json)
- [分析脚本](/data/qwen3-asr/repo/dysarthric_yue_Speech/examples/investigate_confusion_curriculum.py)
- [独立核验脚本](/data/qwen3-asr/repo/dysarthric_yue_Speech/examples/verify_confusion_curriculum_investigation.py)

最终状态：**按用户条件停止。此次不进入新 MAS 的构建和 smoke 阶段。**
