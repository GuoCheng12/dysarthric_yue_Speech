# 患者 TTS 选型：24 条配对诊断

## 固定内容

- 从上一轮48条Q的每个难度桶按原顺序隔条选取8条，共24条、19位患者。
- 不根据新模型的输出筛样本；沿用原GT、同患者reference及原seed。
- 新模型：VoxCPM2、Qwen3-TTS-12Hz-1.7B-Base、Fish Audio S2 Pro。模型revision写入protocol。
- 比较各模型的zero-shot克隆和真实Q的codec/VAE重建，新增144条wav。
- 复用旧原声、已微调Speech Generator、未微调CosyVoice3，以及CosyVoice3两种重建结果；总计11种条件、264条诊断录音。
- 统一使用上一轮冻结replay seed42 checkpoint-50 ASR。复用旧条件的120条回译，只新增144条回译。
- 不训练ASR或TTS。Fish输出仅做诊断，不混入任何训练清单。

## 接口选择

- VoxCPM2：reference＋continuation双条件，关闭denoiser、文本正规化、坏样本重试；使用官方10步Flow、CFG=2。放宽按文本长度限制音频的阈值，保留384个patch的总上限并记录触顶。
- Qwen3-TTS：Base的完整ICL模式，reference audio＋reference transcript，language=Auto，不使用仅音色embedding模式。768帧上限并记录生成长度。
- Fish：官方参考条件与speaker标签；temperature=1、top_p=0.9、top_k=30。记录EOS是否正常出现。Codec必须完整精确载入。
- 各自官方接口负责从续接输出中分离reference前缀。除此以外不裁剪、不降噪、不修音、不换seed重试。
- 保存FLOAT WAV的完整输出，记录采样数、采样率、时长、幅值和SHA-256；不同模型的相同seed并不意味着相同随机轨迹。

## 验证

1. CER：逐句均值与corpus CER并列，easy/medium/hard固定沿用旧标签。
2. 声学：原声时长比、WORLD粗F0轮廓偏移、CAMPPlus对原声的相似度。
3. CAMPPlus不是病理指标；F0粗轮廓不是音节级声调诊断。
4. Codec/VAE重建和文本生成分开解释。CosyVoice的token重建还经过reference-conditioned Flow，不与其他codec视为相同压缩率的竞赛。
5. 生成语音回译分数不是augmentation的训练收益；当前Speech Generator已微调，而三个新模型均未患者微调。

## 存储

- 初始约61GiB可用，保留至少20GiB空闲。
- 三个模型合计约19.1GiB；Qwen Base已带speech_tokenizer，不重复下载。
- 固定模型版本，只下载一份权重。Qwen codec慢链路改由官方ModelScope同模型续传，核对Hugging Face固定版本SHA-256。
- 推理环境与原项目隔离；能兼容的依赖只读复用，Fish所需Torch 2.8依赖安装在其独立环境。
- 系统 `/tmp` 空间不足，安装临时目录改用 `/data/qwen3-asr/tmp/tts_model_probe_pip`；没有清理旧模型或数据。
- 大文件慢链路最终用 aria2 分段续传，下载后验证固定 SHA-256，不保留第二份模型权重。
- 不删除已有checkpoint，不复制已有音频。试听HTML只用symlink链接完整wav。

## 实际运行审计

- 原ASR环境保持Torch/Torchaudio 2.5.1、Transformers 4.57.6。
- Vox/Qwen独立环境使用Transformers 4.57.3；Fish使用Torch/Torchaudio 2.8.0+cu126、Transformers 4.57.3。
- Fish官方codec代码将6个RoPE/causal-mask缓存设为non-persistent，发布权重仍带这些缓存。遵循官方重新生成缓存的行为，明确断言只有这6个unexpected key、没有missing key；没有忽略任何学习参数。
- Vox的48条回译在下载其余模型期间先完成；最终仅对Qwen/Fish剩余96条新增回译，统一汇总。
- 推理过程中允许不同模型任务重叠，记录的wall time用于运行追踪，不作为严格速度benchmark。

运行记录：`/data/qwen3-asr/records/therapist_mas_setting_d_v1/tts_model_comparison24_v1`

入口：`examples/tts_model_comparison.py`、`examples/tts_model_comparison_worker.py`、`examples/report_tts_model_comparison.py`。
