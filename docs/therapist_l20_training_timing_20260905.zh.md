# L20：首轮处方 ASR 训练粗测速

## 结论

本次实测候选中，推荐 micro-batch=8、gradient accumulation=4（有效 batch=32）。不是全空间调优结论；batch=16 没有带来可辨认的速度收益，显存代价却明显更高。

- 单独 24 条合成语音：热运行一遍约 2.22 秒，冷启动第一遍计时约 3.31 秒，另加模型加载等准备时间。仅为速度测试，不是正式 synthetic-only 训练。
- 正式带 real replay 的一轮：2117 real + 24 synthetic = 2141 条，67 个 optimizer step；训练部分推算约 3.3 分钟。
- 当前 1×32：相同一轮训练部分推算约 10.3 分钟。
- 测速进程准备阶段约 47–56 秒，包含导入、模型加载、数据清单/时长统计和部分标签核对。故 8×4 一轮可先按约 4–5 分钟训练流程预留；checkpoint 保存与真实 dev 解码 CER 尚未实测，不能算成完整闭环耗时。

## 方法与结果

GPU：单张 NVIDIA L20，46068 MiB；无其他 GPU 作业。模型为原始 Qwen3-ASR + 同一 Real-only checkpoint-50 的 trainable LoRA，r=16、alpha=4、dropout=.05，19,791,872 个可训练参数。BF16、SDPA、AdamW Torch、不开启 gradient checkpointing。学习率等沿用训练入口默认值。

每个常规候选独立加载相同权重、相同 2141 条清单与 seed=42，测 3 个 optimizer step；表中稳态值取 step 2、3 平均。有效 batch 固定为32。短运行只测吞吐，不比较 loss/CER 或训练效果；3-step scheduler 不用于正式训练。

| micro-batch × 累积 | 稳态秒/step | 峰值 allocated GiB | 峰值 reserved GiB | 67 step 训练估时 |
|---|---:|---:|---:|---:|
| 1 × 32 | 9.211 | 5.546 | 6.775 | 10.29 分钟 |
| 8 × 4 | 2.936 | 14.476 | 20.498 | 3.28 分钟 |
| 16 × 2 | 2.958 | 24.767 | 35.035 | 3.30 分钟 |
| 32 × 1 | 未完成第一个 step | — | — | DataLoader 共享内存错误 |

独立压力测试：将整个候选池中时长最长的32条集中训练2步，使用8×4。成功；稳态step为3.908秒，峰值 allocated=23.669 GiB、reserved=28.797 GiB。全池最长音频29.877秒。这是长语音压力测试，不是完整epoch已经通过的证明。

独立 synthetic24 测试：仅24条，8×4，每个 epoch 只有一个不足32样本的累积组；运行3个step，实际读了3遍，每遍24条，不是每步32条。step2/3耗时2.240/2.204秒，峰值allocated=11.060 GiB。不要用 Trainer 按名义batch计算的samples/sec反推实际样本数。

## 正式增大 batch 前需要处理的两个实际问题

### 1. 左 padding 与标签遮罩不匹配

`finetune/official/qwen3_asr_sft.py` 的 collator 按 `labels[i, :prefix_length] = -100` 遮罩，但 Qwen ASR processor 默认传入 `padding_side=left`。尽管 tokenizer 对象属性显示right，processor调用默认值仍会覆盖它。

对前4条real样本，分别单条处理与批量处理：原实现有3条监督token不一致，批量监督中混入音频占位符与assistant prompt。batch=1不受此padding错位影响。

测速脚本仅在本地 processor 包装器中显式传入 `padding_side=right`，然后核对批量4条的有效标签逐token等于单条结果。8/16/32和压力测试、synthetic测试均使用此修正。正式collator本次没有修改。

此外，即使修复padding，现有loss是micro-batch内的token mean，再对累积micro-batch取平均；1×32偏向每句等权，8×4不是严格相同的句子权重。有效batch相同不代表数学上完全相同的优化目标。正式迁移需明确要保持utterance-mean还是统一token-mean，不能把本次吞吐表当作训练效果等价证明。

### 2. 单卡入口意外探测不可用的 DeepSpeed

首次原样启动在Trainer初始化阶段失败：可选DeepSpeed安装探测需要不存在的CUDA_HOME。历史日志显示旧入口曾禁用该探测，清理后当前入口不再包含处理。

测速脚本明确指定不使用DeepSpeed，在本进程禁用Accelerate的DeepSpeed unwrap探测。没有安装依赖、变更系统CUDA或修改正式入口。正式运行前也需要明确固定非DeepSpeed路径。

## DataLoader 与配置建议

当前 `/dev/shm` 仅64 MiB。batch32、workers2、prefetch2在第一个训练step前发生worker bus error，并有明确共享内存不足提示；不是CUDA OOM，不能据此声称L20装不下batch32。

本次已验证的8×4配置：

```yaml
per_device_train_batch_size: 8
gradient_accumulation_steps: 4
bf16: true
attention_implementation: sdpa
gradient_checkpointing: false
use_cache: false
optim: adamw_torch
dataloader_num_workers: 2
dataloader_pin_memory: true
dataloader_persistent_workers: true
dataloader_prefetch_factor: 2
```

LoRA rank/alpha/target modules、lr、有效batch不因测速而擅自调整。不建议为了填满显存而强行上batch32，也未安装FlashAttention或测试compile。若未来扩大并行预取，应单独调整共享内存配置后再测试，不要默默fallback。

官方说明同样指出，能放得下时更大的micro-batch通常比大量梯度累积更有效率，但最终应以实测吞吐决定；参见[Transformers 单卡训练指南](https://huggingface.co/docs/transformers/v4.26.0/en/perf_train_gpu_one)。上表均来自本机实测，不来自该文档的示例数据。

## 产物与执行边界

测速代码：`examples/benchmark_asr_round.py`。记录目录：`/data/qwen3-asr/records/therapist_mas_setting_d_v1/l20_timing_v1/`。各成功子目录有 `benchmark.json`、测速数据清单及模型配置元数据。

原始失败目录 `mixed_b1` 和 `mixed_b32` 已保留；这两次失败记录没有complete状态，分别对应上述DeepSpeed初始化异常和共享内存异常，不能算成功样本。

本次没有保存任何新adapter/checkpoint，没有覆盖M0，没有启动正式MAS轮次，没有读取dev/test做训练，也没有改变正式protocol。
