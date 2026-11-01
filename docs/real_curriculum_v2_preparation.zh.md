# Real curriculum v2：准备验收（2026-09-07）

状态：**代码和准备验证完成，等待用户另行启动正式训练。** 没有新 ASR performance 结果。

## 已实现

- 学习进展 Evidence：previous→current 与 base→current 分开，逐条曝光次数、固定模式loss、患者级聚合。
- 持久 Uxxx 练习单元、配额编译、暂停/增减记录；定义不可覆盖，暂停不清历史。
- 三组同预算100步：共同20步起步；随后每20步400 coverage +240 course；完整覆盖2117条real，每条每阶段最多2次。
- static首次处方冻结，dynamic首次处方相同，此后才根据实际Reflection调整；不使用TTS或M0权重初始化。
- 单一连续Trainer、无跨阶段预取、optimizer/scheduler/RNG连续；20/40/60/80/100统一评测。
- 三层结果回流：已练录音、排除coverage的同患者其他content、511条合并反馈中的未见content。
- step20共同状态及static/dynamic step40完整指纹校验；相同状态复用decode与fit，减少重复计算。

## 验证结果

**40项测试通过**，包括CPU上的完整三组三百个逻辑更新步模拟，使用小型测试模型与显式模拟Agent/decoder，不冒充正式实验。

- 插入诊断和随机数操作，再继续训练，与不中断路径的参数、optimizer、scheduler、Torch RNG结果一致。
- 三组各3200次曝光、全部2117条唯一真实录音覆盖；共享位置/录音一致；static/dynamic前40步一致。
- 动态组确实接收上一阶段Reflection，并在后续改变真实编译清单。
- 未生成的未来训练位置不能被读取；旧位置不可覆盖；非法ID、证据、容量和配额直接失败，无自动补样或改处方。
- 真实zero-shot/M0预测与原evaluator核对：overall、pooled、patient macro、总错误/参考字符数一致。准备时发现并修正了新helper的patient macro口径：必须先对每位患者pool字符错误率，再等权平均；历史实验没有改动。

**L20真实模型前后向验证通过**：

| 项目 | 实测 |
|---|---:|
| 真实录音 | 32条，8×4 |
| 有限梯度tensor | 536 |
| 峰值已分配显存 | 14.31 GiB |
| 验证耗时（含模型/数据准备） | 29.75秒 |
| optimizer更新 / 参数更新 / 新checkpoint | 0 / 0 / 0 |

GPU已释放；磁盘约31 GiB可用。当前准备产物约11 MiB，早期准备快照另保留约11 MiB；没有复制基础模型或音频，没有清理历史模型。

## Agent接口与启动边界

按OpenAI Docs核对严格结构化输出；沿用现有gpt-5.6-sol/xhigh网关配置，store=false、API重试0。

本次没有线上Agent调用：当前进程没有OPENAI_API_KEY，启动入口支持环境变量或隐藏输入。验证清单和请求形状明确标注为fixture，不会用作正式首次处方。正式首次处方必须读取真实step20结果。

完整2117行准备输入（含system/schema）的o200k估算约208448 tokens；这是准备快照估算，不是网关实际usage或后续历史增长后的固定长度。prompt数值保留6位小数，本地测量保留全精度。

正式运行仍需用户说开始；入口不会由prepare/check/model-smoke自动触发。

## 文件

- [实验协议](./real_curriculum_v2.zh.md)
- [执行入口](../examples/real_curriculum_v2.py)
- [就绪凭证](/data/qwen3-asr/records/therapist_mas_setting_d_v1/real_curriculum_v2/readiness.json)
- [离线验收](/data/qwen3-asr/records/therapist_mas_setting_d_v1/real_curriculum_v2/validation/offline_checks.json)
- [真实音频前后向凭证](/data/qwen3-asr/records/therapist_mas_setting_d_v1/real_curriculum_v2/validation/model_smoke.json)
