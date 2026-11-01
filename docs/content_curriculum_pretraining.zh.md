# Content-centered 课程：训练前接口

本次只准备代码、证据、处方和采样清单，不启动ASR，不改变已有实验。
入口 `examples/prepare_content_curriculum.py` 只有 `prepare` 和 `smoke`，没有训练子命令。

## 数据与初始化边界

- Setting D仍为2117条真实训练录音、66位患者、48个content。
- 反馈沿用旧dev+test合并后的511条语音；不恢复dev，不把反馈录音纳入课程。
- 首次接口smoke复用已完成Dynamic实验step20的真实预测和损失，不加载其模型权重。
- 它不是已经开始的新训练，也不决定下一正式实验从哪个权重开始。正式启动前应在
  选定的实际起点重新计算证据；不能把旧snapshot当成另一个模型的诊断。
- 原训练/反馈数据文件哈希沿用并复核；本次不生成TTS、不复制音频或基础模型。

## 1. 内容证据

每个E_Cxx提供：原始GT、规范化长度、easy/medium/hard各自的样本数、患者数、
当前/前一checkpoint平均CER、具体错误示例，以及每种可执行选择器的容量。
这些指标只是已有ASR错误的描述，不是处方效用预测或训练前Gate。

首版不使用声学phone推断；使用整句、前半、后半、最后两字四种确定的文本区域。
前半按floor(N/2)切分，后半为余下部分；它们不一定对应词语或语法边界。
后半句错误存在并不意味着它显著多于前半句，更不证明患者病理成因。

可执行的选择器：

|selection|入选条件|
|---|---|
|error|指定区域至少一个确定替换或删除|
|deletion / substitution|指定区域至少一个对应确定错误|
|regression|指定区域至少一字从前一checkpoint正确变为当前确定替换/删除|
|correct|整个规范化预测与完整GT完全一致|
|any|该content下所有所选原始难度桶的录音|

确定位置沿用所有最优字符对齐中唯一映射的口径。歧义不当成确定错误。
纯插入错误不进入error选择器；仍计入普通CER、固定coverage。
区域仅用于选样：训练仍使用完整原声、完整GT和原有整句监督loss，不裁剪、不掩码。

容量表每行是 `[region, selection, bucket, rows, patients, exposure_capacity]`。
容量已扣除下一块400次coverage占用；缺失组合的容量为0。选择多个难度桶时，
只能对同一region/selection的不同桶容量求和，不能相加重叠的region/selection。

## 2. 处方是内容课程，不是患者名单

一张处方包含2–3堂课；每堂课有明确识别目标、内容关系、证据、迁移假设和修订条件。
每堂课3–6个content条目，其中至少两个不同target和一个related；整张处方至少一个
protection条目。总课程曝光固定240，但不硬编码三种角色的比例。

- target必须实际匹配error/deletion/substitution/regression。
- related解释为何该不同内容可以提供相关语境或对照；不要求是假设的最小音素对。
- protection只从原始easy/medium录音中选regression或correct，不按患者平均CER选。
- 首版每个content在一块中只承担一个角色，避免重叠预算和混淆训练解释。
- 每条content先分配曝光配额，再在满足条件的患者间均衡，最后轮换患者内录音。
  容量不等时并非保证人人相同；结果提供实际患者分配，不用候选名单冒充覆盖。

固定400coverage +240course，每个有效batch20+12，microbatch8×accumulation4，
每录音每块总曝光最多2次。训练位置和旧BlockDataset格式兼容，交错混合而非分段先易后难。
本次不改LR、LoRA、20-step块大小或正式训练起点；不把这些变量同时开放给Agent。

## 3. 执行反馈与Reflection

delivery逐content记录：计划/实际曝光、目标命中、实际患者份额、难度分布、唯一录音、
近期退化/整句正确的实际曝光及含重复时长。先确认处方实施，再解释学习结果。

`lesson_outcome` 在真实更新后接收固定manifest和前后预测，分别计算：

1. 实际target/related/protection录音；
2. 同content、该块未练过的患者-content组合；
3. 实际受训患者的其他训练content，排除该块coverage和全部课程涉及的组合；
4. 对应患者在511反馈集里的真正未见content；
5. 全局overall/easy/medium/hard反馈。

另外逐content单独测量处方所指定文本区域的确定错误、正确、歧义位置及恢复/退化，
避免整句CER下降掩盖目标位置没有改善。区域测量仍不是声学诊断。

2、3可能在早前块练过，不称为全局未见content；4才与完整训练文本隔离。
实际采样人数不足以覆盖全部反馈患者时，不把局部结果说成全局保护。
多堂课与coverage联合更新，单课相关改善不等于被隔离的因果效果。

Reflection包含执行解释、观测结论、下一步动作和`reusable_lesson`；原始结果和反思
一并进入下一处方的history。这是上下文中的经验积累，不需要额外Skill Optimizer。
没有训练后结果时不生成“成功Reflection”；测试中的前后结果明确是fixture。

## 4. 运行与验收

```bash
PYTHONPATH=.:src /data/qwen3-asr/venvs/qwen3-asr/bin/python \
  examples/prepare_content_curriculum.py prepare --root /absolute/new/preparation_directory

PYTHONPATH=.:src /data/qwen3-asr/venvs/qwen3-asr/bin/python -m pytest \
  tests/test_content_curriculum.py tests/test_curriculum_v2.py -q

# 凭据通过已有OPENAI_API_KEY环境提供；一次处方调用，不启动ASR。
PYTHONPATH=.:src /data/qwen3-asr/venvs/qwen3-asr/bin/python \
  examples/prepare_content_curriculum.py smoke --root /absolute/prepared_directory
```

`draft/` 是研究者拟定、经真实候选容量验证的三堂课，不冒充Agent输出。
`smoke/prescribe_request.json` 是待发送的独立完整prompt；真实调用成功且处方编译通过后，
才产生 `agent_smoke/` 与 `smoke/validated.json`。API调用保留请求、响应和耗时，不自动重试或修补处方。

每份准备目录不可覆盖；输入/代码哈希固定。验收检查内容配额、真实错误匹配、患者分配、
完整音频/GT、重复上限、20:12组成、跨内容评测排除关系，以及历史记录是否进入下一请求。
结构化输出沿用现有gpt-5.6-sol/xhigh配置；按OpenAI Docs使用必填字段与严格JSON schema，
业务语义另由编译器验证：[Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)。
