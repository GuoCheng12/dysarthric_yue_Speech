# Setting E 离线材料选择准备

2026-09-11 后续状态：本离线组件已由新的在线材料采样模块接入诊断、调度、训练与反思。
在线实现和实验状态见 `docs/setting_e_material_curriculum_exploration_20260911.zh.md`。
下文保留侧对话离线准备阶段的实现范围与回执；此离线模块本身仍不调用API或启动训练。

新增部分只构建训练录音索引、诊断交接格式、候选材料和选择结果校验。
未修改现有训练、MAS 提案或反思入口；不调用 API、不启动 GPU。

## 分工和交接

诊断 Agent 提出至多六个具体练习目标，输出 `DiagnosticBrief`：

- `producer: diagnosis_agent`、轮次、原始证据的 SHA256、简短诊断。
- 每个目标有编号、目的（针对问题／对照／保持能力）、证据编号和理由。
- 检索条件为声母、韵腹、韵尾、声调、完整粤拼音节或完整目标文本。
- 查询的是要练习的目标读音；若需要反向对照，由诊断 Agent 另列一个对照目标。

程序根据目标检索训练池，给材料选择 Agent 提供诊断交接和候选材料。
材料选择 Agent 不接收原始 dev/test 转写，不另做诊断，不分配训练比例。
每个目标输出具体录音编号和简短理由；没有适合的录音可以返回空列表。
未来由调度 Agent 决定材料曝光，当前不自动改变任何训练权重。

## 离线索引

使用 Setting E 的 1862 条真实训练录音、46 位患者、60 个目标文本。
每条记录保留音频路径、患者编号、文本编号、时长、原始识别难度。
每个文本保留上下文词典粤拼、逐字位置、声母／韵腹／韵尾／声调、未解析信息。
读音复用项目 `sampling_phonology.pronunciation`；是文本字典读音，不是声学测量。
60 个文本中有 2 个字符未解析，保留记录，不编造读音。

dev/test 清单仅用于检查录音编号、路径、患者隔离，不把其文本加入索引。
索引绑定源清单 SHA256 及自身内容哈希，不宣称重新校验了全部音频文件内容。
历史曝光单独作为可选输入，不写入静态索引；未提供显示 null。

候选检索先交替覆盖不同文本、不同患者，再参考已知曝光次数和稳定编号。
每个目标默认展示 24 条；完整匹配编号保存在本地 candidates.json，可提高 limit 重新查询。
这一步是程序检索，不是 Agent 选择；选择结果只能引用本次展示的候选。
匹配率一并提供，避免将非常普遍的音误称为精确练习。需要更细时诊断可指定完整音节或句子。
同一录音用于多个目标时，校验器合并为一条并保留所有目标标签，不自动增加曝光。

## 入口

在仓库根目录，以 `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src` 运行：

```bash
/data/qwen3-asr/venvs/qwen3-asr/bin/python examples/prepare_material_selection_offline.py index --train TRAIN.jsonl --exclude DEV.jsonl TEST.jsonl --out NEW_DIRECTORY
/data/qwen3-asr/venvs/qwen3-asr/bin/python examples/prepare_material_selection_offline.py query --index INDEX.json --diagnosis BRIEF.json --limit 24 --out NEW_QUERY_DIRECTORY
/data/qwen3-asr/venvs/qwen3-asr/bin/python examples/prepare_material_selection_offline.py validate --index INDEX.json --candidates CANDIDATES.json --selection AGENT_SELECTION.json --out NEW_RESULT.json
```

写入采用新建文件模式，已存在的结果不覆盖。query 生成候选及 GPT-5.6-sol 的结构化请求，
不发送请求。实际 API 执行将来应复用 laicai 调用和回执机制，地址为 https://api.laicaishare.xyz。

## 已运行的离线 smoke

产物：`/data/qwen3-asr/records/therapist_mas_setting_e_v1/material_selection_offline_20260911_v1/`。
示例诊断是从第一轮 dev 证据提取检索条件的 **offline_fixture**，不是诊断 Agent 的新产出。
示例选择由程序取候选前四条，仅验证格式、匹配、来源及去重，不冒充 Agent 的选择。

| 查询 | 匹配录音 | 文本 | 患者 |
| --- | ---: | ---: | ---: |
| 韵尾 k | 1195 | 38 | 46 |
| 韵腹 aa | 1645 | 54 | 33 |
| 声调 6 | 1482 | 49 | 46 |

每项展示的 24 条候选覆盖 24 个文本、24 位患者。三个目标各取四条的校验样例去重后为 5 条，
说明同一录音可能服务多个目标，不能直接把三个列表长度相加当成新增材料数量。
单元测试覆盖正确读音位置、完整音节缩小范围、文本与患者覆盖、无匹配、隔离、来源和去重。
