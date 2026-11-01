# M0 新增数据吸收 probe

运行目录：`/data/qwen3-asr/records/therapist_mas_setting_d_v1/m0_absorption_factorial_v1`。

执行状态：8组单轮扫描和连续pure-synthetic后两轮全部完成。完整结果见[实验报告](/data/qwen3-asr/records/therapist_mas_setting_d_v1/m0_absorption_factorial_v1/report.zh.md)。

单轮推荐32synthetic+0real、1遍（1step）：overall21.7522%，M0为21.9613%，easy/medium/hard均小幅改善。所有带replay的合成混合组均未同时做到overall/hard改善和easy/medium不退化；四遍总体更差。连续96synthetic、0real、2遍/轮三轮CER为21.8145%→22.2336%→22.1005%，没有easy崩坏，但hard退化。未启动连续32条×1遍新实验。

## 要回答的问题

固定内容，单独观察synthetic量、real replay量、重复遍数怎样影响真实语音识别；另测连续三轮synthetic-only。没有Agent调用，不做训练前Gate。

所有单轮分支从同一个原Real-only M0 checkpoint50初始化。使用已有第一轮6文本×16患者的96条Speech Generator音频。32条是其子集，保留16位患者（每人2句），6文本分别5/5/5/5/6/6条。real288按speaker×difficulty比例抽取，real96再从中抽取。固定随机种子，轮内顺序不变，无重采样。

| synthetic | real replay | 每遍新增steps | 观测遍数 |
|---:|---:|---:|---|
| 32 | 0 | 1 | 1/2/4 |
| 32 | 96 | 4 | 1/2/4 |
| 32 | 288 | 10 | 1/2/4 |
| 96 | 0 | 3 | 1/2/4 |
| 96 | 96 | 6 | 1/2/4 |
| 96 | 288 | 12 | 1/2/4 |
| 0 | 96 | 3 | 1/2/4 |
| 0 | 288 | 9 | 1/2/4 |

同遍数表示每条选中的synthetic/real都看到相同次数，不意味着相同optimizer步数。报告同时列出steps和遍数，不将replay量变化误称为纯损失权重变化。

lr=2e-5 constant，warmup0，batch8×gradacc4，LoRA r16/alpha4/dropout0.05及目标模块不变，修复后的utterance-mean梯度路径。每轮权重加载、optimizer重新初始化，轮内连续训练。所有观测点保存可推理adapter，不保存optimizer副本；不删除旧实验。

## 连续纯合成

复用96synthetic+0real分支的第2遍checkpoint作为第一轮终点；第二、三轮使用已有对应轮各96条冻结语音，各自训练2遍（6 steps），无real，按自身父checkpoint连续更新，不重置到M0。这样三轮18个新增steps，内容不是根据本次结果重新生成的。

## 判断

每个观测点解码合并511条真实反馈语音：overall/easy/medium/hard、pooled、patient-macro CER。额外逐GT字位比较M0与新预测，分别记录原正确→错误/不确定、原错误→正确，不让修复数掩盖损失数。字位不唯一对齐记为unresolved，CER仍使用全量。

固定训练样本在0和各观测点关闭dropout测real/synthetic分项loss，只作为确实在拟合的证据。推荐条件预先固定为overall和hard优于M0、easy/medium不高于M0；没有满足者就直接报告没有，并给出最接近的实际权衡，不通过修改阈值制造成功。

验证：18项相关测试通过（数据嵌套/完整更新窗口/字位损益/既有固定loss与梯度累积测试）。最终结果写入运行目录results.json。
