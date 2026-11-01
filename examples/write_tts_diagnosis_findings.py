"""Write the final Chinese findings only after all registered runs are complete."""
import json
from pathlib import Path

ROOT=Path('/data/qwen3-asr/records/therapist_mas_setting_d_v1/tts_utility_diagnosis_v1')
NAMES={
    'real_w0':'纯真实，新配对基线',
    'mas_hard_w025':'MAS hard，固定真实曝光',
    'q_hard_w025':'原 Q hard，固定真实曝光',
    'mas_hard_language_w025':'合成只更新语言侧',
    'mas_hard_audio_w025':'合成只更新音频侧',
    'mas_hard_protected_w025':'合成梯度投影与范数限制',
    'consistency_shuffled_w025':'按患者匹配的随机 reference 对照',
    'consistency_best_w025':'选择转录更准确的 reference 版本',
    'mas_labels_clean_w025':'只统一 MAS 句末标签',
    'mas_hard_decay20':'前20步逐渐减小合成权重',
    'strict_qc_clean_w005':'严格筛查＋统一标签＋低权重',
    'strict_qc_room_w005':'上一组＋轻量混响、噪声、音量处理',
    'real_period_w0':'全部真实标签带句号，新格式基线',
    'mas_period_w025':'全部真实标签带句号＋原 MAS hard',
    'real_full72_w0':'纯真实，完整覆盖＋72步衰减终点',
    'room_full72_w005':'修复TTS，完整覆盖＋72步衰减终点',
}


def read(path):return json.loads(path.read_text())
def link(name,path):return f'[{name}]({path})'


def run():
    data=read(ROOT/'current_results.json');post=read(ROOT/'post_small/results.json')
    for arm in NAMES:
        assert set(data['results'].get(arm,{}))=={'42','43','44'},arm
    for arm in ['synthetic32','real32']:
        assert set(post['results'].get(arm,{}))=={'42','43','44'},arm
    means=data['means'];comparisons=data['paired_vs_real']
    real_arms=['real_w0','real_period_w0','real_full72_w0']
    candidates=[a for a in NAMES if a not in real_arms]
    best=min(candidates,key=lambda a:comparisons[a]['mean_delta_pp'])
    delta=comparisons[best]['mean_delta_pp']
    historical=read(ROOT/'scoring/variants.json')['results']['old_real2117']
    historical_mean=sum(v['overall']['legacy_cer'] for v in historical.values())/len(historical)
    strongest_real=min([means[a]['overall'] for a in real_arms]+[historical_mean])
    lowest_tts=min(candidates,key=lambda a:means[a]['overall'])
    scope=read(ROOT/'content_transfer/results.json')
    uncertainty=read(ROOT/'cluster_sensitivity.json')
    full72_ci=uncertainty['comparisons']['room_full72_w005']['patient_and_text_95ci_pp']
    lines=['# TTS 为什么没有让 ASR tuning 更好：诊断结果','',
        f'完成{sum(len(data["results"][a]) for a in NAMES)}次从base训练的配对实验、6次短步继续训练，以及95条生成端对照音频。以下只报告已经完成的结果。','',
        f"当前配对增量最好的配置是“{NAMES[best]}”，相对其纯真实基线差值为{delta:+.4f}个百分点（CER越低越好）。"+
        ('数值差距不足0.01个百分点，实际可描述为持平，不能作为有效增益证据。' if abs(delta)<0.01 else
         '观察到平均数值改善，仍需结合逐seed方向、分项和探索性评测的不确定性判断。' if delta<0 else
         '本轮没有找到总体均值超过其配对纯真实基线的增强配置。'),'',
        f"最低绝对CER的合成配置为“{NAMES[lowest_tts]}”：{means[lowest_tts]['overall']:.4f}%；现有纯真实配置的最低三seed均值为{strongest_real:.4f}%。配对增量与是否超过最强纯真实配置是两个不同问题。",'',
        '不能把这些结果概括为“全部TTS都是noise”，也不能因为某个子集改善，就宣布全体评测已经胜出。','',
        '## 统一训练对照','',
        '均使用同一Qwen3-ASR初始模型、相同LoRA配置和seed42/43/44。每步32条真实＋8条合成；每组与其配对基线共享完全相同的真实序列及真实损失权重。纯真实对照保留合成零权重计算并隔离其随机状态。','',
        '**2117条是训练池大小；50步协议实际有1600次真实曝光和400次合成曝光，并非把整个2117条或576条都训练了一遍。** 该实验控制了真实曝光被合成替换的问题。历史旧协议全真实均值21.6815%，不与新序列基线混作配对差值。','',
        '上述为50步协议，沿用140步学习率衰减计划。另补充完整覆盖对照：72步、学习率也在72步衰减结束，共2304次真实曝光（覆盖全部2117条）和576次合成曝光（覆盖全部514条修复TTS）。它使用自己的纯真实72步基线，同时改变覆盖量和学习率过程，不将其与50步差值全部归因于TTS。','',
        '| 配置 | 三seed平均 CER% | 对配对纯真实差值，pp | 改善的seed数 |',
        '|---|---:|---:|---:|']
    for arm,name in NAMES.items():
        c=comparisons.get(arm)
        if arm in real_arms:
            diff='基线';wins='—'
        else:
            diff=f"{c['mean_delta_pp']:+.4f}"
            if arm=='mas_period_w025':diff+='（对新格式基线）'
            if arm=='room_full72_w005':diff+='（对72步基线）'
            wins=f"{sum(v<0 for v in c['seed_deltas_pp'].values())}/3"
        lines.append(f'| {name} | {means[arm]["overall"]:.4f} | {diff} | {wins} |')
    lines+=['',
        '所有表内CER均为逐句平均，保留原有计分。'+link('逐seed、患者、难度分项及直接干预对照',ROOT/'current_report.zh.md')+'。','',
        '## 已落地的实用修复效果','',
        f"原MAS hard为{means['mas_hard_w025']['overall']:.4f}%；按reference候选转录筛查、统一句末标签、降低合成权重并加入轻量声学处理后为{means['strict_qc_room_w005']['overall']:.4f}%。两者差值{means['strict_qc_room_w005']['overall']-means['mas_hard_w025']['overall']:+.4f}pp，属于多项组合修复，不能全部归功于其中一个因素。",'',
        f"在相同514条源音频、相同标签、顺序和权重0.05下，仅增加声学处理的三seed平均差值为{means['strict_qc_room_w005']['overall']-means['strict_qc_clean_w005']['overall']:+.4f}pp。轻量混响、噪声与音量归一化作为一个组合，三个seed均改善；尚未拆分这三项。",'',
        f"总体持平也不代表每个患者群体持平：50步修复组相对纯真实，vlink247为{means['strict_qc_room_w005']['vlink247']-means['real_w0']['vlink247']:+.4f}pp，其他患者为{means['strict_qc_room_w005']['other_patients']-means['real_w0']['other_patients']:+.4f}pp，Hard子集为{means['strict_qc_room_w005']['hard']-means['real_w0']['hard']:+.4f}pp。这些分项按原先固定的分组报告，不从中挑选获胜群体替代全体。",'',
        '可复查的修复数据为'+link('514条训练manifest',ROOT/'room_noise/train.jsonl')+'，处理参数为'+link('声学处理协议',ROOT/'room_noise/protocol.json')+'，逐seed训练命令与输入hash保存在'+link('seed42协议',ROOT/'additive/strict_qc_room_w005/seed_42/protocol.json')+'及同目录其他seed中。这解决的是已观察到的部分负迁移，不等于获得了稳定的新增信息收益。','',
        '复现须使用实验入口`examples/tts_additive_control.py`，其负责固定顺序、分域loss、随机状态隔离和step50停止。完整启动参数见'+link('启动队列',ROOT/'final_gpu0_queue.json')+'中的strict_qc_room_w005三条命令；重新运行需给`--root`指定一个新的输出目录。仅直接执行protocol里的底层finetune命令不会加载上述实验行为。','',
        f"完整覆盖的72步对照：纯真实{means['real_full72_w0']['overall']:.4f}%，修复后R+TTS为{means['room_full72_w005']['overall']:.4f}%，配对差值{comparisons['room_full72_w005']['mean_delta_pp']:+.4f}pp；逐seed差值为{comparisons['room_full72_w005']['seed_deltas_pp']}。完整覆盖和完成学习率衰减的效果，已经用该配对直接检查。",'',
        f"72步增量的患者×文本重采样95%探索性区间为[{full72_ci[0]:+.4f},{full72_ci[1]:+.4f}]pp，包含零。当前有配对平均正增益，但还没有稳定超过最强纯真实配置的证据。"+link('全部配置逐seed差值图',ROOT/'paired_effects.png')+'。','',
        '## 哪些问题已经找到，哪些仍是推测','',
        '1. **格式监督确实冲突，但它不是充分解释。** 原真实标签均不带句末标点，MAS全部带“。”。在16条被ASR完整识别正确的MAS音频上，句号占真实微调模型总teacher-forcing损失的86.66%；删除它后平均token loss由0.5154降至0.0704。此比例只对应选定的正确子集。统一标签明显改变训练梯度方向，但其三seed总体收益并不稳定。',
        '2. **生成端有内容不一致及reference残留线索。** 两个同系列ASR都在部分输出中识别出参考句尾，而目标文本没有这些词。切换instruct2＋粤语指令修复了部分例子，另一些词的错读仍存在。SG LoRA与reference接入方式存在交互，不能简单归因于base或患者微调谁总是更好。',
        '3. **更低的合成筛查CER不等于更好的下游ASR。** 配对最佳reference选择将筛查CER降至9.5531%，进一步阈值筛查降至5.2566%；后者保留514条、51患者、35文本。筛查是模型一致性代理，不是人工转录或临床障碍保真度验证。',
        '4. **新增文本带来的输出倾向变化是具体机制线索。** 在“織頸巾…”这句话上，部分预测转为“職警…”或“職棒…”。真实训练文字中“職”出现0次，MAS576中出现80次。这符合语言先验改变的假设；尚未完成只针对该内容因素的因果消融。',
        '5. **真实与TTS的错误有部分对应，却并不等同。** 576对同患者、同文本Q录音中，100条是真实录音识别错、TTS识别正确；另81条是真实正确、TTS识别错。真实侧931个ASR错误位置有317个也在TTS侧出错，约34.05%，高于同文本随机配对的平均203.37个位置。这不是临床发音标注，也不证明其余部分全是随机噪声。','',
        '将MAS中出现、真实2117训练文本中未出现的字符单独检查，原MAS增强后，511条预测中这些字符的正确对齐次数三seed平均由470增至491；错误输出次数也由115.33增至143.67。可见有学习收益，也有错误输出增加。两类计数不是CER百分点，不作简单相减或完整因果归因。'+link('词形输出变化审计',ROOT/'content_transfer/lexical_bias.json')+'。','',
        '严格筛查保留全部51位患者，但真实训练语音更难识别的患者，合成保留比例总体更低（描述性Spearman约−0.394）；参考录音的平均zero-shot CER也从29.97%降到27.78%。因此筛查可能同时改变有价值的困难发音分布。这里只使用真实reference的难度标签；MAS筛查文件里的medium是兼容评测器的占位值，不是合成严重程度。'+link('筛查选择偏差审计',ROOT/'consistency/selection_difficulty_audit.json')+'。','',
        link('逐token损失证据',ROOT/'label_loss/results.json')+'；'+
        link('原始失败案例试听',ROOT/'failure_examples.html')+'；'+
        link('reference接入方式对照试听',ROOT/'instruction_pilot/listening.html')+'；'+
        link('SG与base对照试听',ROOT/'base_instruction_pilot/listening.html')+'。','',
        '## 评测集合为何容易改变结论','',
        '511条包含36位患者，但只有12个不同文本。vlink247占106条；患者和文本均有大量重复观测。这个集合已被反复用于开发反馈，不再是独立最终test。','',
        '| 配置 | 原dev258 CER% | 原test253 CER% | 合并511 CER% |',
        '|---|---:|---:|---:|']
    for arm in ['real_w0','mas_hard_w025','consistency_best_w025','strict_qc_clean_w005','strict_qc_room_w005','real_full72_w0','room_full72_w005']:
        by_seed=scope['results'][arm]
        dev=sum(v['dev']['cer'] for v in by_seed.values())/len(by_seed)
        test=sum(v['test']['cer'] for v in by_seed.values())/len(by_seed)
        lines.append(f'| {NAMES[arm]} | {dev:.4f} | {test:.4f} | {means[arm]["overall"]:.4f} |')
    lines+=['',
        '最佳reference选择在原test部分的三个seed均改善，但其中两个改善很小；原dev部分退化，使合并结果变差。原test六句话中四个改善、两个退化。这是事后机制分析，不是改用获胜子集替代主指标。','',
        '同时重采样患者和文本后，质量筛选组相对纯真实的差值+0.2421pp对应探索性区间约[−0.9016,+1.5951]pp。该区间条件于已训练模型，未校正多次配置选择。'+link('完整敏感性分析',ROOT/'cluster_sensitivity.json')+'。','',
        '另已修复銳/鋭、拐/枴/柺的异体字计分差异，所有组使用同一新规则重算；原计分及历史结果保持不变。计分修复明显降低绝对CER，但没有解释原MAS混合组相对纯真实的差距。'+link('新旧计分完整对照',ROOT/'scoring/variants.zh.md')+'。','',
        '## 历史短步正结果的复现','',
        '| 从当前纯真实parent继续训练 | CER%，三seed均值 | 相对parent，pp |',
        '|---|---:|---:|',
        f'| 不继续训练 | {means["real_w0"]["overall"]:.4f} | — |']
    for arm,name in [('real32','追加32条真实、1步'),('synthetic32','追加历史32条TTS、1步')]:
        m=post['means'][arm];lines.append(f'| {name} | {m["cer_pct"]:.4f} | {m["delta_pp"]:+.4f} |')
    lines+=['',
        '这批TTS是历史另一轮MAS的6个文本，不等于当前576条。旧单模型最佳扫描的收益没有在新三个parent上复现；其数据原本针对旧M0产生，不能据此否定重新针对当前模型生成内容的循环。'+link('短步配对结果',ROOT/'post_small/results.json')+'。','',
        '## 代码和证据','',
        link('实施与实验协议说明',Path('/data/qwen3-asr/repo/dysarthric_yue_Speech/docs/tts_utility_diagnosis_v1.zh.md'))+'；'+
        link('运行结果JSON',ROOT/'current_results.json')+'；'+link('文件完整性核验',ROOT/'integrity_snapshot.json')+'。','',
        '本轮修复和消融以独立脚本、manifest和checkpoint落地。原训练器、生成器和历史数据没有被静默覆盖。尚未由人工确认的参考残留、临床发音保留程度，以及新内容上的独立泛化效果，均保留为未解决环节。','']
    (ROOT/'findings.zh.md').write_text('\n'.join(lines))
    print('Wrote',ROOT/'findings.zh.md',flush=True)


if __name__=='__main__':run()
