"""Keep all exact-to-wrong transitions separate from the three fixed semantic rules."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/data/qwen3-asr/records/therapist_mas_setting_d_v1/whisper_large_v3_real2117_lora_seed42_4configs_20260910_v2')
OUT = ROOT / 'correctness_transition_analysis'


def read(path):
    return json.loads(path.read_text())


def main():
    base = read(ROOT / 'error_analysis/base.json')
    result, sources = {}, {}
    lines = ['# 全部早对后错候选', '',
             f'结果快照：{datetime.now(timezone.utc).isoformat()}。仅包含当时已完成的评估；训练仍可能继续。', '',
             '这份补充分析保留所有“同一音频曾完全正确、后来不完全正确”的记录，不改变原有三个特定词组规则。',
             '这些候选包括同音字、漏字、语义换词和其他错误；不能全部称为更自然的推断。曾经正确的样本数因配置而异，原始候选数量不能直接用于判定哪组更容易退化。', '',
             '|配置|step|当前完全正确|截至当前曾经正确|曾正确而当前错误|原始模型正确而当前错误|',
             '|---|---:|---:|---:|---:|---:|']
    for name in ['W-A', 'W-B', 'W-C', 'W-D']:
        first = {r['utt_id']: {'step': 0, 'prediction': r['prediction']} for r in base['rows'] if r['exact']}
        base_correct = set(first)
        timeline = []
        paths = sorted((ROOT / 'error_analysis' / name).glob('step-*.json'), key=lambda p: int(p.stem.split('-')[1]))
        for path in paths:
            step = int(path.stem.split('-')[1]); audit = read(path)
            sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            transitions = []
            for row in audit['rows']:
                if row['exact']:
                    first.setdefault(row['utt_id'], {'step': step, 'prediction': row['prediction']})
                elif row['utt_id'] in first:
                    transitions.append(dict(utt_id=row['utt_id'], target=row['target'], prediction=row['prediction'],
                                            errors=row['errors'], earlier_correct=first[row['utt_id']],
                                            existing_three_rule_marker=row['marker']))
            state = dict(step=step, currently_exact=sum(r['exact'] for r in audit['rows']), ever_exact=len(first),
                         currently_wrong_after_exact=len(transitions),
                         raw_exact_now_wrong=sum(r['utt_id'] in base_correct for r in transitions), transitions=transitions)
            timeline.append(state)
            lines.append(f"|{name}|{step}|{state['currently_exact']}|{len(first)}|{len(transitions)}|{state['raw_exact_now_wrong']}|")
        result[name] = timeline
    lines += ['', '## 人工查看的语义相近换词实例', '',
              '以下固定选取 W-C step148 的两个实例。它们早期与标注一致，后来换了表达“拿取”的词；此处只比较转写和标注，没有重新听音核验。', '',
              '|音频|更早正确 step|目标与更早转写|step148 转写|', '|---|---:|---|---|']
    selected = {'vlink212/4989761755650311442999', 'vlink234/6756601757415821949230'}
    for stage in result.get('W-C', []):
        if stage['step'] == 148:
            for r in stage['transitions']:
                if r['utt_id'] in selected:
                    lines.append(f"|{r['utt_id']}|{r['earlier_correct']['step']}|{r['target']}|{r['prediction']}|")
    lines += ['', '同一 step148 也有相反的例子：W-A 将“人生曲折”写成“人身曲折”，W-C 将“蠍子”写成“竊子”。这些转写没有明显变得更合乎语境，不能把全部早对后错候选都解释成语言模型的语义合理化。完整候选及音频身份见 all_transitions.json。']
    OUT.mkdir(exist_ok=True)
    (OUT / 'all_transitions.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    (OUT / 'report.zh.md').write_text('\n'.join(lines) + '\n')
    (OUT / 'provenance.json').write_text(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),
        audit_sha256=sources, base_audit_sha256=hashlib.sha256((ROOT / 'error_analysis/base.json').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({name: {k: v for k, v in stages[-1].items() if k != 'transitions'} for name, stages in result.items() if stages}))


if __name__ == '__main__':
    main()
