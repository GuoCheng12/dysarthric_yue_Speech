"""Describe transfer across the 12 fixed development sentences from saved predictions."""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/data/qwen3-asr/records/therapist_mas_setting_d_v1')
W = ROOT / 'whisper_large_v3_real2117_lora_seed42_4configs_20260910_v2'
Q = ROOT / 'qwen3_asr_fresh_G_lr2em4_2gpu_seed42_50epochs_20260910_v1'
OUT = W / 'sentence_transfer_analysis'


def read(path):
    return json.loads(path.read_text())


def stages(root, branch):
    path = root / branch / 'events.jsonl'
    if not path.exists():
        return []
    return [e for line in path.read_text().splitlines(keepends=True)
            if line.endswith('\n') and line.strip()
            for e in [json.loads(line)] if e['stage'] == 'evaluation_complete']


def main():
    dev = {x['utt_id']: x for x in read(W / 'dev.json')}
    targets = sorted({x['clean_gt'] for x in dev.values()})
    assert len(dev) == 511 and len(targets) == 12
    inputs = [('Whisper raw', 0, W / 'base_evaluation/predictions.json'),
              ('Qwen raw', 0, Q / 'base_evaluation/predictions.json'),
              ('Qwen R-only', 0, ROOT / 'qwen3_asr_lora_sft_audit_20260910_v2/evaluations/000_continue_original/dev_merged.predictions.json')]
    selected = {}
    for root, names in [(W, ['W-A', 'W-B', 'W-C', 'W-D']), (Q, ['G_audio_lr2em4_2gpu_50epochs'])]:
        for name in names:
            ev = stages(root, name)
            if not ev:
                continue
            selected[name] = {'best': min(ev, key=lambda e: e['metrics']['CER_percent'])['step'], 'latest': ev[-1]['step']}
            inputs.extend((name, e['step'], root / name / 'evaluations' / f"step-{e['step']}" / 'predictions.json') for e in ev)
    rows, sources = [], []
    for name, step, path in inputs:
        predictions = read(path)
        assert len(predictions) == 511 and {x['utt_id'] for x in predictions} == set(dev)
        sources.append({'model': name, 'step': step, 'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        for target in targets:
            xs = [x for x in predictions if dev[x['utt_id']]['clean_gt'] == target]
            errors, chars = sum(x['errors'] for x in xs), sum(x['chars'] for x in xs)
            rows.append(dict(model=name, step=step, target=target, audio=len(xs), errors=errors, chars=chars,
                             CER_percent=100 * errors / chars, exact_audio=sum(x['errors'] == 0 for x in xs)))
    lookup = {(r['model'], r['step'], r['target']): r for r in rows}
    lines = ['# 未见目标句的适配收益', '',
             f'结果快照：{datetime.now(timezone.utc).isoformat()}。仅包含当时已完成的评估；训练仍可能继续。', '',
             '同一批 511 条开发音频，按 12 个固定目标句拆分；这些句子不在训练文本中，患者与训练集有重叠。',
             '这是已有开发集上的句子迁移描述，不能单独证明学到了某种发音规律，也不是未见患者泛化评估。',
             '全部音频均计入；包括长循环。最佳点按总体 CER 选取，不能逐句挑选不同 checkpoint。', '',
             '|配置/阶段|step|比自身原始模型改善的句子|恶化|持平|', '|---|---:|---:|---:|---:|']
    for name, choices in selected.items():
        base = 'Whisper raw' if name.startswith('W-') else 'Qwen raw'
        for label, step in choices.items():
            deltas = [lookup[(name, step, t)]['errors'] - lookup[(base, 0, t)]['errors'] for t in targets]
            lines.append(f'|{name} {label}|{step}|{sum(d < 0 for d in deltas)}/12|{sum(d > 0 for d in deltas)}|{sum(d == 0 for d in deltas)}|')
    columns = [('Whisper raw', 0), ('Qwen raw', 0), ('Qwen R-only', 0)] + [(n, s['latest']) for n, s in selected.items()]
    lines += ['', '## 最新 checkpoint 逐句 CER', '',
              '|目标句|音频数|' + '|'.join(f'{n} step{s}' for n, s in columns) + '|',
              '|---|---:|' + '|'.join('---:' for _ in columns) + '|']
    for target in targets:
        values = [lookup[(n, s, target)] for n, s in columns]
        lines.append(f"|{target}|{values[0]['audio']}|" + '|'.join(f"{v['CER_percent']:.2f}%" for v in values) + '|')
    OUT.mkdir(exist_ok=True)
    with (OUT / 'all_stages.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (OUT / 'provenance.json').write_text(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),
        selected_checkpoints=selected, source_predictions=sources,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), ensure_ascii=False, indent=2) + '\n')
    (OUT / 'report.zh.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines[:19]))


if __name__ == '__main__':
    main()
