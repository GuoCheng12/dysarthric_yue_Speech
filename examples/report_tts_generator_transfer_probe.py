"""Paired content, error-pattern, duration and listening evidence for the96 probe."""
import base64
from collections import Counter, defaultdict
import html
import json
from pathlib import Path
import random
import statistics
import sys

import numpy as np
import soundfile as sf
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'src'))
from therapist_harness.text import text_normalize, edit_distance
from therapist_harness.text_scoring_v2 import text_normalize_v2
from tts_generator_transfer_probe import ROOT, B, RECIPES, digest, read

MODES = ['real', 'old_sg'] + RECIPES
NAMES = {'real': '真实录音', 'old_sg': '旧 SG zero-shot', 'sg_instruct2': 'SG instruct2 粤语', 'base_instruct2': 'Base instruct2 粤语'}


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def alignment(ref, hyp):
    d = [[0]*(len(hyp)+1) for _ in range(len(ref)+1)]
    for i in range(len(ref)+1): d[i][0] = i
    for j in range(len(hyp)+1): d[0][j] = j
    for i, a in enumerate(ref, 1):
        for j, b in enumerate(hyp, 1):
            d[i][j] = min(d[i-1][j-1]+(a != b), d[i-1][j]+1, d[i][j-1]+1)
    i, j = len(ref), len(hyp)
    events = []
    while i or j:
        if i and j and d[i][j] == d[i-1][j-1]+(ref[i-1] != hyp[j-1]):
            if ref[i-1] != hyp[j-1]: events.append(('S', i-1, ref[i-1], hyp[j-1]))
            i -= 1; j -= 1
        elif i and d[i][j] == d[i-1][j]+1:
            events.append(('D', i-1, ref[i-1], '')); i -= 1
        else:
            events.append(('I', i, '', hyp[j-1])); j -= 1
    assert len(events) == edit_distance(ref, hyp) == d[-1][-1]
    counts = Counter(e[0] for e in events)
    return {'cer': len(events)/len(ref), 'substitutions': counts['S'], 'deletions': counts['D'],
            'insertions': counts['I'], 'events': events, 'positions': sorted({e[1] for e in events})}


def summarize(group):
    return {'n': len(group), 'cer_pct': 100*statistics.mean(r['cer'] for r in group),
        'variant_cer_pct': 100*statistics.mean(r['variant_cer'] for r in group),
        'perfect': sum(r['cer'] == 0 for r in group), 'over50': sum(r['cer'] > .5 for r in group),
        'insertions': sum(r['insertions'] for r in group), 'deletions': sum(r['deletions'] for r in group),
        'substitutions': sum(r['substitutions'] for r in group),
        'reference_fragment_clues': sum(bool(r['reference_fragments']) for r in group)}


def run():
    cfg = read(ROOT/'execution.json'); cohort = cfg['cases']; cases = {p['sample_id']: p for p in cohort}
    ids = list(cases); patients = sorted({p['speaker_id'] for p in cohort})
    for path, expected in cfg['source_and_input_sha256'].items(): assert digest(path) == expected, path
    for mode in RECIPES: assert read(ROOT/mode/'monitor.json')['status'] == 'complete'
    details = {judge: {mode: {} for mode in MODES} for judge in ['real_asr', 'base_asr']}
    judge_hashes = {}
    for judge in details:
        cached = B/'tts_utility_diagnosis_v1/screening'/('asr' if judge == 'real_asr' else 'base_asr')/'predictions.jsonl'
        judge_hashes[str(cached)] = digest(cached)
        selected = [r for r in rows(cached) if r['pair_id'] in cases and r['variant'] in ['q_real','q_hard']]
        assert len(selected) == 192
        inputs = {mode: {r['pair_id']: r for r in rows(ROOT/mode/'input.jsonl')} for mode in RECIPES}
        for mode in RECIPES:
            file = ROOT/mode/judge/'predictions.jsonl'
            judge_hashes[str(file)] = digest(file)
            selected.extend(rows(file))
        for row in selected:
            mode = {'q_real': 'real', 'q_hard': 'old_sg'}.get(row['variant'], row['variant'])
            pair = cases[row['pair_id']]
            assert row['speaker_id'] == pair['speaker_id'] and row['clean_gt'] == pair['target']['clean_gt']
            if mode == 'real': expected_audio = pair['target']['audio']
            elif mode == 'old_sg': expected_audio = pair['existing_sg_zero_shot']['audio']
            else: expected_audio = inputs[mode][row['pair_id']]['audio']
            assert row['audio'] == expected_audio
            assert digest(row['audio']) == row['audio_sha256']
            ref, hyp, prompt = map(text_normalize, [row['clean_gt'], row['prediction'], pair['reference']['reference_text']])
            a = alignment(ref, hyp)
            assert abs(a['cer'] - row['cer']) < 1e-12
            vr, vh = map(text_normalize_v2, [row['clean_gt'], row['prediction']])
            a['variant_cer'] = edit_distance(vr, vh)/len(vr)
            fragments = sorted({prompt[i:i+4] for i in range(max(0, len(prompt)-3)) if prompt[i:i+4] in hyp and prompt[i:i+4] not in ref})
            assert row['pair_id'] not in details[judge][mode]
            details[judge][mode][row['pair_id']] = {**a, 'prediction': row['prediction'], 'audio': row['audio'],
                'reference_fragments': fragments, 'bucket': pair['difficulty_bucket'], 'speaker_id': pair['speaker_id']}
        assert all(set(rr) == set(ids) for rr in details[judge].values())
    metrics = {}; comparisons = {}; patterns = {}
    rng = np.random.default_rng(20260908)
    weights = rng.multinomial(len(patients), np.full(len(patients), 1/len(patients)), size=20000)
    patient_index = {p: i for i, p in enumerate(patients)}
    pi = np.asarray([patient_index[cases[k]['speaker_id']] for k in ids])
    denominator = weights[:, pi].sum(axis=1)
    for judge, modes in details.items():
        metrics[judge] = {}; comparisons[judge] = {}; patterns[judge] = {}
        for mode, records in modes.items():
            metrics[judge][mode] = {bucket: summarize([records[k] for k in ids if bucket == 'overall' or cases[k]['difficulty_bucket'] == bucket])
                for bucket in ['overall', 'easy', 'medium', 'hard']}
            if mode != 'real':
                real_positions = sum(len(modes['real'][k]['positions']) for k in ids)
                synth_positions = sum(len(records[k]['positions']) for k in ids)
                shared = sum(len(set(modes['real'][k]['positions']) & set(records[k]['positions'])) for k in ids)
                real_events = sum(len(set(map(tuple, modes['real'][k]['events']))) for k in ids)
                shared_events = sum(len(set(map(tuple, modes['real'][k]['events'])) & set(map(tuple, records[k]['events']))) for k in ids)
                patterns[judge][mode] = {'real_error_positions': real_positions, 'synthetic_error_positions': synth_positions,
                    'shared_positions': shared, 'position_recall': shared/real_positions if real_positions else None,
                    'position_precision': shared/synth_positions if synth_positions else None,
                    'real_unique_error_events': real_events, 'shared_exact_events': shared_events,
                    'real_wrong_synth_correct': sum(modes['real'][k]['cer'] > 0 and records[k]['cer'] == 0 for k in ids),
                    'real_correct_synth_wrong': sum(modes['real'][k]['cer'] == 0 and records[k]['cer'] > 0 for k in ids),
                    'cer_spearman': float(spearmanr([modes['real'][k]['cer'] for k in ids], [records[k]['cer'] for k in ids]).statistic)}
                patterns[judge][mode]['by_real_bucket'] = {}
                for bucket in ['easy', 'medium', 'hard']:
                    kk = [k for k in ids if cases[k]['difficulty_bucket'] == bucket]
                    rp = sum(len(modes['real'][k]['positions']) for k in kk)
                    overlap = sum(len(set(modes['real'][k]['positions']) & set(records[k]['positions'])) for k in kk)
                    patterns[judge][mode]['by_real_bucket'][bucket] = {'real_error_positions': rp,
                        'shared_positions': overlap, 'position_recall': overlap/rp if rp else None,
                        'real_wrong_synth_correct': sum(modes['real'][k]['cer'] > 0 and records[k]['cer'] == 0 for k in kk),
                        'real_correct_synth_wrong': sum(modes['real'][k]['cer'] == 0 and records[k]['cer'] > 0 for k in kk)}
            if mode in RECIPES:
                delta = np.asarray([records[k]['cer'] - modes['old_sg'][k]['cer'] for k in ids])
                boot = weights[:, pi] @ delta / denominator * 100
                first_attempt = [k for k in ids if inputs[mode][k]['generation_attempt'] == 0]
                comparisons[judge][mode] = {'baseline': 'old_sg', 'delta_pp': float(delta.mean()*100),
                    'patient_bootstrap95_pp': np.quantile(boot, [.025,.975]).tolist(),
                    'improved_pairs': int((delta < 0).sum()), 'worse_pairs': int((delta > 0).sum()),
                    'same_pairs': int((delta == 0).sum()), 'first_attempt_n': len(first_attempt),
                    'first_attempt_delta_pp': statistics.mean(records[k]['cer']-modes['old_sg'][k]['cer'] for k in first_attempt)*100}
    acoustics = {}
    duration = {}
    for mode in MODES:
        duration[mode] = {}
        for key in ids:
            wave, sr = sf.read(details['real_asr'][mode][key]['audio'])
            assert len(wave) > 0 and np.isfinite(wave).all()
            duration[mode][key] = len(wave)/sr
        acoustics[mode] = {'median_seconds': statistics.median(duration[mode].values()),
            'under1second': sum(v < 1 for v in duration[mode].values())}
        if mode != 'real':
            acoustics[mode]['median_duration_ratio_to_real'] = {bucket: statistics.median(duration[mode][k]/duration['real'][k] for k in ids if bucket == 'overall' or cases[k]['difficulty_bucket'] == bucket) for bucket in ['overall','easy','medium','hard']}
            acoustics[mode]['duration_spearman_with_real'] = float(spearmanr([duration['real'][k] for k in ids], [duration[mode][k] for k in ids]).statistic)
    technical = {mode: {'summary': read(ROOT/mode/'generation_summary.json'),
        'retried_pairs': [k for k in ids if len(read(ROOT/mode/'receipts'/f'{k}.json')) > 1]} for mode in RECIPES}
    output = {'scope': 'fixed96 training pairs, 46 patients, equal real-ASR-difficulty strata; not population prevalence',
        'judges': 'same ASR family, base and R1541-tuned; not independent architectures or manual transcription',
        'limits': 'Error alignment and duration are patient-pattern proxies, not clinical impairment validation. No ASR finetuning performed. Bootstrap conditional on fixed judges and this diagnostic cohort.',
        'source_sha256': digest(__file__), 'judge_predictions_sha256': judge_hashes,
        'metrics': metrics, 'paired_vs_old': comparisons, 'error_patterns': patterns,
        'duration': acoustics, 'technical': technical, 'details': details}
    (ROOT/'results.json').write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n')
    lines = ['# 96配对 TTS 生成方案验收', '', '96条训练目标、46位患者；真实ASR难度各32条。全部结果已完成。', '',
        '| 生成方式 | R-only筛查 CER% | Base筛查 CER% | R-only筛查：Easy / Medium / Hard |', '|---|---:|---:|---|']
    for mode in MODES:
        m = metrics['real_asr'][mode]
        lines.append(f"| {NAMES[mode]} | {m['overall']['cer_pct']:.3f} | {metrics['base_asr'][mode]['overall']['cer_pct']:.3f} | "+' / '.join(f"{m[g]['cer_pct']:.3f}" for g in ['easy','medium','hard'])+' |')
    lines += ['', '以上CER是对生成/真实音频的筛查结果，不是新ASR微调后的性能。', '', '## 对旧生成方案的配对差值', '']
    for judge in comparisons:
        for mode, c in comparisons[judge].items():
            lines.append(f"- {judge}，{NAMES[mode]}：{c['delta_pp']:+.3f}pp；患者聚类95%区间{c['patient_bootstrap95_pp']}；改善/退化/相同={c['improved_pairs']}/{c['worse_pairs']}/{c['same_pairs']}。")
    lines += ['', '## 与真实录音的错误和时长对应', '', '| 生成方式 | 真实错误位置召回率（R筛查） | CER秩相关（R筛查） | 时长/真实，中位数 |', '|---|---:|---:|---:|']
    for mode in MODES[1:]:
        p = patterns['real_asr'][mode]
        lines.append(f"| {NAMES[mode]} | {100*p['position_recall']:.2f}% | {p['cer_spearman']:.3f} | {acoustics[mode]['median_duration_ratio_to_real']['overall']:.3f} |")
    lines += ['', '错误对应率和时长不是临床保真度指标；更低CER或更高错误对应率均不能单独决定采用哪个方案。', '',
        'A/B比较同时改变指令与参考token接入；B/C在相同instruct2模式下比较适配器。原有4条历史重试的实际种子已对齐，所有新技术重试保留记录。', '',
        f'[完整JSON]({ROOT}/results.json) · [盲听页]({ROOT}/listening.html)', '']
    (ROOT/'report.zh.md').write_text('\n'.join(lines))
    listening(cohort, details)
    print('\n'.join(lines), flush=True)


def listening(cohort, details):
    # Outcome-independent selection: four patients per real difficulty stratum.
    chosen = []
    for bucket in ['easy','medium','hard']:
        used = set()
        for pair in cohort:
            if pair['difficulty_bucket'] == bucket and pair['speaker_id'] not in used:
                chosen.append(pair); used.add(pair['speaker_id'])
                if len(used) == 4: break
    parts = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>TTS配对验收试听</title>',
        '<style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 20px;line-height:1.6}article{border-top:1px solid #bbb;padding:18px 0}.clip{background:#f3f5f7;padding:12px;margin:10px 0}audio{width:100%}summary{cursor:pointer}small{color:#555}</style>',
        '<h1>TTS配对验收：12例盲听</h1><p>按预定清单每种真实ASR难度取4位患者，未按结果挑选。先听真实录音，再比较随机排列的三个候选；展开后可查看方案和ASR预测。请关注目标内容、额外词句、停顿与患者发音特点的变化。ASR预测并非人工转录。</p>']
    key_map = {}
    def player(path):
        value = base64.b64encode(Path(path).read_bytes()).decode('ascii')
        return f'<audio controls preload="none" src="data:audio/wav;base64,{value}"></audio>'
    rng = random.Random(20260908)
    for pair in chosen:
        key = pair['sample_id']; order = MODES[1:].copy(); rng.shuffle(order)
        key_map[key] = dict(zip(['A','B','C'], order))
        parts.append(f'<article><h2>{key} · {html.escape(pair["speaker_id"])} · {pair["difficulty_bucket"]}</h2><p>目标：{html.escape(pair["target"]["clean_gt"])}</p><h3>真实录音</h3>'+player(pair['target']['audio']))
        parts.append('<details><summary>参考录音及其文字</summary><p>'+html.escape(pair['reference']['reference_text'])+'</p>'+player(pair['reference']['reference_wav'])+'</details>')
        for label, mode in zip(['A','B','C'], order):
            a = details['real_asr'][mode][key]; b = details['base_asr'][mode][key]
            parts.append(f'<div class="clip"><strong>候选 {label}</strong>'+player(a['audio'])+f'<details><summary>揭示方案与ASR预测</summary><p>{NAMES[mode]}</p><p>R-only：{html.escape(a["prediction"])}</p><p>Base：{html.escape(b["prediction"])}</p></details></div>')
        parts.append('</article>')
    parts.append('</html>')
    (ROOT/'listening.html').write_text('\n'.join(parts))
    (ROOT/'listening_key.json').write_text(json.dumps(key_map, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__': run()
