#!/usr/bin/env python3
"""Describe the held-content evaluation and diagnose concentration across prompts."""
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mas_synthetic_fixed_subset_replication_v1 as run
from therapist_harness.text import text_normalize
import therapist_harness.text as normalization


def main():
    assert run.read(run.REPLICATION / 'results.json')['status'] == 'complete_fixed_SG_subset_optimization_seed_replication'
    train_path = Path('/data/qwen3-asr/finetune/data_setting_d_v1/setting_d_train.jsonl')
    train = run.rows(train_path)
    initial = run.rows(run.live.BASELINE)
    seen = Counter(text_normalize(r['clean_gt']) for r in train)
    prompts = sorted({text_normalize(r['clean_gt']) for r in initial})
    patients = sorted({r['speaker_id'] for r in initial})
    assert len(train) == 2117 and len(initial) == 511
    assert len(seen) == 48 and len(prompts) == 12 and not (set(seen) & set(prompts))
    paths = dict(initial=run.live.BASELINE,
        R42=run.OLD / 'R_DPO_KL/evaluation_run/evaluation/predictions.jsonl.gz',
        R_SG42=run.CANDIDATE / 'uniform/evaluation/predictions.jsonl.gz',
        MAS42=run.CANDIDATE / 'MAS/evaluation/predictions.jsonl.gz',
        **{f'{mode}{seed}': run.REPLICATION / f'seed_{seed}' / mode / 'evaluation/predictions.jsonl.gz'
           for seed in [43, 44] for mode in ['R', 'R_SG']})
    ids = [r['utt_id'] for r in initial]
    values = {}
    for name, path in paths.items():
        rows = run.live.gzip_rows(path)
        assert [r['utt_id'] for r in rows] == ids
        assert [r['clean_gt'] for r in rows] == [r['clean_gt'] for r in initial]
        values[name] = np.array([r['edit_count'] for r in rows], dtype=np.float64)
    values['R_mean'] = sum(values[f'R{s}'] for s in [42, 43, 44]) / 3
    values['R_SG_mean'] = sum(values[f'R_SG{s}'] for s in [42, 43, 44]) / 3
    chars = np.array([r['normalized_reference_characters'] for r in initial], dtype=np.float64)
    pi = np.array([prompts.index(text_normalize(r['clean_gt'])) for r in initial])
    si = np.array([patients.index(r['speaker_id']) for r in initial])
    per_prompt = []
    for index, target in enumerate(prompts):
        mask = pi == index
        per_prompt.append(dict(prompt_id=f'P{index + 1:02}', normalized_text=target,
            audio=int(mask.sum()), patients=len(set(si[mask].tolist())), reference_characters=int(chars[mask].sum()),
            errors={key: float(value[mask].sum()) for key, value in values.items()}))
    # Diagnostic crossed resampling: same patient and prompt multiplicities applied to both models.
    # It supplements, and never replaces, the already fixed original patient-based gate.
    rng = np.random.default_rng(420910)
    n = 4000
    prompt_counts = rng.multinomial(len(prompts), [1 / len(prompts)] * len(prompts), size=n)
    patient_counts = rng.multinomial(len(patients), [1 / len(patients)] * len(patients), size=n)
    prompt_weights = prompt_counts[:, pi]
    crossed_weights = prompt_weights * patient_counts[:, si]
    comparisons = {}
    for before, after in [('initial', 'R_SG42'), ('R_SG42', 'MAS42'),
                           ('initial', 'R_SG_mean'), ('R_mean', 'R_SG_mean')]:
        delta = values[after] - values[before]
        summary = dict(delta_pp=float(100 * delta.sum() / chars.sum()))
        for name, weights in [('prompt_cluster', prompt_weights), ('crossed_patient_prompt', crossed_weights)]:
            denominator = weights @ chars
            assert np.all(denominator > 0)
            bootstrap = 100 * (weights @ delta) / denominator
            summary[name + '_diagnostic_95'] = np.quantile(bootstrap, [.025, .975]).tolist()
        comparisons[before + '_to_' + after] = summary
    out = run.REPLICATION / 'content_transfer'
    out.mkdir(exist_ok=False)
    sources = [Path(__file__), Path(normalization.__file__), train_path,
               run.REPLICATION / 'results.json', *paths.values()]
    result = dict(status='complete_content_scope_and_transfer_diagnostic', real_train_audio=2117,
        real_train_unique_normalized_texts=48, feedback_audio=511, feedback_unique_normalized_texts=12,
        feedback_complete_sentence_overlap_with_train=0, per_prompt=per_prompt, comparisons=comparisons,
        original_acceptance_unchanged=True, promoted=False, bootstrap_replicates=n,
        scope='Descriptive content grouping and diagnostic prompt/crossed-cluster bootstrap on reused511dev. Fixed-data optimization-seed means; not new training, an independent test, or an additional checkpoint gate.',
        source_sha256={str(p): run.sha256_file(p) for p in sources})
    run.write_immutable_json(out / 'results.json', result)
    lines = ['按开发内容分组的核对', '',
        '2117条真实训练语音对应48个不同规范化句子；511条开发语音对应12个句子，整句没有重合。同一句开发内容有多名患者和部分重复录音。', '',
        '| 内容 | 开发目标文本（按既有CER规范化） | 音频 | initial字错 | R三seed均错 | R+SG三seed均错 | R+SG减initial |',
        '|---|---|---:|---:|---:|---:|---:|']
    for row in per_prompt:
        e = row['errors']
        lines.append(f'| {row["prompt_id"]} | {row["normalized_text"]} | {row["audio"]} | {e["initial"]:.0f} | {e["R_mean"]:.2f} | {e["R_SG_mean"]:.2f} | {e["R_SG_mean"]-e["initial"]:+.2f} |')
    lines += ['', '跨患者/内容的诊断区间不会改变原验收规则。这里没有剔除任何患者、内容或改动GT，也没有使用这些分组重新挑选模型。', '']
    for name, value in comparisons.items():
        lines.append(f'{name}：{value["delta_pp"]:+.6f}个百分点；按内容bootstrap95%区间{value["prompt_cluster_diagnostic_95"]}；患者×内容交叉重采样区间{value["crossed_patient_prompt_diagnostic_95"]}。')
    with (out / 'results.zh.md').open('x') as f:
        f.write('\n'.join(lines) + '\n')
    print(json.dumps(comparisons), flush=True)


if __name__ == '__main__':
    main()
