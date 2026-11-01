"""Post-hoc fallback proposal using the existing old-audio QC threshold."""
import json
from pathlib import Path
import statistics

ROOT = Path('/data/qwen3-asr/records/therapist_mas_setting_d_v1/generator_transfer_probe_v2')


def run():
    data = json.loads((ROOT/'results.json').read_text())
    details = data['details']; ids = list(details['real_asr']['old_sg'])
    assert len(ids) == 96
    choices = {k: 'old_sg' if details['real_asr']['old_sg'][k]['cer'] <= .25 else 'sg_instruct2' for k in ids}
    subsets = {}
    for subset, keys in [('old_qc_pass', [k for k in ids if choices[k] == 'old_sg']),
                         ('old_qc_fail', [k for k in ids if choices[k] == 'sg_instruct2'])]:
        subsets[subset] = {'n': len(keys), 'judges': {}}
        for judge, models in details.items():
            values = {}
            for mode in models:
                real_positions = sum(len(models['real'][k]['positions']) for k in keys)
                shared = sum(len(set(models['real'][k]['positions']) & set(models[mode][k]['positions'])) for k in keys)
                values[mode] = {'cer_pct': statistics.mean(models[mode][k]['cer'] for k in keys)*100,
                    'real_error_position_recall': shared/real_positions if real_positions else None}
            subsets[subset]['judges'][judge] = values
    audit = {'definition': 'Post-hoc diagnostic strata using the pre-existing old-audio QC threshold<=.25 from the frozen R1541 judge. Selection favors old audio on this judge; base judge is same architecture and not independent clinical truth. These subsets do not replace the full96 primary analysis.', 'results': subsets}
    (ROOT/'qc_conditioned_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n')
    failed = [k for k in ids if choices[k] == 'sg_instruct2']
    recovered = [k for k in failed if details['real_asr']['sg_instruct2'][k]['cer'] <= .25]
    candidate = {'status': 'post-hoc candidate rule, not prospectively validated or ASR-finetuned',
        'rule': 'Keep old SG zero-shot when frozen R1541 CER<=.25; otherwise use SG instruct2 as a content-repair candidate. Apply the same .25 gate afterwards for production. Full96 analysis below retains every pair, including final failures.',
        'old_pass': 96-len(failed), 'fallback_attempts': len(failed), 'fallback_pass': len(recovered),
        'total_pass': 96-len(failed)+len(recovered), 'still_fail': [k for k in failed if k not in recovered],
        'metrics': {}, 'choices': choices}
    for judge, models in details.items():
        rows = [models[choices[k]][k] for k in ids]
        candidate['metrics'][judge] = {'full96_cer_pct': statistics.mean(r['cer'] for r in rows)*100,
            'perfect': sum(r['cer'] == 0 for r in rows),
            'reference_fragment_clues': sum(bool(r['reference_fragments']) for r in rows),
            'insertions': sum(r['insertions'] for r in rows), 'deletions': sum(r['deletions'] for r in rows)}
    (ROOT/'fallback_candidate.json').write_text(json.dumps(candidate, ensure_ascii=False, indent=2)+'\n')
    print({k: v for k, v in candidate.items() if k != 'choices'})


if __name__ == '__main__': run()
