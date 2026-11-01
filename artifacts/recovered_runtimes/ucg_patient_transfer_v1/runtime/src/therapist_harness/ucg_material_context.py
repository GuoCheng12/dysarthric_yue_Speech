"""Read-only dev support for selecting existing UCG training recordings."""
from collections import defaultdict
from copy import deepcopy
from .material_selection_offline import DiagnosticBrief
from .sampling_curriculum_data import _join_predictions
from .sampling_phonology import pronunciation
from .text import text_normalize

def _syllables(rows, focus):
    field = 'jyutping' if focus.level == 'syllable' else focus.level
    return sorted({p.jyutping for row in rows for p in pronunciation(row['clean_gt']).positions
                   if p.resolved and (focus.level == 'text' and row['clean_gt'] == focus.target_unit
                       or focus.level != 'text' and getattr(p, field) == focus.target_unit)})

def annotate_ucg_candidates(pack, *, index, feedback_rows, current_predictions,
                            previous_predictions=None, checkpoint_step):
    from .material_sampling_mas import measure_focus
    train_ids = {r['recording_id'] for r in index['recordings']}
    train_patients = {r['patient_id'] for r in index['recordings']}
    train_texts = {text_normalize(p['target_text']) for p in index['profiles'].values()}
    if train_ids & {r['utt_id'] for r in feedback_rows}:
        raise ValueError('dev recording entered training pool')
    if train_texts & {text_normalize(r['clean_gt']) for r in feedback_rows}:
        raise ValueError('UCG requires disjoint complete texts')
    if not {r['speaker_id'] for r in feedback_rows} <= train_patients:
        raise ValueError('UCG requires dev patients represented in training')
    current = _join_predictions(feedback_rows, current_predictions)
    previous = _join_predictions(feedback_rows, previous_predictions) if previous_predictions is not None else None
    now_by, old_by = defaultdict(list), defaultdict(list)
    for row in current: now_by[row['speaker_id']].append(row)
    for row in previous or []: old_by[row['speaker_id']].append(row)
    anon = {p: f'DEV{i:03d}' for i,p in enumerate(sorted(now_by),1)}
    brief = DiagnosticBrief.model_validate(pack['diagnosis'])
    focuses = {f.focus_id:f for f in brief.focuses}
    result = deepcopy(pack)
    result['ucg_material_task'] = {
        'setting': 'UCG: same patients, different complete sentences',
        'checkpoint_step': checkpoint_step,
        'feedback_role': 'dev',
        'meaning': '选择已有训练录音，尝试让在某些句子观察到的读音问题迁移到其他句子；不把完整dev句子当训练材料。',
        'measurement_limits': 'target_differences只统计文本对齐支持的目标替换差异；不是听音诊断，不包含所有目标删除。cohort_CER统计含目标成分的整句错误。无观察不等于无错误。'
    }
    def compact(rows, focus):
        m = measure_focus(rows, focus, index);c=m['cohort']
        return dict(recordings=c['audio'],cohort_CER=c['CER_percent'],
                    target_occurrences=m['target_occurrences'],
                    target_differences=sum(m['component_differences'].values()) if m['component_differences'] is not None else None)
    for pool in result['focuses']:
        focus = focuses[pool['focus_id']]
        all_units = set(_syllables(current, focus))
        pool['observed_dev_target_syllables'] = sorted(all_units)
        pool['dev_patient_evidence'] = {}
        for candidate in pool['candidates']:
            patient = candidate['patient_id']
            units = {p['jyutping'] for p in candidate['matching_positions']}
            evidence_id = f'step{checkpoint_step}:{anon[patient]}' if patient in anon else None
            candidate['ucg_match'] = dict(
                patient_dev_evidence_id=evidence_id,
                shared_dev_target_syllables=sorted(units & all_units),
                same_patient_shared_target_syllables=sorted(units & set(_syllables(now_by[patient],focus))) if patient in anon else [])
            if evidence_id is not None and evidence_id not in pool['dev_patient_evidence']:
                pool['dev_patient_evidence'][evidence_id] = dict(
                    current=compact(now_by[patient],focus),
                    previous=compact(old_by[patient],focus) if previous is not None else None)
    return result
