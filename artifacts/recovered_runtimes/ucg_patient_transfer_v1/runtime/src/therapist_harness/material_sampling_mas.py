"""Dev diagnosis -> concrete train recordings -> one sampling mixture -> reflection."""
from collections import Counter
from dataclasses import asdict
import copy
import json
import math
from pathlib import Path

from openai.lib._pydantic import to_strict_json_schema
from pydantic import Field

from .material_selection_offline import (
    DiagnosticBrief, MaterialSelection, build_selector_request, digest, matches,
    prepare_candidates, validate_selection,
)
from .real_sampling_mas import (
    SamplingModel, SamplingReflection, SamplingExperience, _call, payload_hash,
)
from .sampling_curriculum_data import _join_predictions, summarize_sampling_rows
from .sampling_curriculum_runtime import ExposureSampler
from .sampling_phonology import LEVELS, pronunciation, transcript_contrasts
from .store import write_immutable_json

PROMPTS = Path(__file__).with_name('prompts')


class MaterialSchedule(SamplingModel):
    round_index: int = Field(ge=1)
    context_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    selected_probability: float = Field(ge=0, le=1)
    reason: str = Field(min_length=1, max_length=1000)
    expected_change: str = Field(min_length=1, max_length=1000)
    history_used: str = Field(min_length=1, max_length=1000)


class GroupAllocation(SamplingModel):
    group_id: str
    probability: float = Field(ge=0, le=1)
    reason: str = Field(min_length=1, max_length=700)


class GroupMaterialSchedule(MaterialSchedule):
    allocations: list[GroupAllocation] = Field(min_length=1)


class MaterialExposureSampler(ExposureSampler):
    """Selected IDs are deduplicated; ordinary positions keep a continuous shuffle."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recording_counts = Counter()

    def draw_materials(self, count, selected_ids, selected_probability):
        if len(set(selected_ids)) != len(selected_ids) or not set(selected_ids) <= set(self.train_ids):
            raise ValueError('selected IDs must be unique training recordings')
        if not math.isfinite(selected_probability) or not 0 <= selected_probability <= 1:
            raise ValueError('invalid mixture probability')
        if selected_probability and not selected_ids:
            raise ValueError('empty selection requires zero selected probability')
        ids, channels = [], []
        for _ in range(count):
            selected = selected_probability == 1 or (selected_probability > 0 and self.rng.random() < selected_probability)
            if selected:
                ids.append(self.rng.choice(selected_ids))
                self.exposures += 1
            else:
                ids.extend(super().draw(1))
            channels.append('selected' if selected else 'ordinary')
        self.recording_counts.update(ids)
        return ids, channels

    def state_dict(self):
        return dict(super().state_dict(), recording_counts=dict(self.recording_counts))

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self.recording_counts = Counter(state['recording_counts'])

    def draw_group_materials(self, count, selected_ids, selected_probability, group_weights):
        if count <= 0 or len(set(selected_ids)) != len(selected_ids) or not set(selected_ids) <= set(self.train_ids):
            raise ValueError('positive count and unique training selected IDs required')
        if not math.isfinite(selected_probability) or not 0 <= selected_probability <= 1:
            raise ValueError('invalid within-group selected probability')
        if not selected_ids and selected_probability:
            raise ValueError('empty selection requires zero selected probability')
        if set(group_weights) != set(self.groups):
            raise ValueError('group weights must cover all training groups')
        keys = list(self.groups)
        weights = [float(group_weights[k]) for k in keys]
        if any(not math.isfinite(w) or w < 0 for w in weights) or not math.isclose(sum(weights), 1, abs_tol=1e-6):
            raise ValueError('group probabilities must sum to one')
        membership = {u: k for k, members in self.groups.items() for u in members}
        if set(membership) != set(self.train_ids) or sum(map(len, self.groups.values())) != len(membership):
            raise ValueError('groups must partition the training pool')
        selected_by_group = {k: [] for k in keys}
        for uid in selected_ids:
            selected_by_group[membership[uid]].append(uid)
        ids, channels = [], []
        for _ in range(count):
            group = self.rng.choices(keys, weights=weights, k=1)[0]
            use_selected = bool(selected_by_group[group]) and (selected_probability == 1 or
                (selected_probability > 0 and self.rng.random() < selected_probability))
            pool = selected_by_group[group] if use_selected else self.groups[group]
            ids.append(self.rng.choice(pool))
            channels.append('selected' if use_selected else 'group_pool')
        self.exposures += count
        self.recording_counts.update(ids)
        return ids, channels


def material_evidence(evidence):
    if evidence.get('feedback_role') != 'dev':
        raise ValueError('material curriculum feedback must be dev only')
    result = copy.deepcopy(evidence)
    result['action_explanation'] = '诊断提出练习目标，材料角色选择录音，调度角色决定六组总曝光及各组共用的组内所选概率。'
    result['group_explanation'] = '先按难度和句长六组分配总曝光，再在每组内混合已选材料与该组完整录音池；两者都有放回。难度不是病情。'
    result['phonology']['training_link_explanation'] = '可检索包含目标成分、完整音节或目标句子的具体真实训练录音；文字覆盖不证明实际发音。'
    return result


def compact_history(history):
    result = []
    for episode in list(history)[-4:]:
        decision = episode['decision']
        result.append(dict(round_index=episode['round_index'],
            diagnosis=decision['diagnosis'], selection_summary=decision['selection_summary'],
            schedule=decision['schedule'], reflection=episode['reflection'],
            **({'experience': episode['experience']} if 'experience' in episode else {})))
    return result


def request(config, phase, context, output_type):
    schema = to_strict_json_schema(output_type)
    if phase == 'diagnose':
        schema['$defs']['PracticeFocus']['properties']['evidence_ids']['items']['enum'] = sorted(set(context['evidence']['observation_ids']))
    if phase == 'schedule' and output_type is GroupMaterialSchedule:
        available = sorted(context['context']['selection_summary']['material_counts_by_group'])
        schema['$defs']['GroupAllocation']['properties']['group_id']['enum'] = available
        schema['properties']['allocations'].update(minItems=len(available), maxItems=len(available))
    if phase == 'experience':
        available = sorted({episode['round_index'] for episode in context['completed_rounds']})
        context = dict(context, available_supporting_rounds=available,
                       required_current_round=max(available))
        schema['properties']['supporting_rounds']['items']['enum'] = available
    return dict(model=config.model, reasoning={'effort': config.reasoning_effort}, store=False,
        stream=True, max_output_tokens=config.max_output_tokens,
        input=[{'role': 'system', 'content': (PROMPTS / 'material_sampling_instruction.md').read_text()},
               {'role': 'developer', 'content': (PROMPTS / f'material_sampling_{phase}.md').read_text()},
               {'role': 'user', 'content': json.dumps(context, ensure_ascii=False, sort_keys=True)}],
        text={'format': dict(type='json_schema', name=f'material_{phase}', strict=True,
                             schema=schema)})


def call(config, phase, context, output_type, directory, validate):
    return _call(config, phase=phase, context=context, directory=directory, validate=validate,
                 output_type=output_type, request=request(config, phase, context, output_type),
                 role={'diagnose': 'Diagnosis Agent', 'schedule': 'Scheduling Agent',
                       'reflect': 'Therapist', 'experience': 'Skill Optimizer'}[phase])


def validate_diagnosis(output, evidence):
    if output.producer != 'diagnosis_agent' or output.round_index != evidence['round_index']:
        raise ValueError('diagnosis must be a current-round Agent output')
    if output.source_evidence_sha256 != payload_hash(evidence):
        raise ValueError('diagnosis evidence mismatch')
    if len({f.focus_id for f in output.focuses}) != len(output.focuses):
        raise ValueError('duplicate focus IDs')
    if any(not set(f.evidence_ids) <= set(evidence['observation_ids']) for f in output.focuses):
        raise ValueError('diagnosis cites unknown dev evidence')


def validate_schedule(output, context):
    if output.round_index != context['round_index'] or output.context_sha256 != payload_hash(context):
        raise ValueError('schedule context mismatch')
    if not context['selection_summary']['unique_recordings'] and output.selected_probability != 0:
        raise ValueError('empty selection must use ordinary sampling only')
    if isinstance(output, GroupMaterialSchedule):
        expected = set(context['selection_summary']['material_counts_by_group'])
        if len(output.allocations) != len(expected) or {a.group_id for a in output.allocations} != expected:
            raise ValueError('allocation must cover each training group once')
        if not math.isclose(sum(a.probability for a in output.allocations), 1, abs_tol=1e-6):
            raise ValueError('group probabilities must sum to one')


def choose_material_sampling(config, evidence, *, index, exposures, stage_exposures,
                             groups, directory, history=(), candidate_limit=24, ucg_feedback=None):
    evidence = material_evidence(evidence)
    directory = Path(directory)
    context = dict(evidence=evidence, evidence_sha256=payload_hash(evidence),
                   history=compact_history(history), stage_exposures=stage_exposures)
    diagnosis, sources, seconds = call(config, 'diagnose', context, DiagnosticBrief,
        directory / 'diagnosis', lambda output: validate_diagnosis(output, evidence))
    pack = prepare_candidates(index, diagnosis.model_dump(), exposures=exposures, limit=candidate_limit)
    if ucg_feedback is not None:
        from .ucg_material_context import annotate_ucg_candidates
        pack = annotate_ucg_candidates(pack, index=index, **ucg_feedback)
    write_immutable_json(directory / 'candidates.json', pack)
    selected, provenance, duration = _call(config, phase='select', context={},
        directory=directory / 'selection', output_type=MaterialSelection, role='Material Selection Agent',
        request=build_selector_request(pack), validate=lambda output: validate_selection(index, pack, output.model_dump()))
    sources.update(provenance)
    seconds += duration
    selected = validate_selection(index, pack, selected.model_dump())
    # The original validator describes its offline scope. Only this connected result enables training.
    selected['training_connected'] = True
    write_immutable_json(directory / 'selected_materials.json', selected)
    selection_summary = {key: selected[key] for key in ('unique_recordings', 'patients', 'texts')}
    selection_summary.update(
        difficulty_counts=dict(Counter(r['initial_difficulty'] for r in selected['recordings'])),
        prior_total_exposures=sum(exposures.get(r['recording_id'], 0) for r in selected['recordings']),
        focuses=[dict(focus_id=f['focus_id'], eligible_count=f['eligible_count'],
                      fraction_of_training_pool=f['fraction_of_training_pool']) for f in pack['focuses']])
    membership = {u: key for key, group in groups.items() for u in group['train_ids']}
    selected_counts = Counter(membership[r['recording_id']] for r in selected['recordings'])
    selection_summary['material_counts_by_group'] = {key: selected_counts[key] for key in groups}
    context = dict(round_index=evidence['round_index'], feedback=evidence,
        diagnosis=diagnosis.model_dump(), selection_summary=selection_summary,
        selected_materials=[{k: v for k, v in r.items() if k != 'audio'} for r in selected['recordings']],
        selection_reasons=selected['choices'], stage_exposures=stage_exposures,
        training_pool_recordings=len(index['recordings']), history=compact_history(history))
    schedule, provenance, duration = call(config, 'schedule',
        dict(context=context, context_sha256=payload_hash(context)), GroupMaterialSchedule,
        directory / 'schedule', lambda output: validate_schedule(output, context))
    sources.update(provenance)
    seconds += duration
    weights = {a.group_id: a.probability for a in schedule.allocations}
    decision = dict(diagnosis=diagnosis.model_dump(), selection_summary=selection_summary,
        sampling_mode='group_then_material', group_weights=weights,
        selected_ids=[r['recording_id'] for r in selected['recordings']],
        selected_purposes={r['recording_id']: r['focus_ids'] for r in selected['recordings']},
        schedule=schedule.model_dump())
    write_immutable_json(directory / 'decision.json', decision)
    return dict(group_weights=weights, decision=decision, sources=sources, seconds=seconds,
                selected_ids=decision['selected_ids'], selected_probability=schedule.selected_probability)


def measure_focus(rows, focus, index):
    # Dev sentences need not exist in the train-only retrieval index (UCG).
    # Build ephemeral dev profiles for measurement; never insert them into index.
    profiles = {}
    for target in {r['clean_gt'] for r in rows}:
        mapping = pronunciation(target)
        profiles[target] = dict(target_text=target,
            positions=[asdict(p) for p in mapping.positions],
            components={level: dict(Counter(getattr(p, level) for p in mapping.positions
                                            if p.resolved)) for level in LEVELS})
    related = [r for r in rows if matches(profiles[r['clean_gt']], focus)]
    summary = summarize_sampling_rows(related)
    if focus.level == 'text':
        return dict(cohort=summary, target_occurrences=None, component_differences=None)
    occurrences, differences = 0, Counter()
    for row in related:
        mapping = pronunciation(row['clean_gt'])
        field = 'jyutping' if focus.level == 'syllable' else focus.level
        occurrences += sum(p.resolved and getattr(p, field) == focus.target_unit for p in mapping.positions)
        events, _, _ = transcript_contrasts(row['clean_gt'], row['prediction'])
        if focus.level == 'syllable':
            # A syllable with several component differences still counts only once.
            seen = {(e.reference_index, e.hypothesis_index, e.hypothesis_jyutping)
                    for e in events if e.reference_jyutping == focus.target_unit}
            differences.update(target for _, _, target in seen)
        else:
            differences.update(e.hypothesis_unit for e in events
                               if e.level == focus.level and e.reference_unit == focus.target_unit)
    return dict(cohort=summary, target_occurrences=occurrences, component_differences=dict(differences))


def augment_material_outcome(outcome, before, after, *, feedback_rows, index, channels, observed, groups):
    result = copy.deepcopy(outcome)
    result['evidence'] = material_evidence(result['evidence'])
    brief = DiagnosticBrief.model_validate(result['decision']['diagnosis'])
    previous, current = _join_predictions(feedback_rows, before), _join_predictions(feedback_rows, after)
    focuses = []
    for focus in brief.focuses:
        oid = f"step{result['checkpoint_step']}:focus:{focus.focus_id}"
        result['observation_ids'].append(oid)
        selected = {u for u, tags in result['decision']['selected_purposes'].items() if focus.focus_id in tags}
        focuses.append(dict(focus=focus.model_dump(), observation_id=oid,
            previous=measure_focus(previous, focus, index), current=measure_focus(current, focus, index),
            matching_selected_ids_exposures=sum(u in selected for u in observed)))
    result['focus_outcomes'] = focuses
    result['delivery'].update(channels=dict(Counter(channels)),
        actual_selected_unique=len({u for u, c in zip(observed, channels) if c == 'selected'}))
    membership = {u: key for key, group in groups.items() for u in group['train_ids']}
    result['delivery']['group_counts'] = dict(Counter(membership[u] for u in observed))
    result['delivery']['selected_group_counts'] = dict(Counter(membership[u] for u, c in zip(observed, channels) if c == 'selected'))
    result['delivery']['selected_probability_scope'] = 'within_group; groups without selected recordings use their full group pool'
    result['interpretation'] += ' 目标相关dev子集与读音差异每轮固定追踪，不因退出常见错误列表而消失；多个目标可重叠，曝光不能相加。'
    return result


def reflect_material_sampling(config, outcome, *, directory, history=()):
    if outcome['evidence']['feedback_role'] != 'dev':
        raise ValueError('dev feedback required')
    def validate_reflection(output):
        if output.round_index != outcome['round_index'] or output.outcome_sha256 != payload_hash(outcome):
            raise ValueError('reflection outcome mismatch')
        if not set(output.evidence_ids) <= set(outcome['observation_ids']):
            raise ValueError('reflection cites unknown evidence')
    directory = Path(directory)
    write_immutable_json(directory / 'outcome.json', outcome)
    reflection, sources, seconds = call(config, 'reflect',
        dict(outcome=outcome, outcome_sha256=payload_hash(outcome), history=compact_history(history)),
        SamplingReflection, directory / 'reflection', validate_reflection)
    episode = dict(round_index=outcome['round_index'], decision=outcome['decision'],
                   reflection=reflection.model_dump(), outcome_sha256=payload_hash(outcome))
    episodes = compact_history([*history, episode])
    def validate_experience(output):
        if outcome['round_index'] not in output.supporting_rounds or not set(output.supporting_rounds) <= {e['round_index'] for e in episodes}:
            raise ValueError('experience must cite available rounds including this one')
    experience, provenance, duration = call(config, 'experience', dict(completed_rounds=episodes),
        SamplingExperience, directory / 'experience', validate_experience)
    episode.update(experience=experience.model_dump(), sources={**sources, **provenance}, seconds=seconds+duration)
    write_immutable_json(directory / 'episode.json', episode)
    return episode
