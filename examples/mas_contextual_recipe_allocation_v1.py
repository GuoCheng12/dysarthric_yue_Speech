#!/usr/bin/env python3
"""Actual delivery evidence and MAS allocation for the matched new-recipe DPO."""
import argparse
from collections import Counter
from dataclasses import asdict
import html
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from openai.lib._pydantic import to_strict_json_schema
import mas_contextual_recipe_dpo_data_v1 as data
from therapist_harness.evidence import _cer_supported_phone_events
from therapist_harness.phonology import align_characters, analyze_phonological_errors
from therapist_harness.text import cer_normalization_projection
from therapist_harness.preference_data import build_preference_pairs
from therapist_harness.retention_loop import training_pairs
from therapist_harness.dpo_training import preference_key

base, ROOT = data.base, data.ROOT
OLD = Path('/tmp/mas_deterministic_matched_DPO_v1')
ROLES = ['target', 'contrast', 'protection', 'maintenance']


def signatures(target, prediction):
    ref, hyp = map(cer_normalization_projection, [target, prediction])
    alignment = align_characters(ref.normalized, hyp.normalized)
    phone, unresolved, unforced = _cer_supported_phone_events(
        analyze_phonological_errors(ref.surface, hyp.surface), ref, hyp,
        {(e.reference_index, e.hypothesis_index) for e in alignment.forced_substitutions})
    selected = [asdict(e) for e in phone if e.context_sensitive is False and
                e.level == 'coda' and e.operation == 'substitution' and
                (e.reference_unit, e.hypothesis_unit) in [('ng', 'n'), ('n', 'ng')]]
    return dict(edit_count=alignment.edit_distance, events=selected,
        ng_to_n=sum(e['reference_unit'] == 'ng' for e in selected),
        n_to_ng=sum(e['reference_unit'] == 'n' for e in selected),
        unresolved_projection_pairs=unresolved, unforced_pairs=unforced)


def report():
    p = base.verify()
    proposal = base.read(ROOT / 'proposal.json')
    assert proposal['primary_evidence_id'] == 'coda-sub-f19ef0ec14f4'
    arms, receipts, sources, listening = {}, {}, [], []
    for arm in ['MAS', 'fixed']:
        out = ROOT / arm; sampling = out / 'sampling'
        inputs, candidates, greedy, pairs = [base.rows(sampling / f'{n}.jsonl') for n in
            ['validated_inputs', 'candidates', 'greedy', 'preferences']]
        receipt = base.read(sampling / 'sampling_receipt.json')
        assert receipt['status'] == 'complete' and receipt['current_asr_sha256'] == p.initial_adapter.sha256
        for key, filename in [('candidates_sha256', 'candidates.jsonl'), ('validated_inputs_sha256', 'validated_inputs.jsonl'), ('greedy_sha256', 'greedy.jsonl')]:
            assert receipt[key] == base.sha256_file(sampling / filename)
        rebuilt, _ = build_preference_pairs(inputs, candidates, p.initial_adapter.sha256, expected_samples=4, greedy_candidates=greedy)
        assert pairs == rebuilt
        assert all(base.sha256_file(Path(x['audio'])) == x['audio_sha256'] for x in inputs)
        transformed = training_pairs(pairs)
        base.jsonl(out / 'training_preferences.jsonl', transformed)
        metadata = {x['sample_id']: x for x in base.read(out / 'role_sidecar.json')['samples']}
        generation = base.read(out / 'generation/generation_summary.json')
        assert len(generation['outputs']) == 24
        cases = []
        for output in generation['outputs']:
            sid = output['sample_id']; draws = [x for x in candidates if x['audio_id'] == sid]
            actual_pairs = [x for x in transformed if x['audio_id'] == sid]
            assert len(draws) == (4 if output['admitted'] else 0)
            item = dict(sample_id=sid, metadata=metadata[sid], target=output['target_text'],
                audio_sha256=output['output_wav_sha256'], admitted=output['admitted'],
                first_attempt_mode=output['attempt_records'][0]['mode'],
                final_mode=output['generation_mode'], attempts=len(output['attempt_records']),
                review=output['review'], candidates=[dict(prediction=x['prediction'], sample_index=x['sample_index'],
                    seed=x['seed'], ended_with_eos=x['ended_with_eos'], reached_max_new_tokens=x['reached_max_new_tokens'],
                    signatures=signatures(output['target_text'], x['prediction'])) for x in draws],
                pairs=[dict(key=preference_key(x), chosen=x['chosen'], rejected=x['rejected'],
                    signatures=signatures(x['chosen'], x['rejected'])) for x in actual_pairs])
            cases.append(item)
            e = html.escape
            listening.append(f'<article><h3>{arm}: {e(sid)}</h3><p>{e(item["target"])}</p><audio controls src="{e(output["output_wav"])}"></audio><p>Reference：{e(output["reference_text"])}</p><p>粗审：{e(output["review"]["verdict"])} / {e(output["review"]["asr_transcription"])}</p><p>实际候选：{e(" / ".join(x["prediction"] for x in draws))}</p></article>')
        aggregate = {role: dict(requested=sum(x['metadata']['role'] == role for x in cases),
            admitted=sum(x['admitted'] for x in cases if x['metadata']['role'] == role),
            paired_audio=sum(bool(x['pairs']) for x in cases if x['metadata']['role'] == role),
            pairs=sum(len(x['pairs']) for x in cases if x['metadata']['role'] == role),
            target_signature_audio=sum(any(d['signatures']['ng_to_n'] for d in x['candidates']) for x in cases if x['metadata']['role'] == role),
            target_signature_draws=sum(d['signatures']['ng_to_n'] > 0 for x in cases if x['metadata']['role'] == role for d in x['candidates'])) for role in ROLES}
        arms[arm] = dict(cases=cases, aggregate=aggregate, candidate_draws=len(candidates),
            greedy_audio=len(greedy), truncations=sum(x['reached_max_new_tokens'] or not x['ended_with_eos'] for x in candidates),
            generation_timing=generation['timing'], generation_seconds=generation['elapsed_seconds'])
        receipts[arm] = receipt
        sources += [out / 'training_preferences.jsonl', out / 'generation/generation_summary.json']
        sources += list(sampling.glob('*.json*'))
    config_keys = ['current_asr_sha256', 'base_model_sha256', 'samples_per_audio', 'seed', 'temperature', 'top_p', 'top_k', 'max_new_tokens', 'language', 'prompt']
    assert all(receipts['MAS'][k] == receipts['fixed'][k] for k in config_keys)
    manifests = {a: base.rows(ROOT / a / 'manifest.jsonl') for a in arms}
    for m, f in zip(manifests['MAS'], manifests['fixed']):
        assert all(m[k] == f[k] for k in ['sample_id', 'target_text', 'seed'])
    maps = {a: {x['sample_id']: x for x in arms[a]['cases']} for a in arms}
    for sid, m in maps['MAS'].items():
        f = maps['fixed'][sid]
        assert m['metadata']['speaker_id'] == f['metadata']['speaker_id']
        if m['admitted'] and f['admitted']:
            assert [x['seed'] for x in m['candidates']] == [x['seed'] for x in f['candidates']]
    sources += [Path(__file__), ROOT / 'proposal.json', ROOT / 'plan.json']
    result = dict(arms=arms, paired_identity_and_sampling_verified=True,
        source_sha256={str(f): base.sha256_file(f) for f in sources},
        scope='Text-derived conservative coda signatures are post-sampling diagnostics, not phoneme/audio admission or proof of fidelity. Each arm retains its own actual wrong pairs; no paired-ID intersection filter. No utility claim before real DPO evaluation.')
    base.save(ROOT / 'delivery_report.json', result)
    base.write_immutable_bytes(ROOT / 'listening.html', ('<!doctype html><meta charset="utf-8"><title>MAS recipe24</title><style>body{font:16px system-ui;max-width:1200px;margin:32px auto}main{display:grid;grid-template-columns:1fr 1fr;gap:12px}article{border:1px solid #ccc;padding:12px}</style><h1>相同新内容和患者：MAS配方 vs 固定配方</h1><p>三文本粗审；未证明逐字保真。</p><main>'+''.join(listening)+'</main>').encode())
    print(json.dumps({a: arms[a]['aggregate'] for a in arms}, ensure_ascii=False), flush=True)


class AudioVisit(BaseModel):
    model_config = ConfigDict(extra='forbid')
    audio_id: str
    visits: int = Field(ge=1, le=8)
    reason_zh: str


class ArmAllocation(BaseModel):
    model_config = ConfigDict(extra='forbid')
    arm: Literal['MAS', 'fixed']
    allocations: list[AudioVisit]


class Decision(BaseModel):
    model_config = ConfigDict(extra='forbid')
    action: Literal['train_matched_DPO', 'close_missing_target_delivery']
    reflection_zh: str
    allocations: list[ArmAllocation]
    limitations_zh: str


def allocate():
    p = base.verify(); report = base.read(ROOT / 'delivery_report.json')
    assert all(base.sha256_file(Path(f)) == h for f, h in report['source_sha256'].items())
    out = ROOT / 'MAS_allocation'; out.mkdir(exist_ok=False)
    context = dict(proposal=base.read(ROOT / 'proposal.json'), actual_delivery=report['arms'],
        evidence_correction='The immutable proposal rationale reverses voice005 glyph direction in one sentence. Actual frozen evidence is 框->館 (ng->n), not 館->框. Patient membership and selected family are valid; do not copy the reversed prose into interpretation.',
        instructions='Decide whether actual target-relevant errors support this bounded matched DPO. Prefer running the proposed experiment when relevant actual pairs exist; no new hyperparameter/probe chain. For train: exactly two allocations, MAS and fixed, each32audio visits. Use only audio with actual pairs in that arm; do not restrict to common pairedIDs. Preserve prospective role totals target18/contrast6/protection4/maintenance4 if eachrole haspairs in botharms. If arole lacks pairs, use the same explicitly explained revised rolebudget for botharms, since no negatives can be fabricated. Eachaudio max8visits. Select and allocate from actualtarget-error context plus contrast/protection/maintenance; do not reward totalCER or pairvolume. Allactualdistinctwrongpairs forselectedaudio are averaged, none replaced with inventedtranscripts. Same32realDPOaudio/91pairs and256KLvisits, sameinitial8cparent/frozenreference, foursteps andoriginalhyperparameters. MASdesigncomparisonincludesrecipeandresultingactualallocation; notpurephysicalreference-only effect. Missing exact phone signature is diagnostic, not SGfailure. If closing, return no allocations and state the actual unmet relevance. No new SG review threshold or fidelity claim.')
    request = dict(model=p.agent.model, reasoning={'effort': 'high'}, store=False, stream=True, max_output_tokens=8000,
        input=[dict(role='system', content='你是本轮实际MAS训练分配Agent，根据已经生成和当前ASR真实采样的偏好对，给出具体可执行的匹配预算训练分配。保留简单SG粗审，不把患者不清晰当失败。'),
            dict(role='user', content=json.dumps(context, ensure_ascii=False))],
        text={'format': dict(type='json_schema', name='actual_recipe_DPO_allocation', strict=True, schema=to_strict_json_schema(Decision))})
    base.save(out / 'request.json', request)
    response = base.execute_therapist_request(p, request); base.save(out / 'response.json', base.serialize_response(response))
    assert response.status == 'completed'
    decision = Decision.model_validate_json(base.terminal_output_text(response)); base.save(out / 'decision.json', decision.model_dump())
    if decision.action != 'train_matched_DPO':
        assert not decision.allocations
        print(decision.model_dump_json(indent=2), flush=True); return
    assert Counter(x.arm for x in decision.allocations) == {'MAS': 1, 'fixed': 1}
    original = base.read(OLD / 'plan/plan.json')
    real = base.read(OLD / 'plan/expected_DPO_exposures.json')['R_DPO_KL']
    kl = base.read(OLD / 'plan/expected_common_KL_exposures.json')
    planned, selections, role_budgets = {}, {}, {}
    for allocation in decision.allocations:
        arm = allocation.arm; cases = {x['sample_id']: x for x in report['arms'][arm]['cases']}
        ids = [x.audio_id for x in allocation.allocations]
        assert len(ids) == len(set(ids)) and sum(x.visits for x in allocation.allocations) == 32
        assert all(cases[x]['pairs'] for x in ids)
        role_budgets[arm] = Counter()
        for x in allocation.allocations: role_budgets[arm][cases[x.audio_id]['metadata']['role']] += x.visits
        # Round robin over immutable manifest order; each visit averages all actual wrong pairs for that audio.
        counts = {x.audio_id: x.visits for x in allocation.allocations}; order = []
        while any(counts.values()):
            for sid in cases:
                if counts.get(sid, 0): order.append(sid); counts[sid] -= 1
        selections[arm] = order; planned[arm] = []
        for step in range(1, 5):
            synthetic = [dict(audio_id=sid, source='synthetic', objective='dpo', step=step,
                occurrence=(step-1)*8+i, preference_key=pair['key'], loss_weight=.25/8/len(cases[sid]['pairs']))
                for i, sid in enumerate(order[(step-1)*8:step*8]) for pair in cases[sid]['pairs']]
            assert abs(sum(x['loss_weight'] for x in synthetic)-.25) < 1e-12
            planned[arm].append(real[step-1] + synthetic)
    assert role_budgets['MAS'] == role_budgets['fixed']
    if all(report['arms'][a]['aggregate'][r]['paired_audio'] for a in ['MAS', 'fixed'] for r in ROLES):
        assert role_budgets['MAS'] == dict(target=18, contrast=6, protection=4, maintenance=4)
    files = [Path(__file__), ROOT / 'delivery_report.json', out / 'decision.json', ROOT / 'plan.json',
        OLD / 'plan/plan.json', OLD / 'plan/expected_DPO_exposures.json', OLD / 'plan/expected_common_KL_exposures.json']
    hashes = {**report['source_sha256'], **base.read(ROOT / 'sources.json'), **{str(f): base.sha256_file(f) for f in files}}
    # Reuse only the relevant old real artifacts; historical synthetic rows are not the new data.
    real_receipt = original['source_sampling_receipts']['historical_real']
    for f, h in original['source_sha256'].items():
        if '/real96_dpo_probe/sampling/' in f or f == original['data_paths']['real_KL']:
            assert base.sha256_file(Path(f)) == h; hashes[f] = h
    assert base.sha256_file(Path(real_receipt['path'])) == real_receipt['sha256']
    source_receipts = {'real': real_receipt}
    for arm in ['MAS', 'fixed']:
        f = ROOT / arm / 'sampling/sampling_receipt.json'
        receipt = base.read(f)
        assert receipt['current_asr_sha256'] == original['parent']
        for key in ['current_asr_sha256','base_model_sha256','language','prompt','samples_per_audio','seed','temperature','top_p','top_k']:
            assert receipt[key] == real_receipt['decode'][key]
        source_receipts[arm] = dict(path=str(f), sha256=base.sha256_file(f), decode=receipt)
    plan = ROOT / 'training_plan'; plan.mkdir(exist_ok=False)
    base.save(plan / 'plan.json', dict(parent=original['parent'], source_sha256=hashes,
        data_paths=dict(real_KL=original['data_paths']['real_KL'], real_preferences=original['data_paths']['real_preferences'],
            synthetic_preferences={a: str(ROOT / a / 'training_preferences.jsonl') for a in ['MAS','fixed']}),
        source_sampling_receipts=source_receipts, selections=selections, role_budgets=role_budgets,
        settings=original['settings'], gate=original['gate'],
        decoding_scope='Same parent, language/prompt/sampling distribution. Existing real cap256 versus SG cap128 retained and reported separately; no fabricated common receipt.'))
    base.save(plan / 'expected_DPO_exposures.json', planned)
    base.save(plan / 'expected_common_KL_exposures.json', kl)
    base.verify(); print(decision.model_dump_json(indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('stage', choices=['report','allocate'])
    args = parser.parse_args(); globals()[args.stage]()
