#!/usr/bin/env python3
"""MAS reallocates actual exposures on the unchanged fresh44-audio preference set."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random

from pydantic import BaseModel, ConfigDict, Field

from mas_selection_dpo_probe import REPO, read, rows, save
from mas_patient_retention_probe import request_mas
from mas_explicit_feedback_correction import explicit_feedback
from mas_edit_mask_dpo_probe import exposure_plan
from mas_adaptive96_trial_recovery_v2 import train
from mas_evaluation_report import compare
from report_ASR_target_format import summarize
from therapist_harness.schema import RunProtocol
from therapist_harness.store import sha256_file


class Choice(BaseModel):
    model_config = ConfigDict(extra='forbid')
    audio_key: str
    exposures: int = Field(ge=1, le=96)
    reason: str = Field(min_length=1, max_length=180)


class Decision(BaseModel):
    model_config = ConfigDict(extra='forbid')
    reflection: str = Field(min_length=1, max_length=1500)
    hypothesis: str = Field(min_length=1, max_length=1000)
    choices: list[Choice] = Field(min_length=44, max_length=44)


def prepare(base, root):
    from openai.lib._pydantic import to_strict_json_schema

    control = base / 'adaptive96_matched_DPO_recovery_v2'
    old = read(control / 'protocol.json')
    assert all(sha256_file(Path(p)) == h for p, h in old['source_sha256'].items())
    assert read(control / 'results.json')['status'] == 'complete'
    root.mkdir(exist_ok=False)
    original = control / 'MAS_reference/preferences.jsonl'
    data = rows(original); grouped = defaultdict(list)
    for x in data: grouped[x['audio_id']].append(x)
    assert len(data) == 107 and len(grouped) == 44
    source = base / 'adaptive_mas96_data_v1/round_001'
    sidecar = {x['sample_id']: x for x in read(source / 'materialization_sidecar.json')['samples']}
    greedy = {x['audio_id']: x['prediction'] for x in rows(source / 'sampling/greedy.jsonl')}
    feedback_path = control / 'MAS_reference/actual_preference_target_feedback.json'
    family = {x['pair_id']: x['matches'] for x in read(feedback_path)['matches']}
    keys = {}; cards = []
    actual = rows(control / 'MAS_reference/training_receipts/training_exposures.jsonl')
    occurrences = {(x['step'],x['source'],x['occurrence'],x['audio_id']) for x in actual}
    for i, (audio_id, group) in enumerate(sorted(grouped.items())):
        key = f'a{i:03d}'; keys[key] = audio_id
        cards.append(dict(audio_key=key, target=group[0]['chosen'], original_role=sidecar[audio_id]['role'],
            reference_text=group[0]['metadata']['review']['reference_text'], greedy=greedy[audio_id],
            previous_exposures=sum(x[1] == 'synthetic' and x[3] == audio_id for x in occurrences),
            actual_distinct_negatives=[dict(rejected=x['rejected'], family_events=family.get(x['pair_id'], [])) for x in group]))
    save(root / 'audio_keys.json', keys); save(root / 'audio_cards.json', cards)
    sources = [Path(__file__), base / 'adaptive96_opportunity_plan_v1.json', REPO / 'examples/mas_patient_retention_probe.py',
        REPO / 'examples/mas_adaptive96_trial_recovery_v2.py', REPO / 'examples/mas_edit_mask_dpo_probe.py',
        control / 'protocol.json', control / 'results.json', control / 'cross_arm_verified.json',
        control / 'actual_target_opportunity_exposures.json', control / 'target_response_digest.json',
        control / 'MAS_reference/training_receipts/training_exposures.jsonl',
        original, feedback_path, source / 'materialization_sidecar.json', source / 'sampling/greedy.jsonl',
        base / 'lexical_feedback_v1.json', base / 'ASR_target_format_training_seeds_v1/results.json',
        root / 'audio_keys.json', root / 'audio_cards.json']
    hashes = {**old['source_sha256'], **{str(p):sha256_file(p) for p in sources}}
    save(root / 'input_sources.json', hashes)
    p = RunProtocol.model_validate(old['source_protocol'])
    request = dict(model=p.agent.model, reasoning={'effort':p.agent.reasoning_effort}, store=False,
        stream=True, max_output_tokens=12288,
        input=[dict(role='system',content='''Reflect on the fresh MAS Cantonese-ASR DPO result, then allocate actual audio exposures.
This bounded probe changes ONLY the 96 synthetic audio exposure allocations.
Return exactly one choice for EACH of the 44 supplied audio cards, each with at
least one exposure, summing to exactly96. Thus every existing audio and every
one of its distinct actual negative transcripts remains in training. All its
negative pairs are averaged; you cannot select, edit or invent negatives or
change chosen, reference, generation, coarse review, or the training objective.
The remaining52 exposures after covering all44 may be allocated using actual
chosen/rejected errors, ordinary greedy errors and the supplied real feedback.
Do not mistake the original content-purpose label for observed training signal.
Do not assume repeatedly training the single i→yu opportunity will generalize.
Consider useful correction, counterexamples and capability retention together.
This is normalized edit+EOS DPO with the same real96 KL10 retention,focus50%,
12steps,lr1e-4,beta0.1,seed42; 192 real exposures and their order/weights stay
exactly fixed. All44 audio/107 pairs are unchanged. Both the completed control
and new branch start from the same initial R-only, also the frozen reference.
All 511 evaluation recordings are repeated development feedback, not gradients
or a sealed final test. Text-derived phone signatures are not audio truth or
independent clinical mechanisms; the same 鉗→鉛 errors contribute to i→yu and m→n.
Do not add CER admission thresholds, listening claims, quality scores or new
word audits. Return a brief reflection grounded in the actual results, a
prospective hypothesis, and short reasons for the proposed allocations.
The new experiment is not yet evaluated. Earlier15-pair targeted allocation
recipes occasionally improved seed42 but their three-seed average tied R-only;
do not present targeted exposure as an already proven solution.'''),
            dict(role='user',content=json.dumps(dict(
                fresh_results=read(control / 'results.json')['arms'], R_only_CER_percent=20.79326923076923,
                target_response=read(control / 'target_response_digest.json')['arms'],
                actual_target_exposures=read(control / 'actual_target_opportunity_exposures.json')['arms'],
                real_surface_feedback=explicit_feedback(read(base / 'lexical_feedback_v1.json')),
                prior_three_seed_result=read(base / 'ASR_target_format_training_seeds_v1/results.json')['mean_CER_percent'],
                input_scope='Fresh text proposal previously lacked the separate concrete surface-confusion table; supplied now for allocation reflection. No new target text is authorized in this probe.',
                audio_cards=cards),ensure_ascii=False))],
        text={'format':{'type':'json_schema','name':'fresh_actual_opportunity_allocation','strict':True,
                        'schema':to_strict_json_schema(Decision)}})
    save(root / 'request.json', request)
    response = request_mas(p, request, root)
    assert response['status'] == 'completed'
    final = [x for x in response['output'] if x.get('type') == 'message' and x.get('phase') == 'final_answer']
    assert len(final) == 1
    decision = Decision.model_validate_json(''.join(x['text'] for x in final[0]['content'] if x['type'] == 'output_text'))
    assert len({x.audio_key for x in decision.choices}) == 44
    assert {x.audio_key for x in decision.choices} == set(keys)
    assert sum(x.exposures for x in decision.choices) == 96
    save(root / 'mas_decision.json', decision.model_dump(mode='json'))
    sequence = [keys[x.audio_key] for x in decision.choices for _ in range(x.exposures)]
    random.Random(42).shuffle(sequence)
    allocation = dict(steps=[sequence[i:i+8] for i in range(0,96,8)], total_audio_exposures=96,
        rule='Actual MAS exposure counts, all44 audio retained, all107 original negatives unchanged, seed42 shuffle')
    save(root / 'allocation.json', allocation)
    out = root / 'MAS_reference'; out.mkdir()
    scratch = Path('/tmp/qwen3_asr_adaptive96_opportunity_v1'); scratch.mkdir(exist_ok=False)
    a = dict(old['arguments']['MAS_reference'], synthetic_allocation_file=str(root / 'allocation.json'),
             output_dir=str(scratch / 'training'))
    expected = exposure_plan(a, rows(a['real_train_file']), data, allocation, read(a['real_loss_factors'])['factors'])
    assert [x for x in expected if x['source'] == 'real'] == [x for x in actual if x['source'] == 'real']
    assert set(sequence) == set(grouped) and len(sequence) == 96
    save(out / 'expected_exposures.json', expected)
    save(root / 'cpu_smoke.json', dict(status='passed', same_real_exposures_exact=True,
        all44_original_audio_retained=True, all107_original_pairs_and_weights_unchanged=True,
        actual_new_GPU_smoke=False, same_engine_GPU_smoke=old['same_engine_GPU_smoke']))
    for f in [root / n for n in ['input_sources.json','request.json','response.json','mas_decision.json',
                                 'allocation.json','cpu_smoke.json']] + [out / 'expected_exposures.json']:
        hashes[str(f)] = sha256_file(f)
    assert all(sha256_file(Path(f)) == h for f,h in hashes.items())
    save(root / 'protocol.json', dict(source_protocol=old['source_protocol'], source_sha256=hashes,
        arguments={'MAS_reference':a}, same_engine_GPU_smoke=old['same_engine_GPU_smoke'], control=str(control / 'MAS_reference'),
        actual_new_generation=False, actual_new_sampling=False, actual_new_MAS_allocation=True,
        intervention='Only synthetic audio exposure counts/order change; unchanged44audio/107pairs and all real data/weights/order, parent/reference, objective and training settings.'))
    print(dict(status='prepared', actual_MAS_decision=True, audio=len(set(sequence)), exposures=len(sequence)),flush=True)


def report(base, root):
    cfg = read(root / 'protocol.json'); control = Path(cfg['control']); out = root / 'MAS_reference'
    result = summarize(base, out)
    first = read(control / 'training_receipts/training_result.json')['policy_trainable_sha256_before']
    assert first == read(out / 'training_receipts/training_result.json')['policy_trainable_sha256_before']
    assert read(control / 'training_receipts/reference_log_probs.json')['values'] == read(out / 'training_receipts/reference_log_probs.json')['values']
    paired = compare(control / 'evaluation_run/evaluation/predictions.jsonl', out / 'evaluation_run/evaluation/predictions.jsonl')
    save(root / 'paired_allocation.json', paired)
    save(root / 'results.json', dict(status='complete', MAS_allocation=result, parent_promoted=False,
        delta_vs_original_role_allocation_pp=paired['groups']['overall']['delta_pp'],
        bootstrap95_pp=paired['groups']['overall']['patient_cluster_bootstrap_delta_pp_95'],
        same_initial_residual_and_all107_reference_scores=True, actual_DPO_update=True, postprocessing_used=False))
    print(result,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','train','report','all']); parser.add_argument('base',type=Path)
    args = parser.parse_args(); base = args.base.resolve(); root = base / 'adaptive96_opportunity_allocation_v1'
    if args.stage in ['prepare','all']: prepare(base,root)
    if args.stage in ['train','all']: train(base,root,'MAS_reference')
    if args.stage in ['report','all']: report(base,root)
