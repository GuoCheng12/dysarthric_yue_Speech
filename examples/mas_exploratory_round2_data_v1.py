#!/usr/bin/env python3
"""Adaptive second-round data from an actual learned current ASR endpoint."""

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from mas_selection_dpo_probe import REPO, read, rows, save, jsonl, execute
from mas_patient_retention_probe import request_mas
from therapist_harness.evidence import extract_evidence
from therapist_harness.capabilities import extract_capability_anchors
from therapist_harness.generation_controller import load_catalog, public_cards
from therapist_harness.schema import RunProtocol
from therapist_harness.store import sha256_file, sha256_directory
from therapist_harness.text import text_normalize


class Action(BaseModel):
    model_config = ConfigDict(extra='forbid')
    sample_id: str
    target_text: str = Field(min_length=6, max_length=40)
    reference_id: str
    mode: Literal['zero_shot','instruct2']
    instruction_id: Literal['cantonese_en','cantonese_zh']
    evidence_ids: list[str] = Field(min_length=1, max_length=3)
    reason: str = Field(min_length=1, max_length=500)


class AdaptiveProposal(BaseModel):
    model_config = ConfigDict(extra='forbid')
    reflection: str = Field(min_length=1, max_length=2000)
    primary_evidence_id: str
    hypothesis: str = Field(min_length=1, max_length=1000)
    assignments: list[Action] = Field(min_length=15, max_length=15)
    expected_observation: str = Field(min_length=1, max_length=1000)
    observation_against_hypothesis: str = Field(min_length=1, max_length=1000)


def verify(root):
    cfg = read(root / 'execution.json')
    assert all(sha256_file(Path(f)) == h for f,h in cfg['source_sha256'].items())
    return cfg


def prepare(base, root):
    parent_root = root.parent; plan = read(parent_root / 'plan.json')
    assert all(sha256_file(Path(f)) == h for f,h in plan['source_sha256'].items())
    p = RunProtocol.model_validate(plan['source_protocol'])
    assert sha256_directory(Path(plan['current_parent'])) == plan['current_parent_sha256']
    current_path = base / 'ASR_target_format_dpo_probe_v1/edit_eos/evaluation_run/evaluation/predictions.jsonl'
    initial_path = Path(p.initial_adapter_dev_predictions.path); zero_path = Path(p.zero_shot_dev_predictions.path)
    current = rows(current_path); initial = rows(initial_path); old = {x['utt_id']:x for x in initial}
    extraction = extract_evidence(current, initial, rows(zero_path), current_prediction_column='prediction',
        m0_prediction_column='prediction', zero_shot_prediction_column=p.zero_shot_prediction_column,
        previous_rows=initial, previous_prediction_column='prediction', max_clusters_per_level=6)
    assert extraction.current_metrics.overall.total_edits == 1373
    assert extraction.m0_metrics.overall.total_edits == 1384
    capabilities = extract_capability_anchors(current, initial, current_prediction_column='prediction',
        m0_prediction_column='prediction', previous_rows=initial, previous_prediction_column='prediction')
    catalog_path = base / 'reference_catalog.json'
    cards = load_catalog(dict(catalog_path=str(catalog_path),catalog_sha256=sha256_file(catalog_path)), rows(p.real_train.path))
    voices = {x['speaker_id']:x['voice_group'] for x in cards.values()}
    fixed_inputs = rows(parent_root / 'fixed_curriculum_inputs.jsonl')
    sidecar_path = base / 'lexical96_mas_v1/round_001/materialization_sidecar.json'
    sidecar = {x['sample_id']:x for x in read(sidecar_path)['samples']}
    slots = [dict(sample_id=x['audio_id'], role=sidecar[x['audio_id']]['role'],
        previous_target=x['clean_gt'], seed=x['generation_seed']) for x in fixed_inputs]
    assert Counter(x['role'] for x in slots) == dict(target=6,contrast=2,protection=3,maintenance=4)
    fixed_sampling = parent_root / 'fixed_curriculum_sampling'
    receipt = read(fixed_sampling / 'sampling_receipt.json')
    assert receipt['status'] == 'complete' and receipt['current_asr_sha256'] == plan['current_parent_sha256']
    audit = read(fixed_sampling / 'preference_audit.json')
    greedy = {x['audio_id']:x for x in rows(fixed_sampling / 'greedy.jsonl')}
    for x in slots:
        x['current_greedy_on_previous_audio'] = greedy[x['sample_id']]['prediction']
        x['current_sampling'] = next(a for a in audit['audio_results'] if a['audio_id'] == x['sample_id'])
    patient_cards = []
    for patient, voice in sorted(voices.items()):
        subset = [x for x in current if x['speaker_id'] == patient]
        patient_cards.append(dict(voice_group=voice, feedback_audio=len(subset),
            characters=sum(x['normalized_reference_characters'] for x in subset),
            initial_errors=sum(old[x['utt_id']]['edit_count'] for x in subset),
            current_errors=sum(x['edit_count'] for x in subset)))
    concrete = []; counts = Counter(); local_ids = {}
    ordered = sorted(current, key=lambda x: (-(x['edit_count']-old[x['utt_id']]['edit_count']),-x['edit_count'],x['utt_id']))
    for x in ordered:
        if not x['edit_count'] or counts[x['speaker_id']] >= 2: continue
        key = f'feedback.{len(concrete)+1:02d}'; local_ids[key] = x['utt_id']; counts[x['speaker_id']] += 1
        concrete.append(dict(evidence_id=key, voice_group=voices[x['speaker_id']], target=x['clean_gt'],
            initial_prediction=old[x['utt_id']]['prediction'], current_prediction=x['prediction'],
            initial_errors=old[x['utt_id']]['edit_count'], current_errors=x['edit_count']))
        if len(concrete) == 30: break
    root.mkdir(exist_ok=False)
    save(root / 'feedback_key_map.json', local_ids); save(root / 'slots.json', slots)
    save(root / 'role_policy.json', dict(requested_audio_by_role=dict(target=6,contrast=2,protection=3,maintenance=4),
        requested_training_exposures=dict(target=32,contrast=16,protection=16,maintenance=32),
        total_audio_exposures=96, max_audio_per_selected_patient=4,
        matching='Same15logical role slots and current-ASR sampling seeds. Train common slots with actual pairs on BOTH sides; no fabricated negative. Use identical role/case occurrence schedule and real order. If a role has no common paired slot, renormalize requested exposure shares over available roles by largest remainder, identically for both arms.',
        scope='Role exposure shares are explicit optimizer budgets, not fake generated-copy counts. This whole-policy comparison may change content/patient/reference/recipe together.'))
    evidence = dict(schema_version='external_round2_evidence_components_v1', round_index=2,
        current_checkpoint_sha256=plan['current_parent_sha256'], initial_R_only_sha256=p.initial_adapter.sha256,
        extraction=asdict(extraction), capability_anchors=list(capabilities), patient_cards=patient_cards,
        concrete_current_errors=concrete,
        scope='Reuse the existing EvidencePack extraction/comparison/capability components on actual paired511snapshots. This is an explicit external adapter, not a fabricated core EpisodeMemory/ReflectionPack or retroactive ledger commit.')
    evidence = json.loads(json.dumps(evidence, default=lambda x:x.model_dump(mode='json'), ensure_ascii=False))
    save(root / 'evidence_components.json', evidence)
    previous_texts = sorted({x['clean_gt'] for x in fixed_inputs} | {x['target_text'] for x in read(base / 'patient_content_probe_v1/cases.json')})
    prior = read(base / 'patient_content_feedback_export_v2/proposal_context.json')
    context = dict(evidence=evidence, slots=slots, reference_cards=public_cards(cards),
        role_policy=read(root / 'role_policy.json'), previous_whole_sentences=previous_texts,
        prior_failed_content_probe=dict(content_proposal=prior['source_content_proposal'],
            actual_targets=prior['declared_content_target_observations'], comparison=prior['matched_comparison']),
        parent_choice_and_limits=dict(current_CER_percent=20.628004807692307,initial_R_only_CER_percent=20.79326923076923,
            prior_three_training_seed_mean_equals_R_only=True, explanation=plan['parent_choice']))
    body = json.dumps(context,ensure_ascii=False); assert '/data/' not in body and 'vlink' not in body
    save(root / 'request_context.json', context)
    files = [Path(__file__), parent_root / 'plan.json', current_path, initial_path, zero_path, catalog_path, sidecar_path,
        parent_root / 'fixed_curriculum_inputs.jsonl', fixed_sampling / 'sampling_receipt.json',
        fixed_sampling / 'preference_audit.json', fixed_sampling / 'greedy.jsonl',
        base / 'patient_content_feedback_export_v2/proposal_context.json']
    files += list(root.glob('*.json'))
    files += [REPO / x for x in ['src/therapist_harness/evidence.py','src/therapist_harness/capabilities.py',
        'src/therapist_harness/generation_controller.py','examples/mas_patient_retention_probe.py',
        'examples/generate_reviewed_sg_recovery_v2.py','examples/generate_reviewed_sg_resumable.py',
        'examples/sample_residual_asr_preferences_v1.py','examples/transcribe_sg_review.py']]
    save(root / 'execution.json', dict(plan=plan, source_sha256={**plan['source_sha256'],**{str(f):sha256_file(f) for f in files}},
        actual_new_generation=False, actual_new_sampling=False, original_ledgers_unchanged=True))
    print(dict(status='prepared', current_errors=1373, source_errors=1384, clusters=len(extraction.error_clusters)), flush=True)


def prescribe(base, root):
    from openai.lib._pydantic import to_strict_json_schema

    cfg = verify(root); p = RunProtocol.model_validate(cfg['plan']['source_protocol'])
    out = root / 'MAS_proposal'; out.mkdir()
    context = read(root / 'request_context.json')
    request = dict(model=p.agent.model, reasoning={'effort':p.agent.reasoning_effort},store=False,stream=True,
        max_output_tokens=18000, input=[dict(role='system', content='''You are the MAS in an actual second-round Cantonese patient-ASR experiment.
The current ASR is a learned residual endpoint:1373 real-dev errors versus1384 for
initial R-only. It passed the original seed42 single-checkpoint gate, but prior
three training-seed mean merely tied initialR-only; no stable gain is claimed.
Both next arms START FROM THIS SAME CURRENT1373-ENDPOINT, not initialR-only.
The DPO AND real-KL teacher remain frozen INITIAL R-only. Candidate sampling uses
current learned ASR. A real warm-start smoke verified this distinction.
Choose15new natural Cantonese texts and concrete train-only references/recipes,
one for each supplied exact role slot. Preserve roles6target/2contrast/3protection/
4maintenance and TTS seeds. Choose one current targetable and tts_actionable
error-cluster as primary_evidence_id. Cite current cluster/capability IDs or
feedback.NN for each assignment; target cases must include the primary ID.
Prefer ordinary varied8-18character sentences, under24surfaceChinesecharacters,
ending with punctuation. Do not repeat supplied full real-feedback or previous
generated sentences. Contrasts make competing forms correct in other meaningful
contexts; protection and maintenance should retain existing gains and abilities.
Use actual CURRENT versus initial/previous errors, particularly new regressions;
do not blindly copy the previous lexical targets. The last patient-specific
content probe reduced30overall errors versus its control yet failed to correct
the named target words and remained worse than R-only. Do not call that target
transfer. More ASR errors or textual carriers are not proof of training value.
Select any listed reference_id. At most4 of the15new audio may use one voice_group:
the previous all-one-patient allocation regressed overall. Within this bounded
budget, explain why the selected patient's measured errors/coverage and reference
facts support the choice. Unknown acoustic traits remain unknown; do not infer
clinical severity or slurring from IDs/text. Use explicit Cantonese instructions
cantonese_en or cantonese_zh, zero_shot or instruct2. This tests the complete
adaptive data policy against continued fixed curriculum; no single-reference
causal claim. Both arms have15audio and4actual stochastic candidates plus greedy
per audio. The fixed side reuses actual old audio but refreshes current-ASR
candidates. Common actual paired role slots receive identical96synthetic and
192realKL occurrences; role shares32/16/16/32 are renormalized only for unavailable
roles. All distinct actual wrong transcripts average within audio. All-correct
skips, one repeated error forms one pair. Never invent a wrong transcript.
Keep simple three-text coarse review/rulebook: obvious leakage, severe truncation,
off-topic collapse or mechanical looping only. No CER cutoff, phoneme/word audit,
listening claim or assertion that a coarse pass proves target fidelity.
Same12step edit+EOS DPO,realKL10,lr1e-4,beta.1,seed42; freshAdamW continues the
learned residual weights. Do not change loss, teacher, optimizer or evaluation.
Return a concise Chinese reflection, one primary cluster, concrete15assignments,
a hypothesis and prospective observation for AND against it. Existing aggregate
clinical labels are not acoustic measurements. Both current1373 and initial1384
real-error baselines plus original pooled/macro/hard gates will be reported.'''),
            dict(role='user',content=json.dumps(context,ensure_ascii=False))],
        text={'format':dict(type='json_schema',name='adaptive_round2_proposal',strict=True,
            schema=to_strict_json_schema(AdaptiveProposal))})
    save(out / 'request.json', request); response = request_mas(p,request,out)
    assert response['status'] == 'completed'
    final = [x for x in response['output'] if x.get('type')=='message' and x.get('phase')=='final_answer']; assert len(final)==1
    proposal = AdaptiveProposal.model_validate_json(''.join(x['text'] for x in final[0]['content'] if x['type']=='output_text'))
    slots = {x['sample_id']:x for x in read(root / 'slots.json')}
    actions = {x.sample_id:x for x in proposal.assignments}; assert len(actions)==15 and actions.keys()==slots.keys()
    clusters = {x['evidence_id']:x for x in context['evidence']['extraction']['error_clusters']}
    primary = clusters[proposal.primary_evidence_id]; assert primary['targetable'] and primary['tts_actionable']
    known = set(clusters) | {x['evidence_id'] for x in context['evidence']['capability_anchors']} | {x['evidence_id'] for x in context['evidence']['concrete_current_errors']}
    cards = {x['reference_id']:x for x in read(base / 'reference_catalog.json')['cards']}
    counts = Counter(cards[x.reference_id]['voice_group'] for x in actions.values()); assert max(counts.values())<=4
    forbidden = {text_normalize(x['clean_gt']) for path in [p.real_train.path,p.feedback_split.path] for x in rows(path)}
    forbidden |= {text_normalize(x) for x in context['previous_whole_sentences']}
    texts = [text_normalize(x.target_text) for x in actions.values()]
    assert len(set(texts))==15 and not (set(texts)&forbidden)
    assignments=[]; choices=[]; (root/'audio').mkdir()
    for key,slot in slots.items():
        action=actions[key]; ref=cards[action.reference_id]
        assert set(action.evidence_ids)<=known
        if slot['role']=='target':assert proposal.primary_evidence_id in action.evidence_ids
        assert action.target_text[-1] in '。！？!?'
        assignments.append(dict(sample_id=key,target_text=action.target_text,reference_text=ref['reference_text'],
            reference_wav=ref['reference_wav'],reference_wav_sha256=ref['reference_wav_sha256'],
            output_wav=str(root/'audio'/(key+'.wav')),seed=slot['seed']))
        choices.append(dict(sample_id=key,reference_id=action.reference_id,mode=action.mode,
            instruction_id=action.instruction_id,reason=action.reason,reference_wav_sha256=ref['reference_wav_sha256'],
            target_text_sha256=hashlib.sha256(action.target_text.encode()).hexdigest()))
    save(out/'proposal.json',proposal.model_dump(mode='json')); jsonl(root/'manifest.jsonl',assignments)
    save(root/'generation_plan.json',dict(schema_version='generation_plan_v1',choices=choices,
        hypothesis=proposal.hypothesis,expected_observation=proposal.expected_observation))
    save(root/'target_descriptor.json',{k:primary[k] for k in ['evidence_id','operation','level','reference_unit','hypothesis_unit','left_context','right_context']})
    verify(root)
    save(root/'proposal_receipt.json',dict(actual_MAS_call=True,response_id=response['id'],
        bound_sha256={str(f):sha256_file(f) for f in [out/'request.json',out/'response.json',out/'proposal.json',root/'manifest.jsonl',root/'generation_plan.json',root/'target_descriptor.json']},
        patients=dict(counts),original_ledgers_unchanged=True))
    print(dict(status='prescribed',primary=proposal.primary_evidence_id,patients=dict(counts)),flush=True)


def run(root):
    cfg=verify(root); plan=cfg['plan']; p=RunProtocol.model_validate(plan['source_protocol']); c=p.preference_loop
    assert all(sha256_file(Path(f))==h for f,h in read(root/'proposal_receipt.json')['bound_sha256'].items())
    command=[sys.executable,str(REPO/'examples/generate_reviewed_sg_recovery_v2.py'),'--manifest',str(root/'manifest.jsonl'),
        '--out-dir',str(root/'generation'),'--generation-plan',str(root/'generation_plan.json'),
        '--review-mode','asr_transcript','--review-asr-worker','--audio-review-model',c.audio_review_model,
        '--audio-review-base-url',c.audio_review_base_url,'--review-asr-model',p.base_model_dir,
        '--review-asr-adapter',c.review_asr_adapter,'--review-asr-adapter-sha256',c.review_asr_adapter_sha256,
        '--review-asr-base-model-sha256',c.review_asr_base_model_sha256,'--review-allow-pending']
    execute(root,command,'generation'); verify(root)
    execute(root,[sys.executable,str(REPO/'examples/sample_residual_asr_preferences_v1.py'),'sample',
        '--inputs',str(root/'generation/accepted.jsonl'),'--real-train',p.real_train.path,'--feedback',p.feedback_split.path,
        '--out-dir',str(root/'sampling'),'--base-model',p.base_model_dir,'--current-adapter',plan['current_parent'],
        '--initial-r-only-adapter',p.initial_adapter.path,'--expected-parent-sha256',plan['current_parent_sha256'],
        '--samples','4','--include-greedy','--seed','42','--temperature','.8','--top-p','.95','--max-new-tokens','128'],'sampling')
    verify(root); save(root/'data_complete.json',dict(status='complete',audit=read(root/'sampling/preference_audit.json')))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','prescribe','run','all']);parser.add_argument('base',type=Path)
    args=parser.parse_args();base=args.base.resolve();root=base/'exploratory_round2_v1/adaptive_data'
    if args.stage in ['prepare','all']:prepare(base,root)
    if args.stage in ['prescribe','all']:prescribe(base,root)
    if args.stage in ['run','all']:run(root)
