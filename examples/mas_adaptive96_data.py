#!/usr/bin/env python3
"""Fresh MAS proposal/reference decisions from measured generation and DPO feedback.

This entry runs the data stages only. Its synthesized ledger and unmodified raw
preferences are inputs to a separately bound corrected DPO training execution.
"""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from mas_selection_dpo_probe import REPO, read, save
from mas_patient_retention_probe import request_mas
from mas_generation_experience import latest_committed_experience
from therapist_harness import agent, preference_loop as loop
from therapist_harness.store import RunStore, sha256_file
from therapist_harness.text import identity_keys


def prepare(base, root):
    previous = base / 'lexical96_mas_v1'
    old = read(previous / 'protocol.json')
    prior_paths = sorted(base.glob('*/round_*/proposal.json'))
    prior_texts = sorted({s['text'] for p in prior_paths for s in read(p)['stimuli']})
    format_result = read(base / 'ASR_target_format_training_seeds_v1/results.json')
    real_result = read(base / 'real_preference_bridge_v1/results.json')
    measured = dict(initial_R_only_CER_percent=20.79326923076923,
        normalized_synthetic_preference_plus_real_KL_training_seeds={k:v['CER_percent'] for k,v in format_result['arms'].items()},
        normalized_three_seed_mean_CER_percent=format_result['mean_CER_percent'],
        actual_real_DPO_vs_real_plus_MAS_DPO={k:v for k,v in real_result['arms'].items()},
        real_DPO_MAS_incremental_CER_pp=real_result['MAS_incremental_CER_pp'],
        real_DPO_incremental_bootstrap95_pp=real_result['MAS_incremental_bootstrap95_pp'],
        interpretation='No stable overall gain or promoted parent. One normalized seed passed; the three-seed mean exactly tied initial R-only. Direct real DPO also regressed; adding MAS reduced5 errors but remained worse than R-only. A higher-learning-rate matched comparison is still running, so its result is unknown.',
        protection_gap='The dominant feedback patient and hard subgroup often regress while other patients improve. Textual target-position recovery has not established the full intended clinical/phonetic mechanism.')
    # Anonymous patient-group names already used in the experiment summaries;
    # do not include filesystem provenance or identifiers in the Agent payload.
    for values in measured['actual_real_DPO_vs_real_plus_MAS_DPO'].values():
        values['dominant_patient_edit_delta'] = values.pop('vlink247_edit_delta')
        values['dominant_patient_CER_percent'] = values.pop('vlink247_CER_percent')
    experience = latest_committed_experience(previous)
    observations = experience['observations'] if experience else []
    context = dict(measured_DPO_feedback=measured, previous_whole_sentences=prior_texts,
        design_instruction='''Design24 fresh natural Cantonese sentences,4 copies each:8 target,4 contrast,4 protection,8 maintenance.
Use the CURRENT EvidencePack for eligible target IDs, not an invented result ID.
Reconsider the content from the measured downstream DPO feedback. Explain one
concrete change from the previous lexical exercises and what real-speech change
would support it. Do not merely repeat the same rare-word sentence templates.
Keep useful correction opportunities while providing meaningful reverse contrasts
and retaining already-correct alternatives; avoid indiscriminate lexical rewriting.
Use varied ordinary short contexts within the existing length limit. Do not copy
any complete real train/dev sentence or any supplied prior generated sentence.
Written carriers and ASR error counts do not prove acoustic realization or utility.
Reference planning will choose among the same patient-specific recording catalog
and existing recipe options using actual prior observations. No listening claim,
new quality score, CER admission threshold, fabricated negative or forced pairing.
The downstream run will use corrected residual ASR execution and explicitly bound
training-label encoding; this data-stage protocol does not claim a completed DPO
update or fix a successful training configuration. All actual candidates come
from unchanged initial R-only until a trained endpoint is separately accepted.''')
    context_path = base / 'adaptive96_generation_context_v1.json'
    save(context_path, dict(previous_paired_generation_probes=read(base / 'probe_evidence_lexical96_v1.json'),
        measured_DPO_feedback=measured, previous_actual_generation_and_sampling=observations,
        limits='Text evidence only; coarse pass is not word fidelity. Whole-arm DPO outcomes do not identify a single reference causal effect.'))
    design = read(base / 'generation_design_lexical96_v1.json')
    design.update(proposal_context=context, probe_evidence_path=str(context_path), probe_evidence_sha256=sha256_file(context_path))
    design_path = base / 'adaptive96_generation_design_v1.json'; save(design_path, design)
    loop.prepare(root, parent=Path(old['initial_adapter']['path']), parent_predictions=base / 'parent_eval_check/predictions.jsonl',
        review_model=old['preference_loop']['audio_review_model'], review_base_url=old['preference_loop']['audio_review_base_url'],
        rounds_count=1, synthetic_count=96, unique_texts=24, real_count=0, steps=12,
        real_per_step=0, synthetic_per_step=8, seed=42, learning_rate=1e-4,
        review_mode='asr_transcript', comparison='synthetic_only', execution_version='persistent_synthetic_only_v1',
        generation_design_path=design_path, checkpoint_acceptance='real_dev_v1', include_greedy=True,
        review_allow_pending=True, training_allocation='role_budget_v1')
    sources = [Path(__file__), REPO / 'examples/mas_patient_retention_probe.py', REPO / 'examples/mas_generation_experience.py',
        REPO / 'examples/generate_reviewed_sg_recovery_v2.py', REPO / 'examples/generate_reviewed_sg_resumable.py',
        context_path, design_path, previous / 'protocol.json', base / 'ASR_target_format_training_seeds_v1/results.json',
        base / 'real_preference_bridge_v1/results.json', root / 'protocol.json', root / 'sources.json'] + prior_paths
    save(root / 'data_stage_execution.json', dict(source_sha256={str(p):sha256_file(p) for p in sources},
        actual_new_MAS_proposal=True, actual_new_generation_planning=True, actual_new_generation=False,
        source_generation_experience=experience['source_sha256'] if experience else {},
        execution_scope='Proposal, concrete reference/recipe planning, SG coarse review and current-parent sampling only. Stop at synthesized; do not invoke the legacy trainer. Corrected DPO objective/encoding and evaluation will receive a separate execution binding.',
        proposed_audio=96, samples_per_audio=4, independent_greedy=True,
        frozen_parent_and_reference=old['initial_adapter'], parent_promoted=False))
    loop.assess(root, 1)


def verify(root):
    binding = read(root / 'data_stage_execution.json')
    assert all(sha256_file(Path(p)) == h for p,h in binding['source_sha256'].items())
    assert all(sha256_file(Path(p)) == h for p,h in binding['source_generation_experience'].items())
    loop.verify_sources(root)


def propose(root):
    verify(root)
    from openai.types.responses import Response
    from therapist_harness import generation_controller
    number = 1

    def call(protocol, request):
        name = request['text']['format']['name']
        directory = loop.round_dir(root, number) / ('generation_prescribe' if name == 'generation_decision' else 'prescribe')
        api = directory / 'api'; api.mkdir(parents=True, exist_ok=False)
        raw = request_mas(protocol, request, api)
        assert raw['status'] == 'completed'
        final = [x for x in raw['output'] if x.get('type') == 'message' and x.get('phase') == 'final_answer']
        assert len(final) == 1
        text = ''.join(x['text'] for x in final[0]['content'] if x['type'] == 'output_text')
        projected = dict(raw, output=final)
        save(api / 'final_answer_projection.json', dict(raw_response_sha256=sha256_file(api / 'response.json'),
            selected_message_id=final[0]['id'], selected_phase='final_answer', text_sha256=hashlib.sha256(text.encode()).hexdigest(),
            meaning='Core response.json is a final-message-only parsing view. Original complete server response remains in api/response.json.'))
        return Response.model_validate(projected)

    agent.execute_therapist_request = call
    loop.execute_therapist_request = call
    loop.prescribe(root, number)
    config = generation_controller.load_design(RunStore(root).protocol())
    prior = {item for text in config['proposal_context']['previous_whole_sentences'] for item in identity_keys(text).items()}
    proposal = read(loop.round_dir(root, number) / 'proposal.json')
    assert not any(prior.intersection(identity_keys(s['text']).items()) for s in proposal['stimuli'])
    save(root / 'new_proposal_verified.json', dict(status='passed', actual_new_API_calls=2,
        new_texts=len(proposal['stimuli']), assignments=96, no_prior_whole_sentence_repeat=True,
        exact_reference_choices_validated_by_existing_same_patient_compiler=True))
    verify(root)


def generate_sample(root):
    verify(root)
    assert read(root / 'new_proposal_verified.json')['status'] == 'passed'
    original = loop.run_command
    def execute(command, log, *, env_overrides=None):
        command = list(command)
        if command[1] == str(REPO / 'examples/generate_reviewed_sg.py'):
            command[1] = str(REPO / 'examples/generate_reviewed_sg_recovery_v2.py')
        save(Path(log).with_suffix('.command.json'), dict(command=command,
            execution_binding_sha256=sha256_file(root / 'data_stage_execution.json')))
        return original(command, log, env_overrides=env_overrides)
    loop.run_command = execute
    loop.generate(root, 1)
    paired = loop.sample_and_pair(root, 1)
    save(root / 'data_stage_result.json', dict(status='complete', any_preference_pairs=paired,
        journal_phase=RunStore(root).current().phase, actual_DPO_training_executed=False,
        downstream_training='Separately bound corrected residual DPO; legacy train stage was not run.'))
    verify(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'propose', 'generate_sample'])
    parser.add_argument('base', type=Path)
    args = parser.parse_args(); base = args.base.resolve(); root = base / 'adaptive_mas96_data_v1'
    if args.stage == 'prepare': prepare(base, root)
    elif args.stage == 'propose': propose(root)
    else: generate_sample(root)
