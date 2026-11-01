#!/usr/bin/env python3
"""Prepare or execute the integrated MAS/reflection loop with explicit real KL."""

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))

from therapist_harness import preference_loop as loop
from therapist_harness.retention_contract import RetentionDesign
from therapist_harness.source_migration import verify_historical_sources
from therapist_harness.store import sha256_file, write_immutable_json


def prepare_inputs(base, root, *, smoke):
    inputs = root.with_name(root.name + '_inputs'); inputs.mkdir(parents=True, exist_ok=False)
    source = base / 'round2_reference_selection_DPO_v1/protocol.json'; previous = loop.read(source)
    archive = base / 'execution_source_archive_before_KL_loop_v1/archive.json'
    migration = verify_historical_sources(previous['source_sha256'], archive)
    write_immutable_json(inputs / 'historical_input_verification.json', dict(migration,
        historical_protocol=str(source), historical_protocol_sha256=sha256_file(source)))
    original = previous['arguments']['MAS_reference']; real = loop.rows(original['real_train_file'])
    if smoke:
        real = real[:4]
        real_path = inputs / 'real_KL.jsonl'; factors_path = inputs / 'real_loss_factors.json'
        loop.jsonl(real_path, real); loop.save(factors_path, dict(factors={x['audio_id']: 1. for x in real}))
    else:
        real_path = Path(original['real_train_file']); factors_path = Path(original['real_loss_factors'])
    design = RetentionDesign(schema_version='real_KL_edit_DPO_v1', real_manifest=str(real_path),
        real_manifest_sha256=sha256_file(real_path), real_loss_factors=str(factors_path),
        real_loss_factors_sha256=sha256_file(factors_path), frozen_reference_asr_sha256=previous['source_protocol']['initial_adapter']['sha256'],
        expected_frozen_base_sha256=original['expected_frozen_base_sha256'], real_per_step=2 if smoke else 16,
        kl_weight=10.)
    write_immutable_json(inputs / 'retention_design.json', design.model_dump(mode='json'))
    catalog = base / 'reference_catalog.json'; experience = base / 'reference_experience_v2/MAS_context.json'
    reference_history = loop.read(experience)
    assert len(reference_history['cards']) == 120 and reference_history['references_with_observations'] == 77
    write_immutable_json(inputs / 'generation_design.json', dict(strategy='mas', catalog_path=str(catalog),
        catalog_sha256=sha256_file(catalog), recipe_options=[dict(mode=mode, instruction_id='cantonese_en') for mode in ['zero_shot', 'instruct2']],
        probe_evidence_path=str(experience), probe_evidence_sha256=sha256_file(experience),
        proposal_context=dict(scope='Prior measured probes are context, not new eligible EvidencePack IDs. This new core run starts from actual initial R-only, not a selected external residual.',
            observations=[
                'New natural target content sometimes produced actual real target-position recovery while worsening global CER. Local target recovery is insufficient for checkpoint acceptance.',
                'Same-patient reference probes changed2of6 target references. At4steps this policy had1367errors versus1377control; at8steps it had1383versus1373. Both policies already used the same newMAScontent.',
                'Fixed4step43/44 training replications gave MAS1380/1374errors. The3training-seed means were1373.667MAS versus1377.667originalreference, with initialR-only1384 and the common selectedparent1373. This is conditional on that parent and data; not stable iteration improvement.',
                'Keep ordinary varied contexts. More sampled errors, harder reference categories and textual phonetic carriers are not evidence of better learning or acoustic realization.',
                'Actual paired audio receives an explicit MAS exposure allocation after current-ASR sampling. All-correct samples are skipped; maintenance additionally has separately fixed real-training KL retention.'],
            reference_experience_scope='255 observed generated audio,77 observed references and43 unobserved; not a per-reference causal performance ranking.')))
    p = previous['source_protocol']
    command = [sys.executable, str(REPO / 'examples/mas_asr_dpo_loop.py'), 'smoke' if smoke else 'prepare', str(root),
        '--retention-design', str(inputs / 'retention_design.json'), '--generation-design', str(inputs / 'generation_design.json'),
        '--parent', p['initial_adapter']['path'], '--parent-predictions', str(base / 'parent_eval_check/predictions.jsonl'),
        '--audio-review-model', p['preference_loop']['audio_review_model'],
        '--audio-review-base-url', p['preference_loop']['audio_review_base_url'],
        '--review-mode', 'asr_transcript', '--review-allow-pending', '--lr', '0.0001',
        '--steps', '4', '--synthetic-per-step', '8', '--seed', '42', '--rounds', '2',
        '--synthetic-count', '96', '--unique-texts', '12']
    write_immutable_json(inputs / 'execution_command.json', dict(command=command, smoke=smoke,
        actual_real_KL_audio=len(real), primary_real_preference_audio=0,
        formal_round_budget=dict(generated_audio=96, unique_texts=12, optimizer_steps=4, synthetic_audio_exposures=32),
        scope='Smoke uses the native CLI tiny6audio/1step mode and builds a next proposal after actual Reflection. Formal preparation freezes96audio/4steps per round; run executes its committed rounds.'))
    return inputs, command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'smoke', 'run', 'all'])
    parser.add_argument('base', type=Path); parser.add_argument('root', type=Path)
    args = parser.parse_args(); base = args.base.resolve(); root = args.root.resolve()
    if args.stage in ['prepare', 'smoke', 'all']:
        inputs, command = prepare_inputs(base, root, smoke=args.stage == 'smoke')
        loop.run_command(command, inputs / ('smoke.log' if args.stage == 'smoke' else 'prepare.log'))
    if args.stage in ['run', 'all']:
        loop.run_command([sys.executable, str(REPO / 'examples/mas_asr_dpo_loop.py'), 'run', str(root)],
            root / 'integrated_run.log')
    print(json.dumps(dict(status='finished_requested_stage', stage=args.stage, root=str(root))), flush=True)


if __name__ == '__main__':
    main()
