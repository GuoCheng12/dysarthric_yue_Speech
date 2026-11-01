#!/usr/bin/env python3
"""Paired reference/instruction probe using the existing isolated SG review worker."""

import argparse
from collections import Counter, defaultdict
import hashlib
import html
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]
from therapist_harness.store import sha256_file, write_immutable_json, write_immutable_bytes
from therapist_harness.generation_design import load_generation_choices

BASE = Path('/data/qwen3-asr/records/therapist_mas_setting_d_v1')
OLD = BASE / 'mas_tts_only_dpo_seed42_20260908_retry_v2'
REAL = Path('/data/qwen3-asr/finetune/data_setting_d_v1/setting_d_train.jsonl')
CATALOG = Path('/data/qwen3-asr/records/cosyvoice3_reference_sft_setting_d_v1/therapist_reference_manifest_v1.json')


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def jsonl(path, values):
    write_immutable_bytes(path, ''.join(json.dumps(x, ensure_ascii=False)+'\n' for x in values).encode())


def prepare(root, cases_count):
    import soundfile as sf
    root.mkdir(parents=True, exist_ok=False)
    pool = rows(REAL)
    by_speaker = defaultdict(list)
    for row in pool:
        if float(row['duration']) <= 30:
            by_speaker[row['speaker_id']].append(row)
    catalog = read(CATALOG)
    ref_by_hash = {r['reference_wav_sha256']: r for r in catalog['references']}
    old = rows(OLD / 'round_001/synthesis_manifest.jsonl')
    sidecar = {r['sample_id']: r for r in read(OLD / 'round_001/materialization_sidecar.json')['samples']}
    stimuli = read(OLD / 'round_001/proposal.json')['stimuli'][:cases_count]
    cases, cards, assignments, choices = [], {}, [], []
    for index, stimulus in enumerate(stimuli):
        group = ['medium', 'hard', 'easy'][index % 3]
        original = next(x for x in old if sidecar[x['sample_id']]['stimulus_id'] == stimulus['stimulus_id']
                        and sidecar[x['sample_id']]['reference_difficulty_group'] == group)
        ref = ref_by_hash[original['reference_wav_sha256']]
        eligible = sorted([x for x in by_speaker[ref['speaker_id']]
                           if x['audio'] != ref['reference_wav']], key=lambda x:x['utt_id'])
        if not eligible:
            raise ValueError('same-patient alternative reference unavailable')
        alternative = random.Random(42 + index).choice(eligible)
        references = [ref, dict(speaker_id=ref['speaker_id'], difficulty_group=group,
                               reference_text=alternative['clean_gt'], reference_wav=alternative['audio'],
                               reference_wav_sha256=sha256_file(Path(alternative['audio'])))]
        case_id = f'case{index:02d}'
        cases.append(dict(case_id=case_id, stimulus_id=stimulus['stimulus_id'], role=stimulus['role'],
                          target_text=stimulus['text'], speaker_id=ref['speaker_id'], group=group,
                          old_sample_id=original['sample_id'], seed=original['seed']))
        for ref_index, reference in enumerate(references):
            rid = f'{case_id}.ref{ref_index}'
            info = sf.info(reference['reference_wav'])
            assert sha256_file(Path(reference['reference_wav'])) == reference['reference_wav_sha256']
            cards[rid] = dict(reference, reference_id=rid, seconds=info.duration,
                             sample_rate=info.samplerate, channel_count=info.channels,
                             acoustic_traits='unknown: no direct listening annotation',
                             selection='existing reference' if ref_index == 0 else 'fixed-seed alternative, no CER ranking')
            for instruction in ['default', 'cantonese_en']:
                sid = f'{case_id}.ref{ref_index}.{instruction}'
                assignments.append(dict(sample_id=sid, target_text=stimulus['text'],
                    reference_text=reference['reference_text'], reference_wav=reference['reference_wav'],
                    reference_wav_sha256=reference['reference_wav_sha256'],
                    output_wav=str(root/'audio'/f'{sid}.wav'), seed=original['seed']))
                choices.append(dict(sample_id=sid, mode='zero_shot', instruction_id=instruction,
                    reference_id=rid, role=stimulus['role'], case_id=case_id,
                    reference_wav_sha256=reference['reference_wav_sha256'],
                    target_text_sha256=hashlib.sha256(stimulus['text'].encode()).hexdigest()))
    (root/'audio').mkdir()
    jsonl(root/'manifest.jsonl', assignments)
    write_immutable_json(root/'reference_cards.json', cards)
    write_immutable_json(root/'cases.json', cases)
    write_immutable_json(root/'generation_plan.json', {'schema_version':'generation_plan_v1','choices':choices})
    load_generation_choices(root/'generation_plan.json', {x['sample_id']:x for x in assignments})
    write_immutable_json(root/'protocol.json', dict(schema_version='mas_generation_probe_v1',
        source_root=str(OLD), seed=42, planned_audio=len(assignments),
        design='paired text/patient/seed: same-patient reference 0/1 x default/explicit Cantonese instruction',
        primary_comparison='first attempts; rulebook recoveries reported separately',
        criteria='simple three-text coarse review; no CER cutoff; pending quarantined',
        source_sha256={str(p):sha256_file(p) for p in [REAL,CATALOG,OLD/'protocol.json',OLD/'round_001/proposal.json',
            OLD/'round_001/synthesis_manifest.jsonl',OLD/'round_001/materialization_sidecar.json',Path(__file__),
            REPO/'src/therapist_harness/generation_design.py',REPO/'examples/generate_reviewed_sg.py',
            REPO/'src/therapist_harness/transcript_review.py',REPO/'src/therapist_harness/audio_review.py',
            REPO/'src/therapist_harness/review_worker.py',REPO/'examples/transcribe_sg_review.py',
            REPO/'src/therapist_harness/prompts/transcript_review_v2.md',
            REPO/'synthesis/cosyvoice3_patient_sft/synthesize_manifest.py']},
        independent_test=False, new_targets=False, proposal_reuse='frozen MAS proposal isolates SG actions'))


def report(root):
    cases = {x['case_id']:x for x in read(root/'cases.json')}
    choices = {x['sample_id']:x for x in read(root/'generation_plan.json')['choices']}
    generation = read(root/'generation/generation_summary.json')
    records = generation['outputs']
    summary, sections = {}, []
    for record in records:
        c = choices[record['sample_id']]
        first = record['attempt_records'][0]
        review = first['review']
        key = c['instruction_id']
        summary.setdefault(key, Counter())[review['verdict']] += 1
        e = html.escape
        wav = first['generation']['output_wav']
        sections.append(f'<article><h3>{e(record["sample_id"])}</h3><p>{e(cases[c["case_id"]]["target_text"])}</p>'
                        f'<audio controls src="{e(wav)}"></audio><p>Reference: {e(review["reference_text"])}</p>'
                        f'<p>ASR: {e(review["asr_transcription"])}</p><p>{e(review["verdict"])}: {e(review["reason"])}</p></article>')
    write_immutable_json(root/'summary.json', dict(first_attempt_by_instruction=summary, audio=len(records),
        admitted=sum(x['admitted'] for x in records), word_fidelity_verified=False,
        elapsed_seconds=generation['elapsed_seconds'], timing=generation['timing']))
    write_immutable_bytes(root/'listening.html', ('<!doctype html><meta charset="utf-8"><title>MAS generation probe</title>'
        '<style>body{font:16px system-ui;max-width:1200px;margin:32px auto}main{display:grid;grid-template-columns:1fr 1fr;gap:16px}article{border:1px solid #ccc;padding:16px}</style>'
        '<h1>同患者 reference × instruction 配对试听</h1><p>三文本粗审，不代表逐字保真；pending 不准入训练。</p><main>'
        + ''.join(sections) + '</main>').encode())
    print(json.dumps(read(root/'summary.json'), ensure_ascii=False), flush=True)


def run(root):
    protocol = read(root/'protocol.json')
    for path,digest in protocol['source_sha256'].items():
        assert sha256_file(Path(path)) == digest, path
    p = read(OLD/'protocol.json'); c = p['preference_loop']
    command = [sys.executable,str(REPO/'examples/generate_reviewed_sg.py'),
        '--manifest',str(root/'manifest.jsonl'),'--out-dir',str(root/'generation'),
        '--generation-plan',str(root/'generation_plan.json'),'--review-mode','asr_transcript',
        '--review-allow-pending','--review-asr-worker','--audio-review-model',c['audio_review_model'],
        '--audio-review-base-url',c['audio_review_base_url'],'--review-asr-model',p['base_model_dir'],
        '--review-asr-adapter',c['review_asr_adapter'],'--review-asr-adapter-sha256',c['review_asr_adapter_sha256'],
        '--review-asr-base-model-sha256',c['review_asr_base_model_sha256']]
    write_immutable_json(root/'launch.json', dict(command=command,started_at=time.time(),pid=os.getpid()))
    with (root/'generation.log').open('x') as log:
        subprocess.run(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
    for path,digest in protocol['source_sha256'].items():
        assert sha256_file(Path(path)) == digest, path
    report(root)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','run','report'])
    parser.add_argument('root',type=Path)
    parser.add_argument('--cases',type=int,choices=[1,6],default=6)
    args=parser.parse_args()
    if args.stage=='prepare':prepare(args.root.resolve(),args.cases)
    else:globals()[args.stage](args.root.resolve())
