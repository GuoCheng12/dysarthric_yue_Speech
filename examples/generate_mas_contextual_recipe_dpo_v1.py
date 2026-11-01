#!/usr/bin/env python3
"""Four-patient smoke and verified reuse for the new matched SG recipe experiment."""
import argparse
from pathlib import Path
import subprocess
import sys
import time

import mas_contextual_recipe_dpo_data_v1 as data

base, ROOT, REPO = data.base, data.ROOT, data.base.REPO


def generate(arm, smoke):
    p = base.verify(); c = p.preference_loop
    assert all(base.sha256_file(Path(f)) == h for f, h in base.read(ROOT / 'plan.json')['manifest_hashes'].items())
    parent = ROOT / arm
    out = ROOT / (f'smoke_{arm}' if smoke else arm)
    if smoke:
        out.mkdir(exist_ok=False); (out / 'audio').mkdir()
        manifest = base.rows(parent / 'manifest.jsonl')[:4]
        assert len(manifest) == 4 and all(x['sample_id'].startswith('recipe24.t0.') for x in manifest)
        for x in manifest:
            x['output_wav'] = str(out / 'audio' / Path(x['output_wav']).name)
        base.jsonl(out / 'manifest.jsonl', manifest)
        ids = {x['sample_id'] for x in manifest}
        plan = base.read(parent / 'generation_plan.json')
        plan['choices'] = [x for x in plan['choices'] if x['sample_id'] in ids]
        base.save(out / 'generation_plan.json', plan)
    command = [sys.executable, str(REPO / 'examples/generate_reviewed_sg_recovery_v2.py'),
        '--manifest', str(out / 'manifest.jsonl'), '--out-dir', str(out / 'generation'),
        '--generation-plan', str(out / 'generation_plan.json'), '--review-mode', 'asr_transcript',
        '--review-asr-worker', '--review-allow-pending', '--audio-review-model', c.audio_review_model,
        '--audio-review-base-url', c.audio_review_base_url, '--review-asr-model', p.base_model_dir,
        '--review-asr-adapter', c.review_asr_adapter, '--review-asr-adapter-sha256', c.review_asr_adapter_sha256,
        '--review-asr-base-model-sha256', c.review_asr_base_model_sha256]
    if not smoke:
        previous = ROOT / f'smoke_{arm}' / 'generation'
        assert len(base.read(previous / 'generation_summary.json')['outputs']) == 4
        sources = sorted(previous.rglob('assignment_result.json'))
        assert len(sources) == 4
        base.save(out / 'reuse.json', dict(assignments=[dict(
            sample_id=base.read(f)['sample_id'], assignment_result=str(f),
            assignment_result_sha256=base.sha256_file(f),
            provenance='Completed four-patient smoke in this execution; generator verifies identity, audio and review hashes.')
            for f in sources]))
        command += ['--reuse-manifest', str(out / 'reuse.json')]
    files = [Path(__file__), ROOT / 'sources.json', ROOT / 'plan.json',
        out / 'manifest.jsonl', out / 'generation_plan.json', Path(command[1])]
    base.save(out / 'generation_launch.json', dict(command=command, start=time.time(),
        source_sha256={str(f): base.sha256_file(f) for f in files}, smoke=smoke))
    with (out / 'generation.log').open('x') as log:
        subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
    base.verify()
    print(f'Completed {arm} generation; smoke={smoke}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['generate', 'sample'])
    parser.add_argument('--arm', choices=['MAS', 'fixed'], required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.stage == 'generate':
        generate(args.arm, args.smoke)
    else:
        assert not args.smoke
        base.sample(args.arm)
