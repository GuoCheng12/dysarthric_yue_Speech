#!/usr/bin/env python3
"""Execution extension: an observed empty-token SG failure uses the finite rulebook.

No audio or Agent verdict is invented for an empty output. Other runtime errors
still stop execution. The existing generator/reviewer and frozen core are reused.
"""

import json
from pathlib import Path
import sys

import generate_reviewed_sg as base


def run_attempts(initial_seed, generate, review):
    results=[]
    for attempt,mode in enumerate(('zero_shot','zero_shot','instruct2')):
        seed=initial_seed+attempt
        try:
            generation=generate(mode,seed,attempt)
        except RuntimeError as error:
            if str(error)!='SG completed with no speech tokens':raise
            results.append(dict(attempt=attempt,mode=mode,seed=seed,generation=None,review=None,
                action='retry_known_empty_speech',technical_failure='SG completed with no speech tokens',
                actual_audio_created=False,actual_agent_call=False))
            if attempt==2:raise
            continue
        judgment=review(generation,attempt)
        action=base.next_generation_action(judgment,attempt)
        results.append(dict(attempt=attempt,mode=mode,seed=seed,generation=generation,
            review=judgment.to_dict() if hasattr(judgment,'to_dict') else judgment.model_dump(),action=action))
        if action in ('keep','exclude','quarantine'):return results,action
    raise AssertionError('finite recovery rulebook must terminate')


def main(argv=None):
    args=base.parse_args(argv)
    extension=dict(version='known_empty_speech_recovery_v1',path=str(Path(__file__).resolve()),
        sha256=base.sha256_file(Path(__file__)),
        observed_failure='SG completed with no speech tokens',
        rule='Only this exact observed no-output error advances to the next existing seed/mode slot; no audio or review is fabricated. Other technical errors still stop.')
    original_writer=base.write_immutable_json
    def write(path,payload):
        if Path(path).name in {'generation_config.json','runtime.json','generation_summary.json','generation_failure.json'}:
            payload=dict(payload,execution_extension=extension)
            if 'technical_failure_policy' in payload:
                payload['technical_failure_policy']='known empty speech uses the finite rulebook; all other technical failures stop'
        original_writer(path,payload)
    base.write_immutable_json=write
    base.run_attempts=run_attempts
    try:
        result=base.run(args)
        print(json.dumps(dict(status=result['status'],generated=result['output_count'],admitted=result['admitted_count'])),flush=True)
        return 0
    except Exception as error:
        print(json.dumps(dict(status='failed',error_type=type(error).__name__,output=str(args.out_dir))),flush=True)
        return 1


if __name__=='__main__':raise SystemExit(main())
