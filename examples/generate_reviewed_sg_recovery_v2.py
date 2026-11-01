#!/usr/bin/env python3
"""Finite retries for observed empty-token and sub-250 ms SG breakdowns."""

import json
from pathlib import Path
import traceback

import generate_reviewed_sg_resumable as base


def run_attempts(initial_seed,generate,review):
    results=[]
    for attempt,mode in enumerate(('zero_shot','zero_shot','instruct2')):
        seed=initial_seed+attempt
        try:generation=generate(mode,seed,attempt)
        except RuntimeError as error:
            message=str(error);duration=None
            if message.startswith('Speech Generator returned implausible duration for '):
                duration=float(message.rsplit(': ',1)[1])
            if message!='SG completed with no speech tokens' and not (duration is not None and 0<=duration<.25):raise
            record=dict(attempt=attempt,mode=mode,seed=seed,generation=None,review=None,
                action='retry_observed_sg_breakdown',technical_failure=message,
                actual_audio_created=False,actual_agent_call=False,
                temporary_waveform_deleted_by_existing_validator=duration is not None,
                observed_duration_seconds=duration)
            results.append(record);print(json.dumps(dict(sg_recovery=record)),flush=True)
            if attempt==2:raise
            continue
        judgment=review(generation,attempt);action=base.next_generation_action(judgment,attempt)
        results.append(dict(attempt=attempt,mode=mode,seed=seed,generation=generation,
            review=judgment.to_dict() if hasattr(judgment,'to_dict') else judgment.model_dump(),action=action))
        if action in ('keep','exclude','quarantine'):return results,action
    raise AssertionError('finite rulebook must terminate')


def main():
    args=base.parse_args()
    extension=dict(version='observed_sg_breakdown_recovery_v2',path=str(Path(__file__).resolve()),
        sha256=base.sha256_file(Path(__file__)),
        resumable_base_path=str(Path(base.__file__).resolve()),resumable_base_sha256=base.sha256_file(Path(base.__file__)),
        observed_failures=['empty speech tokens','existing duration validator rejected 0.08-second output'],
        rule='Only empty or sub-0.25-second outputs advance through the existing three seed/mode slots. Completed pending/quarantined outputs may be reused without admission. Existing waveform validation and three-text Agent standard unchanged; no audio or review invented.')
    writer=base.write_immutable_json
    def write(path,payload):
        if Path(path).name in {'generation_config.json','runtime.json','generation_summary.json','generation_failure.json'}:
            payload=dict(payload,execution_extension=extension)
            if 'technical_failure_policy' in payload:payload['technical_failure_policy']=extension['rule']
        writer(path,payload)
    base.write_immutable_json=write;base.run_attempts=run_attempts
    try:
        result=base.run(args)
        print(json.dumps(dict(status=result['status'],generated=result['output_count'],admitted=result['admitted_count'])),flush=True)
        return 0
    except Exception as error:
        traceback.print_exc()
        print(json.dumps(dict(status='failed',error_type=type(error).__name__,output=str(args.out_dir))),flush=True)
        return 1


if __name__=='__main__':raise SystemExit(main())
