"""Carry actual generation attempts into the next reference-selection request."""

from pathlib import Path

from mas_selection_dpo_probe import read
from therapist_harness.store import RunStore, sha256_file


def summarize_event(event):
    paths = event.artifact_paths
    names = ['generation_receipt', 'preference_audit', 'preference_greedy']
    for name in names:
        if sha256_file(Path(paths[name])) != event.artifact_sha256[name]:
            raise ValueError('generation experience artifact changed: ' + name)
    summary = read(paths['generation_receipt'])
    audit = read(paths['preference_audit'])
    by_audio = {x['audio_id']: x for x in audit['audio_results']}
    import json
    greedy = {x['audio_id']: x['prediction'] for x in
              (json.loads(line) for line in Path(paths['preference_greedy']).read_text().splitlines())}
    observations = []
    for output in summary['outputs']:
        attempts = []
        for item in output['attempt_records']:
            generation = item.get('generation') or {}
            review = item.get('review')
            attempts.append(dict(attempt=item['attempt'], mode=item['mode'], action=item['action'],
                instruction_id=generation.get('effective_instruction_id'),
                reference_id=generation.get('generation_choice', {}).get('reference_id'),
                audio_sha256=generation.get('output_wav_sha256'),
                asr_transcription=None if review is None else review.get('asr_transcription'),
                verdict=None if review is None else review.get('verdict'),
                reason=None if review is None else review.get('reason'),
                agent_review_executed=review is not None,
                technical_details={k: v for k, v in item.items()
                                   if k not in ['generation', 'review', 'attempt', 'mode', 'action']}))
        audio_id = output['sample_id']
        sample_audit = by_audio.get(audio_id)
        observations.append(dict(sample_id=audio_id, target_text=output['target_text'],
            reference_text=output['reference_text'], final_reference_id=output['generation_choice']['reference_id'],
            admitted=output['admitted'], attempts=attempts,
            current_parent_greedy=greedy.get(audio_id),
            correct_random_draws=None if sample_audit is None else sample_audit['correct_samples'],
            actual_pair_count=None if sample_audit is None else sample_audit['pair_count']))
    return dict(round_index=event.round_index, current_asr_sha256=audit['current_asr_sha256'],
        source_sha256={paths[n]: event.artifact_sha256[n] for n in names}, observations=observations,
        interpretation='Actual previous-round text evidence, including failed attempts before recovery. Coarse pass does not prove pronunciation. Missing review means no Agent review, not a failure verdict. No CER ranking or new admission criterion.')


def latest_committed_experience(root):
    store = RunStore(root)
    store.verify_integrity()
    events = store.events()
    committed = {e.round_index for e in events if e.phase == 'committed'}
    if not committed:
        return None
    latest = max(committed)
    event = next(e for e in events if e.round_index == latest and e.phase == 'synthesized')
    return summarize_event(event)
