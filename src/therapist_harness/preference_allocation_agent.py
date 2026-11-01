"""MAS chooses actual audio exposure counts after observing preference availability."""

from collections import Counter
import json
import math
from pathlib import Path
import random

from pydantic import BaseModel, ConfigDict, Field


class AudioExposure(BaseModel):
    model_config = ConfigDict(extra='forbid')
    sample_id: str
    exposures: int = Field(ge=0)
    reason: str = Field(min_length=1, max_length=300)


class ExposureDecision(BaseModel):
    model_config = ConfigDict(extra='forbid')
    hypothesis: str = Field(min_length=1, max_length=1200)
    assignments: list[AudioExposure]
    expected_observation: str = Field(min_length=1, max_length=1200)


def audio_cap(total, available):
    return min(total, max(8, math.ceil(total / available)))


def compile_exposure_allocation(decision, pairs, roles, *, steps, per_step, seed):
    available = {x['audio_id'] for x in pairs}
    if not available:
        raise ValueError('no actual preference audio; no forced negative or training allocation')
    assignments = {x.sample_id: x for x in decision.assignments}
    if len(assignments) != len(decision.assignments) or set(assignments) != available:
        raise ValueError('allocation must explicitly account for every actual paired audio, and no pairless audio')
    total = steps * per_step; cap = audio_cap(total, len(available))
    if sum(x.exposures for x in assignments.values()) != total:
        raise ValueError('MAS audio counts differ from the frozen optimizer exposure budget')
    if any(x.exposures > cap for x in assignments.values()):
        raise ValueError('MAS audio count exceeds the declared per-audio cap')
    sequence = [key for key in sorted(assignments) for _ in range(assignments[key].exposures)]
    random.Random(seed + 2017).shuffle(sequence)
    role_counts = Counter(roles[key] for key in sequence)
    return dict(schema_version='mas_actual_audio_allocation_v1',
        steps=[sequence[i:i + per_step] for i in range(0, total, per_step)],
        audio_exposures={key: assignments[key].exposures for key in sorted(assignments)},
        role_exposures=dict(role_counts), total_audio_exposures=total, per_audio_cap=cap,
        decision=decision.model_dump(mode='json'),
        scope='Counts actual audio occurrences. Distinct rejected transcripts remain averaged within audio by the trainer; their count does not multiply the prescribed dose.')


def validate_allocation_sources(protocol, allocation, pairs, slots):
    from openai.types.responses import Response
    from .agent import terminal_output_text

    paths = {Path(path).name: Path(path) for path in allocation['agent_sources']}
    if set(paths) != {'request.json', 'response.json', 'decision.json'}:
        raise ValueError('actual MAS exposure request, terminal response and decision are required')
    response = Response.model_validate_json(paths['response.json'].read_text())
    decision = ExposureDecision.model_validate_json(terminal_output_text(response))
    if response.status != 'completed' or decision.model_dump(mode='json') != allocation['decision']:
        raise ValueError('audio allocation does not match the actual final MAS response')
    if json.loads(paths['decision.json'].read_text()) != allocation['decision']:
        raise ValueError('stored exposure decision differs from its actual MAS response')
    request = json.loads(paths['request.json'].read_text())
    if request['model'] != protocol.agent.model or request['reasoning'] != {'effort': protocol.agent.reasoning_effort}:
        raise ValueError('allocation request used a different MAS configuration')
    context = json.loads(request['input'][-1]['content'])
    by_id = {}
    for pair in pairs:
        by_id.setdefault(pair['audio_id'], []).append(pair)
    observed = {x['sample_id']: x for x in context['paired_audio']}
    if set(observed) != set(by_id) or len(observed) != len(context['paired_audio']):
        raise ValueError('allocation agent did not see the actual paired audio set')
    for key, values in by_id.items():
        item = observed[key]
        if item['role'] != slots[key]['role'] or item['target_text'] != values[0]['chosen'] or item['actual_distinct_rejected'] != [x['rejected'] for x in values]:
            raise ValueError('allocation agent input differs from actual role, chosen or sampled errors')
    total = protocol.lora.optimizer_steps_per_round * protocol.preference_loop.synthetic_per_step
    if context['total_audio_exposures'] != total or context['max_exposures_per_audio'] != audio_cap(total, len(by_id)):
        raise ValueError('allocation agent was given a different exposure budget')


def choose_exposure_allocation(protocol, proposal, sidecar, pairs, greedy, *, directory, history):
    from openai.lib._pydantic import to_strict_json_schema
    from .agent import execute_therapist_request, serialize_response, terminal_output_text
    from .store import write_immutable_json, sha256_file

    directory = Path(directory); config = protocol.preference_loop
    by_id = {}
    for pair in pairs:
        by_id.setdefault(pair['audio_id'], []).append(pair)
    if not by_id:
        raise ValueError('all sampled audio are pairless; do not call an allocation agent to invent pairs')
    slots = {x['sample_id']: x for x in sidecar['samples']}
    normal = {x['audio_id']: x['prediction'] for x in greedy}
    total = protocol.lora.optimizer_steps_per_round * config.synthetic_per_step
    context = dict(proposal=proposal.model_dump(mode='json'),
        paired_audio=[dict(sample_id=key, role=slots[key]['role'], target_text=values[0]['chosen'],
            greedy_transcript=normal.get(key), actual_distinct_rejected=[x['rejected'] for x in values])
            for key, values in sorted(by_id.items())],
        pairless_samples=[dict(sample_id=key, role=value['role']) for key, value in slots.items() if key not in by_id],
        total_audio_exposures=total, max_exposures_per_audio=audio_cap(total, len(by_id)),
        previous_experiments=history)
    request = dict(model=protocol.agent.model, reasoning={'effort': protocol.agent.reasoning_effort},
        store=protocol.agent.store, stream=True, max_output_tokens=16000,
        input=[dict(role='system', content='''You allocate the existing synthetic audio preference data for the Cantonese-ASR MAS.
Choose an integer exposure count, including zero when appropriate, for EVERY
paired sample_id. Sum exactly to the supplied total and respect the supplied
per-audio cap. Pairless samples cannot be allocated. Use the actual proposal,
normal decode, distinct sampled errors and previous real outcomes to balance
target correction, counterexamples, protection and retained abilities.
More errors or more distinct rejections is not evidence of greater training
value. A coarse review pass is not proof of verbatim acoustic fidelity. You do
not hear audio here. Patient ambiguity alone is not generation failure.
Do not change text, references, rejections, loss, teachers, step count or learning
rate. The fixed trainer averages actual rejections WITHIN each audio occurrence.
Real-training full-GT-prefix KL to initial R-only is separately fixed and is not
real preference sampling or cross entropy. These counts govern synthetic audio
occurrences only. Explain briefly what the allocation tests and the prospective
real observation that would support it. Return concise Chinese explanations.'''),
            dict(role='user', content=json.dumps(context, ensure_ascii=False))],
        text={'format': dict(type='json_schema', name='audio_exposure_decision', strict=True,
            schema=to_strict_json_schema(ExposureDecision))})
    write_immutable_json(directory / 'request.json', request)
    response = execute_therapist_request(protocol, request)
    write_immutable_json(directory / 'response.json', serialize_response(response))
    if response.status != 'completed':
        raise RuntimeError('MAS exposure request did not complete')
    decision = ExposureDecision.model_validate_json(terminal_output_text(response))
    allocation = compile_exposure_allocation(decision, pairs, {k: x['role'] for k, x in slots.items()},
        steps=protocol.lora.optimizer_steps_per_round, per_step=config.synthetic_per_step, seed=protocol.lora.seed)
    write_immutable_json(directory / 'decision.json', decision.model_dump(mode='json'))
    allocation['agent_sources'] = {str(directory / name): sha256_file(directory / name)
        for name in ['request.json', 'response.json', 'decision.json']}
    return allocation
