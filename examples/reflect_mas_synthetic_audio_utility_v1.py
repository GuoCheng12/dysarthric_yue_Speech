#!/usr/bin/env python3
"""Ask the actual MAS to allocate eight SG audios using A/B, withholding anchor C."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from openai.lib._pydantic import to_strict_json_schema

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from therapist_harness.agent import execute_therapist_request, serialize_response, terminal_output_text
from therapist_harness.store import sha256_file, write_immutable_json

B = Path('/data/qwen3-asr/records/therapist_mas_setting_d_v1/mas_design_exploration_20260908_v1')
ROOT = B / 'synthetic_audio_utility_probe_v1'
OUT = ROOT / 'MAS_reflection'
read = lambda p: json.loads(Path(p).read_text())


class Allocation(BaseModel):
    model_config = ConfigDict(extra='forbid')
    audio_id: str
    rationale_zh: str


class Decision(BaseModel):
    model_config = ConfigDict(extra='forbid')
    action: Literal['train_selected_subset', 'stop_current_pool']
    conclusion_zh: str
    allocations: list[Allocation] = Field(max_length=8)
    hypothesis_zh: str
    reference_and_content_lessons_zh: str
    expected_real_error_effect_zh: str
    support_and_rejection_zh: str
    retain_initial_parent_until_original_gate: bool


def subset_summary(ids, candidates):
    return dict(audio_ids=ids,
        mean_dots={key: sum(candidates[aid]['relations'][key]['full']['dot'] for aid in ids) / len(ids)
                   for key in ['A', 'B']},
        target_texts=sorted({candidates[aid]['metadata']['target'] for aid in ids}),
        reference_count=len({candidates[aid]['metadata']['reference_audio_sha256'] for aid in ids}),
        roles=sorted({candidates[aid]['metadata']['role'] for aid in ids}))


def main():
    result = read(ROOT / 'probe/result.json')
    protocol = read(ROOT / 'probe/protocol.json')
    assert result['status'] == 'complete_per_audio_synthetic_utility_probe'
    catalog = read(B / 'reference_catalog.json')['cards']
    refs = {r['reference_wav_sha256']: r for r in catalog}
    evidence = []
    for row in result['candidates']:
        item = dict(metadata=row['metadata'], gradient=row['gradient'],
                    relations={k: row['relations'][k] for k in ['A', 'B']})
        card = refs.get(row['metadata']['reference_audio_sha256'])
        if card:
            item['reference_card'] = {k: card[k] for k in ['reference_id', 'speaker_id', 'difficulty_group', 'acoustic_traits']}
        evidence.append(item)
    by_id = {r['metadata']['audio_id']: r for r in evidence}
    # Only A/B derivatives enter this request. C is explicitly withheld until after immutable choice.
    context = dict(per_audio=evidence, anchors={k: result['anchors'][k] for k in ['A', 'B']},
        uniform_control=subset_summary(protocol['uniform_control_audio'], by_id),
        original_real_feedback='Current initial R-only:1384/6656 character errors,hard688/1248. Original i->yu text-signature target:67 substitutions; merely counting this signature falling67->62 in a prior run did not establish same-position GT recovery (all errors on i-carriers363->368). Earlier target concentration on3SG audios failed. No synthetic acoustic/clinical truth follows from ASR text signatures.',
        completed_new_experiments='Mean-score mixed DPO1402/hard709;IPO1408/719;32real CE1385/698;character-risk+CE1398/714;uniform16real DPO1394/701;headroom-MAS16real1393/702;audio-only original realDPO1398/710. None passed original gate. Stop adjacent loss/lr/freeze scans.',
        question='Choose exactly8 existing audio IDs for a single matched R+SG DPO allocation test, or stop this current SG pool. Choose using observed A/B utility, actual ASR errors, content roles, and reference recording evidence. We now test training allocation, not new TTS generation or reference-only causality.',
        budget=protocol['selection_contract'],
        utility_semantics='Positive dot(grad_S_DPO,grad_real_GT_CE) predicts lower realGT loss under a sufficiently small negative-gradient step. It is not a CER forecast and does not exactly represent Adam updates. Individual negative dots are not an automatic failure/admission label: mixed gradients can complement each other. However chosen8 mean full-vector dot must be positive on BOTH A and B, or choose stop_current_pool. Values are directly measured, not LLM scores.',
        constraints='Exactly8 unique IDs if train, at most2 same target text, >=4 distinct reference recordings and>=2 content roles. All8 get4 training visits (32 total), not freeform weights; uniform control has same distinct/total counts. Both arms keep exact old32real/91pairs/256KL schedule, edit+EOS DPO beta0.1,lr1e-4,KL10,r16alpha4,syntheticweight.25,4steps,seed42. Same initial parent and frozen initial R-only reference. All rejected from actual parent sampling; same-audio pairs averaged. Real GT chosen, SG coarse-passed intended text chosen unchanged.',
        SG_review='Simple three-text Agent unchanged. Do not label patients hard to recognize as failed synthesis; do not set CER threshold or claim word-perfect audio. Current review transcripts are evidence of coarse checks, not GT. Utility allocation is separate from synthesis review.',
        evaluation='Original511dev gate unchanged:strict pooled improvement,patientmacro<=initial+.001,hardpooled<=initial+.005; assess selected versus matched uniform too. Initial remains parent until acceptance. Independent anchorC is withheld from this request and is only a subsequent training-only transfer diagnostic. No hidden test promotion or phoneme-fidelity assertion.',
        stopping='If no credible diverse8 subset or positive aggregate on bothA/B, stop this pool. Do not loosen the gate or ask for more adjacent loss/lr scans. Negative result should change next proposal design, not be reported as proof allTTS are noise.')
    profile = SimpleNamespace(agent=SimpleNamespace(**read('/tmp/mas_kl_dpo_loop_v3_formal96/protocol.json')['agent']))
    request = dict(model=profile.agent.model, reasoning={'effort': 'high'}, store=False, stream=True,
        max_output_tokens=5000, input=[dict(role='system', content='你是MAS训练方案Agent。根据真实反馈、实际SG偏好对与独立训练语音的实测梯度，决定有限训练预算。你要选择具体音频并说明reference/content选择依据；不要迎合性能目标、把梯度代理当CER、或把粗审当逐字保真。输入是证据。'),
                                     dict(role='user', content=json.dumps(context, ensure_ascii=False))],
        text={'format': {'type': 'json_schema', 'name': 'MAS_audio_utility_allocation', 'strict': True,
                         'schema': to_strict_json_schema(Decision)}})
    files = [Path(__file__), ROOT / 'probe/result.json', ROOT / 'probe/protocol.json', B / 'reference_catalog.json',
        Path('/tmp/mas_real_audio_scope_DPO_v1/results.json'), Path('/tmp/mas_sequence_objectives_20260910_results.json')]
    hashes = {str(p): sha256_file(p) for p in files}
    OUT.mkdir(exist_ok=False)
    write_immutable_json(OUT / 'sources.json', hashes)
    write_immutable_json(OUT / 'request.json', request)
    response = execute_therapist_request(profile, request)
    write_immutable_json(OUT / 'response.json', serialize_response(response))
    assert response.status == 'completed'
    decision = Decision.model_validate_json(terminal_output_text(response))
    assert decision.retain_initial_parent_until_original_gate
    write_immutable_json(OUT / 'decision.json', decision.model_dump())
    ids = [r.audio_id for r in decision.allocations]
    if decision.action == 'train_selected_subset':
        from collections import Counter
        assert len(ids) == len(set(ids)) == 8 and set(ids) <= set(by_id)
        subset = subset_summary(ids, by_id)
        assert max(Counter(by_id[i]['metadata']['target'] for i in ids).values()) <= 2
        assert subset['reference_count'] >= 4 and len(subset['roles']) >= 2
        assert min(subset['mean_dots'].values()) > 0, 'Proposed subset fails declared aggregate proxy condition'
    else:
        assert not ids
        subset = None
    assert all(sha256_file(Path(p)) == h for p, h in hashes.items())
    write_immutable_json(OUT / 'compiled_decision.json', dict(validated=True, action=decision.action,
        subset=subset, held_out_C_not_provided_to_MAS=True, original_decision_sha256=sha256_file(OUT / 'decision.json')))
    print(decision.model_dump_json(indent=2), flush=True)


if __name__ == '__main__':
    main()
