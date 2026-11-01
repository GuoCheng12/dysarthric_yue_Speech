#!/usr/bin/env python3
"""Ask MAS to revise its design from measured external DPO experiments."""

import argparse
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from mas_selection_dpo_probe import read, save
from mas_patient_retention_probe import request_mas
from therapist_harness.schema import RunProtocol
from therapist_harness.store import sha256_file


EvidenceID = Literal['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8']


class Finding(BaseModel):
    model_config = ConfigDict(extra='forbid')
    statement: str = Field(min_length=1, max_length=1000)
    evidence: list[EvidenceID] = Field(min_length=1, max_length=4)


class DesignReflection(BaseModel):
    model_config = ConfigDict(extra='forbid')
    findings: list[Finding] = Field(min_length=3, max_length=6)
    reference_and_recipe_policy: str = Field(min_length=1, max_length=1500)
    next_content_policy: str = Field(min_length=1, max_length=1500)
    actual_pair_and_exposure_policy: str = Field(min_length=1, max_length=1500)
    checkpoint_acceptance_policy: str = Field(min_length=1, max_length=1000)
    next_probe_if_prefix_retention_helps: str = Field(min_length=1, max_length=1500)
    next_probe_if_prefix_retention_does_not_help: str = Field(min_length=1, max_length=1500)
    unsupported_claims: list[str] = Field(min_length=2, max_length=5)


def main(base):
    from openai.lib._pydantic import to_strict_json_schema

    root = base / 'external_MAS_design_reflection_v1'; root.mkdir(exist_ok=False)
    paths = dict(E1=base / 'adaptive96_matched_feedback_export_v2/proposal_context.json',
        E2=base / 'real2117_residual_feedback_export_v1/proposal_context.json',
        E3=base / 'adaptive96_preference_score_audit_v2/results.json',
        E4=base / 'adaptive96_matched_feedback_export_v2/chosen_sampling_support.json',
        E5=base / 'adaptive96_opportunity_allocation_v1/results.json',
        E6=base / 'adaptive96_greedy_preference_v1/results.json',
        E7=base / 'ASR_target_format_training_seeds_v1/results.json',
        E8=base / 'real_KL_greedy_prefix_inputs_v1/tokenization_audit.json')
    evidence = {k: read(p) for k, p in paths.items()}
    evidence['E3'] = {k: evidence['E3'][k] for k in ['status', 'groups', 'max_difference_from_original_frozen_reference_edit_scores', 'interpretation']}
    evidence['E4'] = dict(interpretation=evidence['E4']['interpretation'],
        arms={a:{k:v for k,v in x.items() if k != 'cases'} for a,x in evidence['E4']['arms'].items()})
    metrics = lambda x: {k:x[k] for k in ['CER_percent', 'edits', 'macro_CER_percent', 'hard_edits', 'eligible']}
    evidence['E5'] = dict(MAS_allocation=metrics(evidence['E5']['MAS_allocation']),
        delta_vs_original_role_allocation_pp=evidence['E5']['delta_vs_original_role_allocation_pp'],
        bootstrap95_pp=evidence['E5']['bootstrap95_pp'])
    for key in ['E6', 'E7']:
        source = evidence[key]
        evidence[key] = dict(arms={a:metrics(x) for a,x in source['arms'].items()},
            mean_CER_percent=source.get('mean_CER_percent'), bootstrap95_pp=source.get('bootstrap95_pp'))
    evidence['E8'] = {k:v for k,v in evidence['E8'].items() if k != 'records'}
    evidence['E8']['training_result'] = 'Unknown at this request binding; actual matched12-step prefix-retention experiment is running.'
    payload = json.dumps(evidence, ensure_ascii=False)
    assert '/data/' not in payload and 'vlink' not in payload
    source_protocol = base / 'adaptive_mas96_data_v1/protocol.json'
    p = RunProtocol.model_validate(read(source_protocol))
    request = dict(model=p.agent.model, reasoning={'effort':p.agent.reasoning_effort},
        store=False, stream=True, max_output_tokens=10000,
        input=[dict(role='system', content='''You are the MAS design/reflection agent for a Cantonese dysarthric-ASR training loop.
Revise the design from the eight supplied measured evidence records. Respond in Chinese,
with concise findings citing their evidence IDs and concrete prospective design rules.
Mainline: actual real errors -> MAS new practice content/reference/recipe -> TTS ->
simple three-text coarse review -> actual current-ASR negatives -> DPO -> real evaluation.
Inference postprocessing cannot substitute for trained-ASR performance. There is no
stable gain over initial R-only yet, and no endpoint has been promoted.
E1 is the fresh same-content/reference-policy comparison, including actual patient,
target-position and delivery feedback. E2 is matched full2117 realCE versus the sameCE
plus MAS DPO. E3 checks absolute target scores, not just relative margins. E4 describes
actual correct-draw support; it is not an acoustic-fidelity certificate or an admission
threshold. E5 is the failed actual MAS exposure reallocation; E6 compares actual greedy
versus sampled wrong negatives. E7 is the earlier three-training-seed tied result.
E8 measures GT versus reference-greedy prefix differences; its DPO result is UNKNOWN.
Distinguish a causal result, a measured association and a hypothesis. Do not claim that
every TTS label is noise, that every teacher-forced improvement transfers to real speech,
or that the same surface error proves multiple independent impairment mechanisms.
MAS may choose concrete references/recipes from its existing same-patient catalog,
new natural content informed by concrete surface confusions, and actual pair/exposure
budgets. A content-purpose label is not a separate loss or proof of delivered learning.
Keep coarse review simple: target/reference/ASR text for obvious leakage, truncation,
off-topic output or mechanical looping; existing retry/instruct2 rulebook. No listening
claim, CER rejection threshold, phoneme audit, or rejection just because speech is hard.
Repeated real ASR mistakes on a generated audio are not by themselves generation failure.
Keep actual negative provenance, deduplicate errors, skip all-correct DPO cases without
inventing negatives, and allow one repeated actual error to form one pair.
Both scientific arms start at the same accepted parent; fixed DPO reference remains
initial R-only, distinct from TTS reference speech. Real2117 anddev511 stay isolated.
Suggest ONE bounded next experiment for each pending-prefix outcome. State matched
controls and an observation that would change the proposed policy. Avoid another broad
hyperparameter sweep, recycling failed exposure changes without a new distinction,
or demanding new permission/listening/review machinery. You are interpreting completed
external probes; do not claim a retroactively committed core ReflectionPack/ledger or
that the proposed next experiment has already run.'''), dict(role='user',content=payload)],
        text={'format':{'type':'json_schema', 'name':'external_mas_design_reflection', 'strict':True,
                        'schema':to_strict_json_schema(DesignReflection)}})
    files = [Path(__file__), Path(__file__).with_name('mas_patient_retention_probe.py'), source_protocol, *paths.values()]
    hashes = {str(f):sha256_file(f) for f in files}
    save(root / 'input_sources.json', hashes); save(root / 'request.json', request)
    response = request_mas(p, request, root)
    assert response['status'] == 'completed'
    final = [x for x in response['output'] if x.get('type') == 'message' and x.get('phase') == 'final_answer']
    assert len(final) == 1
    reflection = DesignReflection.model_validate_json(''.join(x['text'] for x in final[0]['content'] if x['type'] == 'output_text'))
    assert all(sha256_file(Path(f)) == h for f,h in hashes.items())
    save(root / 'reflection.json', reflection.model_dump(mode='json'))
    save(root / 'receipt.json', dict(status='complete', actual_MAS_API_calls=1,
        response_id=response['id'], request_sha256=sha256_file(root / 'request.json'),
        response_sha256=sha256_file(root / 'response.json'), source_ledger_mutated=False,
        actual_new_proposal_or_training=False, pending_prefix_result_was_not_in_request=True))
    print(dict(status='complete', reflection=str(root / 'reflection.json')), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('base', type=Path)
    main(parser.parse_args().base.resolve())
