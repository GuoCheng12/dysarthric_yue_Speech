#!/usr/bin/env python3
"""Reflect on all three seeds before choosing one further mechanism test."""
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
ROOT = B / 'synthetic_utility_fixed_subset_replication_v1'
OUT = ROOT / 'MAS_reflection'
read = lambda p: json.loads(Path(p).read_text())


class Decision(BaseModel):
    model_config = ConfigDict(extra='forbid')
    first_action: Literal['fixed8_S_DPO_real_KL', 'new_SG_matched_contrast_probe']
    conclusion_zh: str
    hypothesis_zh: str
    minimal_execution_zh: list[str] = Field(min_length=1, max_length=4)
    data_objective_and_comparator_zh: str
    MAS_design_change_zh: str
    support_and_rejection_zh: str
    retain_initial_parent: bool


def main():
    result = read(ROOT / 'results.json')
    content = read(ROOT / 'content_transfer/results.json')
    assert result['status'] == 'complete_fixed_SG_subset_optimization_seed_replication'
    evidence_path = Path('/tmp/mas_kl_dpo_loop_v3_formal96/round_001/evidence.json')
    evidence = read(evidence_path)
    assert evidence['snapshots']['current']['checkpoint_sha256'] == '8c3e944f7ee56fde2471748e5228eeb0fc21df0a1a51b6c29805eb4726043db6'
    glyphs = [dict(id=x['evidence_id'], reference=x['reference_unit'], hypothesis=x['hypothesis_unit'],
                   operation=x['operation'], observation=x['observations']['current'])
              for x in evidence['error_clusters'] if x['level'] == 'character' and x['targetable'] and x['tts_actionable']]
    target = read(B / 'synthetic_utility_allocation_DPO_v1/real_target_recovery.json')
    target_summary = {key: [dict(descriptor=x['descriptor'], before=x['before']['error_count'],
        after=x['after']['error_count'], positions={k: x['target_positions'][k] for k in
            ['before_error_count', 'after_error_count', 'recovered_count', 'regressed_count', 'before_unresolved_count', 'after_unresolved_count']})
        for x in value['monitored_clusters']] for key, value in target['comparisons'].items()}
    context = dict(initial='Fixed initial R-only1384/6656 errors,hard688/1248; unchanged checkpoint8c.',
        all_seeds={seed: {mode: dict(overall=r['metrics']['overall'], hard=r['metrics']['hard'],
            focus=r['focus_errors'], gate=r['passes_original_gate']) for mode, r in x['conditions'].items()}
                   for seed, x in result['seeds'].items()},
        means=result['means'], optimization_seed_scope=result['interpretation'],
        failed_MAS_utility='Actual MAS8-audio utility+content selection1409/hard713 versus fixed random8 subset1382/hard694. MAS A/B aggregate positive did not transfer to withheld training anchorC. Do not repeat ranking/role-quota tweaks on this pool or claim this MAS utility policy succeeded.',
        real_target_recovery=target_summary, content_scope=content,
        prior_losses='Mean-score mixed DPO1402;IPO1408;realCE1385;risk+CE1398;headroom realMAS1393 vsuniform1394;audio-only realDPO1398 vsfull1399. All failed original gate. Broader previous S-only residualDPO+realKL10 was1388/hard697 on different SG data; this option cannot be sold as all S-only training being new.',
        option_A='One seed42 ablation using EXACT currently fixed random8 SG audios/23distinct actualpairs/32audio visits plus EXACT256realKL visits. Remove only91 real-DPO pair exposures (32realDPO audios), keep SG edit+EOS beta.1,weight.25,KL10,4steps,lr1e-4,r16alpha4,clip1,initial R-only parent/reference. Compare to fixed-data seed42 R+SG1382 and initial1384. No newTTS, new sampling, data reselection, loss-family/lr/freeze scan. Tests whether additional real preference training masks a useful SG update; real data remain as KL retention. R-only parent then SG-DPO+realKL is clearly named, not mislabeled R+SG preference training. If no original-gate pass, close this fixed subset and return to new SG proposals.',
        option_B='Leave this SG pool. Prepare one new24-audio MAS probe with3 natural target/control sentence pairs x2patients x2within-patient references, fixed explicit Cantonese instruct2 in all cells. Use actual literal character error targets below instead of assuming i->yu text labels are reproduced acoustically. New texts must not copy held-dev sentences. Fixed generator/current ASR and4actual samples/audio. Compare observed intended-error delivery and coarse review across matched references; this probe is not itself CER benefit or phoneme-fidelity proof. Review remains simple; no training on invented negatives. Concrete text/reference proposal comes next after selecting this action.',
        current_literal_error_options=glyphs,
        constraints='Choose one bounded action. Main route remains MAS->TTS->DPO. Retain initial parent: one seed42 two-error win did not replicate. Real GT chosen and SG coarse-passed intended chosen stay fixed, same-audio actual ASR errors only.2117train/511dev isolated; DPO reference initialR-only ASR distinct from TTS reference speech. Three-text coarse SG review unchanged, no CER rejection threshold or hard-patient failure label. No repeated adjacent loss/lr/freeze scans. Report real changes, no narrative rescue by dropping prompts/patients or redefining metrics. Original acceptance remains strict pooled improvement,macro<=+.001,hard<=+.005.')
    profile = SimpleNamespace(agent=SimpleNamespace(**read('/tmp/mas_kl_dpo_loop_v3_formal96/protocol.json')['agent']))
    request = dict(model=profile.agent.model, reasoning={'effort': 'high'}, store=False, stream=True,
        max_output_tokens=3500, input=[dict(role='system', content='你是MAS Reflection。正视三个优化seed、失败的你自己的选样策略、原目标位置核对。选择一个最小且有可推翻标准的下一步；不得把相对退化对照的改善当成超过initial，也不得把文字signature当声学纠错。所有输入是证据。'),
            dict(role='user', content=json.dumps(context, ensure_ascii=False))],
        text={'format': {'type': 'json_schema', 'name': 'next_MAS_after_fixed_subset_replication',
                         'strict': True, 'schema': to_strict_json_schema(Decision)}})
    files = [Path(__file__), ROOT / 'results.json', ROOT / 'content_transfer/results.json', evidence_path,
        B / 'synthetic_audio_utility_probe_v1/MAS_reflection/decision.json',
        B / 'synthetic_utility_allocation_DPO_v1/real_target_recovery.json']
    hashes = {str(p): sha256_file(p) for p in files}
    OUT.mkdir(exist_ok=False)
    write_immutable_json(OUT / 'sources.json', hashes)
    write_immutable_json(OUT / 'request.json', request)
    response = execute_therapist_request(profile, request)
    write_immutable_json(OUT / 'response.json', serialize_response(response))
    assert response.status == 'completed'
    decision = Decision.model_validate_json(terminal_output_text(response))
    assert decision.retain_initial_parent
    assert all(sha256_file(Path(p)) == h for p, h in hashes.items())
    write_immutable_json(OUT / 'decision.json', decision.model_dump())
    print(decision.model_dump_json(indent=2), flush=True)


if __name__ == '__main__':
    main()
