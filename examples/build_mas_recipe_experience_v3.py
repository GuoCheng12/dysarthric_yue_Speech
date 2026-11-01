#!/usr/bin/env python3
"""Mode/instruction-specific reference evidence; no quality score or new gate."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from therapist_harness.store import sha256_file,write_immutable_json,write_immutable_bytes

B=Path('/data/qwen3-asr/records/therapist_mas_setting_d_v1/mas_design_exploration_20260908_v1')
OUT=B/'reference_recipe_experience_v3'
read=lambda p:json.loads(Path(p).read_text())
rows=lambda p:[json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def main():
    OUT.mkdir(exist_ok=False)
    legacy_path=B/'reference_experience_v1/observations.jsonl'
    legacy=rows(legacy_path);assert len(legacy)==255
    catalog_path=B/'reference_catalog.json';catalog=read(catalog_path)['cards']
    audit_path=B/'instruct2_reference_conditioning_audit_v1/result.json';audit=read(audit_path)
    assert audit['status']=='passed_actual_fixed_SG_frontend_conditioning_audit'
    files=[Path(__file__),legacy_path,catalog_path,audit_path]
    groups=defaultdict(lambda:dict(legacy=[],attempts=[]))
    for x in legacy:
        key=(x['reference_id'],x['mode'],x['instruction_id'],x['speech_generator_sha256'],x['review_asr_sha256'])
        groups[key]['legacy'].append(x)
    attempts=[];final_hashes=set()
    for name in ['literal_matched_SG24_v1','contextual_matched_SG24_v2','mode_reference_probe_v1']:
        for arm in ['A','B']:
            root=B/name/arm;path=root/'generation/generation_summary.json';gen=read(path)
            choices={x['sample_id']:x for x in read(root/'generation_plan.json')['choices']}
            receipt_path=root/'sampling/sampling_receipt.json';receipt=read(receipt_path)
            assert receipt['status']=='complete'
            sample_audit_path=root/'sampling/preference_audit.json'
            sample_audit={x['audio_id']:x for x in read(sample_audit_path)['audio_results']}
            sg_hash=gen['runtime']['speech_generator_sha256'];review_asr_hash=gen['review_asr_adapter_sha256']
            for output in gen['outputs']:
                sid=output['sample_id'];choice=choices[sid]
                assert output['output_wav_sha256'] not in final_hashes
                final_hashes.add(output['output_wav_sha256'])
                for index,attempt in enumerate(output['attempt_records']):
                    generated=attempt['generation'];review=attempt['review']
                    # A recorded rulebook slot name alone is not its effective mode.
                    mode=generated['generation_mode'] if generated else (choice['mode'] if index<2 else 'instruct2')
                    instruction=generated['effective_instruction_id'] if generated else choice['instruction_id']
                    if mode=='instruct2' and instruction=='default':instruction='cantonese_zh'
                    if generated:assert sha256_file(Path(generated['output_wav']))==generated['output_wav_sha256']
                    final=index==len(output['attempt_records'])-1
                    item=dict(source_run=name,arm=arm,sample_id=sid,attempt=index,first_attempt=index==0,final_attempt=final,
                        reference_id=choice['reference_id'],mode=mode,instruction_id=instruction,
                        speech_generator_sha256=sg_hash,review_asr_sha256=review_asr_hash,
                        audio_sha256=generated['output_wav_sha256'] if generated else None,
                        verdict=review['verdict'] if review else 'technical_failure',reason=review['reason'] if review else attempt['technical_failure'],
                        target_text=output['target_text'],review_asr_transcription=review['asr_transcription'] if review else None,
                        final_admitted=output['admitted'] if final else None,
                        final_sampling_observation=sample_audit.get(sid) if final else None,
                        final_sampling_receipt=dict(path=str(receipt_path),sha256=sha256_file(receipt_path)) if final else None,
                        assignment_result_source=dict(summary=str(path),summary_sha256=sha256_file(path),sample_id=sid,attempt=index))
                    attempts.append(item)
                    groups[(choice['reference_id'],mode,instruction,sg_hash,review_asr_hash)]['attempts'].append(item)
            files += [path,root/'generation_plan.json',receipt_path,sample_audit_path]
    assert len(final_hashes)==56 and len(attempts)==64
    assert sum(x['verdict']=='technical_failure' for x in attempts)==4
    assert sum(x['verdict']=='obvious_failure' for x in attempts)==4
    cards=[]
    for reference in catalog:
        conditions=[]
        for key,data in sorted(groups.items()):
            if key[0]!=reference['reference_id']:continue
            old,new=data['legacy'],data['attempts'];first=[x for x in new if x['first_attempt']];final=[x for x in new if x['final_attempt']]
            conditions.append(dict(mode=key[1],instruction_id=key[2],speech_generator_sha256=key[3],review_asr_sha256=key[4],
                legacy_final_audio=len(old),legacy_final_verdicts=Counter(x['verdict'] for x in old),
                legacy_final_audio_with_prior_attempts=sum(x['attempts']>1 for x in old),
                legacy_sampling_protocols=dict(Counter(x['sampling_protocol_id'] for x in old)),
                new_actual_attempts=len(new),new_first_attempts=len(first),new_first_attempt_verdicts=Counter(x['verdict'] for x in first),
                new_all_attempt_verdicts=Counter(x['verdict'] for x in new),new_final_audio=len(final),
                new_final_admitted=sum(x['final_admitted'] for x in final),
                new_sampling_observations=[dict(source_run=x['source_run'],arm=x['arm'],sample_id=x['sample_id'],
                    observation=x['final_sampling_observation'],receipt=x['final_sampling_receipt']) for x in final if x['final_sampling_observation']],
                examples=[{k:x[k] for k in ['source_run','sample_id','attempt','first_attempt','final_attempt','verdict','reason','target_text','review_asr_transcription']} for x in sorted(new,key=lambda x:(x['verdict']=='pass',x['source_run'],x['sample_id'],x['attempt']))[:3]],
                interpretation='Legacy records describe final mode only: no missing early-attempt failures are inferred. New first attempts, retry attempts and final admission stay separate. Different sampling receipts remain separate; error/pair counts are opportunities, not utility.'))
        cards.append(dict(reference_id=reference['reference_id'],voice_group=reference['voice_group'],conditions=conditions,
            unknown_unlisted_combinations=True,acoustic_traits=reference['acoustic_traits']))
    def condition(rid,mode):
        return next(c for r in cards if r['reference_id']==rid for c in r['conditions'] if c['mode']==mode and c['instruction_id']=='cantonese_zh')
    assert condition('voice009.ref1','zero_shot')['new_all_attempt_verdicts']=={'technical_failure':4}
    assert condition('voice042.ref1','zero_shot')['new_all_attempt_verdicts']=={'obvious_failure':4}
    assert condition('voice009.ref0','zero_shot')['new_first_attempt_verdicts']=={'pass':2}
    assert condition('voice042.ref0','zero_shot')['new_first_attempt_verdicts']=={'pass':2}
    capability=dict(verified_runtime=audit['actual_LM_inference'],audit_path=str(audit_path),audit_sha256=sha256_file(audit_path),
        instruct2='LM:target+instruction, no reference speech tokens; actualLMdoesnotreadembeddingargument. Flow:reference tokens/features and speaker embedding retained.',
        zero_shot='LM:target+instruction+reference text+reference speech tokens. Flow:reference tokens/features and speaker embedding retained.',
        caveat='Input paths are verified; clinical impairment transfer, word-level fidelity and DPO utility remain unproved.')
    context=dict(schema_version='reference_experience_cards_v2',content_version='mode_conditioned_v3',
        transport_compatibility='Existing generation_controller.materialize_control recognizes v2 transport and filters cards by reference_id; conditions are additive per-reference content. No frozen controller source edited.',
        cards=cards,generation_path_capability=capability,legacy_unique_final_audio=255,new_unique_final_audio=56,new_actual_attempts=64,
        references_with_observations=sum(bool(x['conditions']) for x in cards),
        scope='Only observed (reference,mode,instruction,SG,reviewASR) combinations support recipe-specific statements. A successful fallback belongs to its actualfinalmode. Unknown combination is unknown, not failed.',
        semantics='No globalqualityscore, no CER admission threshold, no phoneme/word audit. Preserve unchanged three-text coarseAgent/rulebook. These cards inform nextMASgeneration choices; realDPOutility requires matchedreal evaluation.',
        closed_training_proposals=['literal_matched_SG24_v1:主→指','contextual_matched_SG24_v2:主→指'],
        sources={str(f):sha256_file(f) for f in files})
    write_immutable_json(OUT/'MAS_context.json',context)
    write_immutable_bytes(OUT/'new_attempt_observations.jsonl',''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in attempts).encode())
    write_immutable_json(OUT/'receipt.json',dict(status='complete_mode_conditioned_reference_evidence',legacy_final_audio=255,new_final_audio=56,
        new_attempts=64,technical_failures=4,obvious_failures=4,first_and_fallback_modes_verified=True,
        source_sha256=context['sources'],output_sha256={str(f):sha256_file(f) for f in [OUT/'MAS_context.json',OUT/'new_attempt_observations.jsonl']}))
    print(json.dumps(dict(status='complete',references=len(cards),conditions=sum(len(x['conditions']) for x in cards),new_attempts=64,
        existing_controller_compatible=True,old_sources_unchanged=True)),flush=True)


if __name__=='__main__':main()
