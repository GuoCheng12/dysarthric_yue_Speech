#!/usr/bin/env python3
"""Compare mode/reference observations while preserving separate provenance."""
from collections import Counter
import html
import json
from pathlib import Path
import mas_mode_reference_probe_v1 as run
from therapist_harness.evidence import _stable_align
from therapist_harness.preference_data import build_preference_pairs


def main():
    p=run.base.verify();records={};sources=[Path(__file__),run.ROOT/'plan.json',run.ROOT/'MAS_decision.json']
    pictures=[]
    for mode in ['instruct2','zero_shot']:
        for arm in ['A','B']:
            out=run.ROOT/arm if mode=='zero_shot' else run.ROOT/'reused_instruct2'/arm
            if mode=='zero_shot':
                gen=run.base.read(out/'generation/generation_summary.json');outputs=gen['outputs'];sample=out/'sampling'
                receipt=run.base.read(sample/'sampling_receipt.json')
                assert receipt['status']=='complete'
                for key,name in [('candidates_sha256','candidates.jsonl'),('validated_inputs_sha256','validated_inputs.jsonl'),('greedy_sha256','greedy.jsonl')]:
                    assert receipt[key]==run.base.sha256_file(sample/name)
                sources += [out/'generation/generation_summary.json',sample/'sampling_receipt.json']
            else:
                outputs=run.base.read(out/'generation_outputs.json');sample=out
                reuse=run.base.read(out/'reuse_receipt.json');receipt=reuse['original_complete_receipt']
                assert reuse['actual_new_sampling'] is False and reuse['upstream_full_audio']==12 and reuse['selected_audio']==4
                assert all(run.base.sha256_file(Path(f))==h for f,h in {**reuse['source_sha256'],**reuse['subset_sha256']}.items())
                sources += [out/'generation_outputs.json',out/'reuse_receipt.json']
            inputs=run.base.rows(sample/'validated_inputs.jsonl');draws=run.base.rows(sample/'candidates.jsonl')
            greedy=run.base.rows(sample/'greedy.jsonl');pairs=run.base.rows(sample/'preferences.jsonl')
            assert receipt['current_asr_sha256']==p.initial_adapter.sha256
            if inputs:
                rebuilt,_=build_preference_pairs(inputs,draws,p.initial_adapter.sha256,expected_samples=4,greedy_candidates=greedy)
                assert rebuilt==pairs
            else:assert not draws and not greedy and not pairs
            assert all(run.base.sha256_file(Path(x['audio']))==x['audio_sha256'] for x in inputs)
            assert len(outputs)==4
            ids={x['sample_id'] for x in outputs};assert ids=={x['sample_id'] for x in run.base.rows(run.ROOT/arm/'manifest.jsonl')}
            for x in outputs:
                assert x['admitted']==any(v['audio_id']==x['sample_id'] for v in inputs)
                assert run.base.sha256_file(Path(x['output_wav']))==x['output_wav_sha256']
                first=x['attempt_records'][0]
                if first['generation'] is not None:assert first['generation']['generation_mode']==mode
                if mode=='instruct2':assert len(x['attempt_records'])==1 and x['generation_mode']=='instruct2'
                e=html.escape
                pictures.append(f'<article><h3>{mode}/{arm}: {e(x["sample_id"])}</h3><p>{e(x["target_text"])}</p><audio controls src="{e(x["output_wav"])}"></audio><p>Reference：{e(x["reference_text"])}</p><p>最终审核转写：{e(x["review"]["asr_transcription"])}</p><p>{e(x["review"]["verdict"])}：{e(x["review"]["reason"])}</p><p>最终模式：{e(x["generation_mode"])}；尝试{len(x["attempt_records"])}次。</p></article>')
            errors=Counter()
            by_id={x['audio_id']:x for x in inputs}
            for d in draws:
                _,events,_=_stable_align(run.base.text_normalize(by_id[d['audio_id']]['clean_gt']),run.base.text_normalize(d['prediction']))
                errors.update((x.operation,x.reference_unit,x.hypothesis_unit) for x in events)
            key=mode+'_'+arm
            records[key]=dict(outputs=outputs,inputs=inputs,draws=draws,greedy=greedy,receipt=receipt,
                summary=dict(requested_audio=4,admitted_audio=len(inputs),actual_draws=len(draws),actual_distinct_pairs=len(pairs),
                    first_attempt_verdicts=Counter(x['attempt_records'][0]['review']['verdict'] if x['attempt_records'][0]['review'] else 'technical_failure' for x in outputs),
                    final_modes=Counter(x['generation_mode'] for x in outputs),retry_audio=sum(len(x['attempt_records'])>1 for x in outputs),
                    first_attempt_technical_failures=sum(x['attempt_records'][0]['generation'] is None for x in outputs),
                    truncated_or_no_EOS=sum(d['reached_max_new_tokens'] or not d['ended_with_eos'] for d in draws),
                    common_error_events=[dict(operation=k[0],reference=k[1],hypothesis=k[2],count=v) for k,v in errors.most_common(12)]))
            sources += [sample/name for name in ['validated_inputs.jsonl','candidates.jsonl','greedy.jsonl','preferences.jsonl']]
    comparisons={}
    config_keys=['current_asr_sha256','base_model_sha256','samples_per_audio','seed','temperature','top_p','top_k','max_new_tokens','language','prompt','do_sample']
    for before,after in [('instruct2_A','zero_shot_A'),('instruct2_B','zero_shot_B'),('instruct2_A','instruct2_B'),('zero_shot_A','zero_shot_B')]:
        a,b=records[before],records[after]
        assert all(a['receipt'][k]==b['receipt'][k] for k in config_keys)
        aa={(d['audio_id'],d['sample_index']):d for d in a['draws']};bb={(d['audio_id'],d['sample_index']):d for d in b['draws']}
        common=sorted(aa.keys()&bb.keys())
        assert all(aa[k]['seed']==bb[k]['seed'] for k in common)
        ao={x['sample_id']:x for x in a['outputs']};bo={x['sample_id']:x for x in b['outputs']}
        assert ao.keys()==bo.keys()
        assert all(ao[k]['target_text']==bo[k]['target_text'] and ao[k]['assignment_seed']==bo[k]['assignment_seed'] for k in ao)
        if before.split('_')[-1]==after.split('_')[-1]:
            assert all(ao[k]['reference_wav_sha256']==bo[k]['reference_wav_sha256'] for k in ao)
        else:assert all(ao[k]['reference_wav_sha256']!=bo[k]['reference_wav_sha256'] for k in ao)
        direct=[k for k in common if len(ao[k[0]]['attempt_records'])==len(bo[k[0]]['attempt_records'])==1]
        comparisons[before+'_vs_'+after]=dict(paired_final_draws=len(common),
            changed_final_transcripts=sum(run.base.text_normalize(aa[k]['prediction'])!=run.base.text_normalize(bb[k]['prediction']) for k in common),
            both_first_attempt_kept_draws=len(direct),
            changed_transcripts_without_retry=sum(run.base.text_normalize(aa[k]['prediction'])!=run.base.text_normalize(bb[k]['prediction']) for k in direct),
            paired_records=[dict(audio_id=k[0],sample_index=k[1],before=aa[k]['prediction'],after=bb[k]['prediction'],
                                 both_no_retry=k in direct) for k in common])
    result=dict(status='complete_mode_reference_mechanism_probe',new_audio=8,reused_audio=8,
        summaries={k:v['summary'] for k,v in records.items()},comparisons=comparisons,
        ASR_training=False,closed_literal_target_reopened=False,selected_winner=None,
        interpretation='Only a pairedmode/reference mechanism diagnostic. Changing mode adds both reference text and reference speech tokens on the LM path; Flow remains reference-conditioned. Transcript changes or more errors/pairs do not establish patient impairment, target fidelity or useful DPO updates. First attempts and fallback finals separated. No winner selected by syntheticCER or errorvolume; narrow主->指target remainsclosed.',
        source_sha256={str(f):run.base.sha256_file(f) for f in sources})
    run.base.save(run.ROOT/'results.json',result)
    lines=['生成方式×reference：机制探针','', '| 条件 | 准入 | 首次粗审 | 需重试音频 | 最终实际模式 |', '|---|---:|---|---:|---|']
    for key,v in result['summaries'].items():lines.append(f'|{key}|{v["admitted_audio"]}/4|{dict(v["first_attempt_verdicts"])}|{v["retry_audio"]}|{dict(v["final_modes"])}|')
    lines+=['','配对ASR转写变化（规范化后）：']
    for key,v in comparisons.items():lines.append(f'- {key}: {v["changed_transcripts_without_retry"]}/{v["both_first_attempt_kept_draws"]}，两侧均无重试；最终交付{v["changed_final_transcripts"]}/{v["paired_final_draws"]}。')
    lines+=['',result['interpretation']]
    run.base.write_immutable_bytes(run.ROOT/'results.zh.md',('\n'.join(lines)+'\n').encode())
    run.base.write_immutable_bytes(run.ROOT/'listening.html',('<!doctype html><meta charset="utf-8"><title>MAS mode/reference</title><style>body{font:16px system-ui;max-width:1200px;margin:32px auto}main{display:grid;grid-template-columns:1fr 1fr;gap:12px}article{border:1px solid #ccc;padding:12px}</style><h1>instruct2 / zero_shot × reference</h1><p>仅8条新音频；另8条核验复用。未训练ASR；粗审不等于逐字保真。</p><main>'+''.join(pictures)+'</main>').encode())
    print(json.dumps(dict(summaries=result['summaries'],comparisons={k:{kk:vv for kk,vv in v.items() if kk!='paired_records'} for k,v in comparisons.items()}),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
