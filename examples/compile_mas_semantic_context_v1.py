#!/usr/bin/env python3
"""Rebuild actual pairs, report all content outcomes, compile matched DPO slots."""
from collections import Counter
import html
import json
from pathlib import Path
import mas_semantic_context_probe_v1 as run
from mas_contextual_recipe_allocation_v1 import signatures
from therapist_harness.preference_data import build_preference_pairs
from therapist_harness.retention_loop import training_pairs
from therapist_harness.dpo_training import preference_key

base,ROOT=run.base,run.ROOT
base.ROOT=ROOT
OLD=Path('/tmp/mas_deterministic_matched_DPO_v1')
ANCHORS=run.previous.PREVIOUS/'MAS'


def verified_sampling(path,parent):
    inputs,candidates,greedy,pairs=[base.rows(path/f'{n}.jsonl') for n in ['validated_inputs','candidates','greedy','preferences']]
    receipt=base.read(path/'sampling_receipt.json')
    assert receipt['status']=='complete' and receipt['current_asr_sha256']==parent
    for key,file in [('candidates_sha256','candidates.jsonl'),('validated_inputs_sha256','validated_inputs.jsonl'),('greedy_sha256','greedy.jsonl')]:
        assert base.sha256_file(path/file)==receipt[key]
    rebuilt,_=build_preference_pairs(inputs,candidates,parent,expected_samples=4,greedy_candidates=greedy)
    assert rebuilt==pairs and all(base.sha256_file(Path(x['audio']))==x['audio_sha256'] for x in inputs)
    return dict(inputs=inputs,candidates=candidates,greedy=greedy,pairs=training_pairs(pairs),receipt=receipt)


def main():
    p=base.verify();data={};sources=[Path(__file__),ROOT/'proposal.json',ROOT/'plan.json',ROOT/'review_policy_smoke/result.json']
    assert base.read(ROOT/'review_policy_smoke/result.json')['passed']
    arm_summary={};listening=[]
    for arm in ['natural','odd']:
        path=ROOT/arm;d=verified_sampling(path/'sampling',p.initial_adapter.sha256)
        d['metadata']={x['sample_id']:x for x in base.read(path/'role_sidecar.json')['samples']}
        outputs=base.read(path/'generation/generation_summary.json')['outputs'];assert len(outputs)==24
        d['outputs']={x['sample_id']:x for x in outputs}
        d['by_id']={sid:[x for x in d['pairs'] if x['audio_id']==sid] for sid in d['metadata']}
        d['cases']=[]
        for sid,m in d['metadata'].items():
            output=d['outputs'][sid];pairs=d['by_id'][sid]
            assert output['review']['prompt_sha256']==base.sha256_file(run.PROMPT)
            infos=[]
            for pair in pairs:
                sig=signatures(pair['chosen'],pair['rejected'])
                hits=[e for e in sig['events'] if e['reference_unit']=='ng' and e['reference_character'] in m['literal_carriers']]
                infos.append(dict(chosen=pair['chosen'],rejected=pair['rejected'],literal_ng_to_n=len(hits),events=hits))
            draws=[x for x in d['candidates'] if x['audio_id']==sid]
            case=dict(sample_id=sid,metadata=m,target=output['target_text'],admitted=output['admitted'],
                pairs=len(pairs),literal_pairs=sum(x['literal_ng_to_n']>0 for x in infos),actual_pairs=infos,
                attempts=len(output['attempt_records']),final_mode=output['generation_mode'],review=output['review']['verdict'],
                sampled_transcripts=[x['prediction'] for x in draws])
            d['cases'].append(case)
            e=html.escape
            listening.append(f'<article><h3>{arm}: {e(sid)}</h3><p>{e(output["target_text"])}</p><audio controls src="{e(output["output_wav"])}"></audio><p>reference: {e(output["reference_text"])}</p><p>粗审: {e(output["review"]["asr_transcription"])}</p><p>审查: {e(output["review"]["reason"])}</p><p>实际候选: {e(" / ".join(case["sampled_transcripts"]))}</p></article>')
        arm_summary[arm]=dict(audio=24,admitted=sum(x['admitted'] for x in d['cases']),
            paired_audio=sum(x['pairs']>0 for x in d['cases']),pairs=len(d['pairs']),
            literal_audio=sum(x['literal_pairs']>0 for x in d['cases']),literal_pairs=sum(x['literal_pairs'] for x in d['cases']),
            retries=sum(x['attempts']-1 for x in d['cases']))
        data[arm]=d;sources+=list((path/'sampling').glob('*.json*'))+[path/'role_sidecar.json',path/'generation/generation_summary.json']
    keys=['current_asr_sha256','base_model_sha256','samples_per_audio','seed','temperature','top_p','top_k','max_new_tokens','language','prompt']
    assert all(data['natural']['receipt'][k]==data['odd']['receipt'][k] for k in keys)
    for sid,m in data['natural']['metadata'].items():
        n=data['odd']['metadata'][sid]
        assert all(m[k]==n[k] for k in ['case_id','wave','voice_group','speaker_id','reference_id'])
        a,b=[data[arm]['outputs'][sid] for arm in ['natural','odd']]
        assert Counter(base.text_normalize(a['target_text']))==Counter(base.text_normalize(b['target_text']))
        assert all(a[k]==b[k] for k in ['assignment_seed','reference_wav_sha256'])
        if a['admitted'] and b['admitted']:
            assert [x['seed'] for x in data['natural']['candidates'] if x['audio_id']==sid]==[x['seed'] for x in data['odd']['candidates'] if x['audio_id']==sid]
    common=[sid for sid in data['natural']['metadata'] if all(data[a]['by_id'][sid] for a in ['natural','odd'])]
    report=dict(arms=arm_summary,cases={a:d['cases'] for a,d in data.items()},common_eligible_slots=common,
        unmatched_eligible={a:[sid for sid,pairs in d['by_id'].items() if pairs and sid not in common] for a,d in data.items()},
        interpretation='Whole48audio outcomes are reported before matched training selection. Common eligibility is a conditional content comparison; extra negatives on one side do not prove benefit. Text-derived literal events do not verify pronunciation. Intended semantic oddness is allowed by identical new coarse prompt.',
        train_modes=['MAS','fixed'] if len(common)>=3 else [],source_sha256={str(f):base.sha256_file(f) for f in sources})
    base.save(ROOT/'delivery_report.json',report)
    base.write_immutable_bytes(ROOT/'listening.html',('<!doctype html><meta charset="utf-8"><title>自然与古怪句对照</title><style>body{font:16px system-ui;max-width:1200px;margin:32px auto}main{display:grid;grid-template-columns:1fr 1fr;gap:12px}article{border:1px solid #ccc;padding:12px}</style><h1>同用字的自然／古怪句</h1><p>48条新音频；同reference、模式、seed和审查提示。粗审通过不等于逐字保真。</p><main>'+''.join(listening)+'</main>').encode())
    if len(common)<3:
        print(json.dumps(dict(summary=arm_summary,common_slots=len(common),decision='Insufficient matched actual error slots; no forced negatives.'),ensure_ascii=False));return
    shared=verified_sampling(ANCHORS/'sampling',p.initial_adapter.sha256)
    meta={x['sample_id']:x for x in base.read(ANCHORS/'role_sidecar.json')['samples'] if x['role'] in ['protection','maintenance']}
    shared_pairs=[x for x in shared['pairs'] if x['audio_id'] in meta]
    assert len(meta)==8 and {x['audio_id'] for x in shared_pairs}==set(meta)
    assert shared['pairs']==base.rows(ANCHORS/'training_preferences.jsonl')
    sources+=list((ANCHORS/'sampling').glob('*.json*'))+[ANCHORS/'role_sidecar.json',ANCHORS/'training_preferences.jsonl']
    original=base.read(OLD/'plan/plan.json');real=base.read(OLD/'plan/expected_DPO_exposures.json')['R_DPO_KL']
    kl=base.read(OLD/'plan/expected_common_KL_exposures.json')
    # Prospective uniform matched allocation, not an ASR-error-severity selector.
    target_order=[common[i%len(common)] for i in range(24)]
    protection=sorted(k for k,v in meta.items() if v['role']=='protection')
    maintenance=sorted(k for k,v in meta.items() if v['role']=='maintenance')
    planned,selections={},{}
    for mode,arm in [('MAS','odd'),('fixed','natural')]:
        out=ROOT/mode;out.mkdir(exist_ok=False)
        target_pairs=[x for x in data[arm]['pairs'] if x['audio_id'] in common]
        prefs=target_pairs+shared_pairs
        base.jsonl(out/'training_preferences.jsonl',prefs)
        base.save(out/'role_sidecar.json',dict(samples=[data[arm]['metadata'][sid] for sid in common]+list(meta.values())))
        sources+=[out/'training_preferences.jsonl',out/'role_sidecar.json']
        by_id={sid:[x for x in prefs if x['audio_id']==sid] for sid in common+list(meta)}
        selections[mode]=[];planned[mode]=[]
        for step in range(1,5):
            ids=target_order[(step-1)*6:step*6]+[protection[step-1],maintenance[step-1]]
            selections[mode]+=ids
            synthetic=[dict(audio_id=sid,source='synthetic',objective='dpo',step=step,occurrence=(step-1)*8+i,
                preference_key=preference_key(pair),loss_weight=.25/8/len(by_id[sid])) for i,sid in enumerate(ids) for pair in by_id[sid]]
            assert abs(sum(x['loss_weight'] for x in synthetic)-.25)<1e-12
            planned[mode].append(real[step-1]+synthetic)
    assert selections['MAS']==selections['fixed']
    sources += [ROOT/'delivery_report.json',base.REPO/'examples/train_mas_semantic_context_v1.py',base.REPO/'examples/train_mas_contextual_recipe_dpo_v1.py',
        OLD/'plan/plan.json',OLD/'plan/expected_DPO_exposures.json',OLD/'plan/expected_common_KL_exposures.json']
    hashes={**base.read(ROOT/'sources.json'),**{str(f):base.sha256_file(f) for f in sources}}
    real_receipt=original['source_sampling_receipts']['historical_real'];receipts={'historical_real':real_receipt}
    for f,h in original['source_sha256'].items():
        if '/real96_dpo_probe/sampling/' in f or f==original['data_paths']['real_KL']:
            assert base.sha256_file(Path(f))==h;hashes[f]=h
    for name,path in [('new_natural',ROOT/'natural/sampling/sampling_receipt.json'),('new_odd',ROOT/'odd/sampling/sampling_receipt.json'),('shared_anchors',ANCHORS/'sampling/sampling_receipt.json')]:
        receipt=base.read(path)
        assert all(receipt[k]==real_receipt['decode'][k] for k in keys if k!='max_new_tokens')
        receipts[name]=dict(path=str(path),sha256=base.sha256_file(path),decode=receipt,
            use='Original full receipt retained; matched audio subset or shared anchors used explicitly. Not a fabricated merged sampling receipt.')
    out=ROOT/'training_plan';out.mkdir(exist_ok=False)
    base.save(out/'plan.json',dict(parent=original['parent'],source_sha256=hashes,
        data_paths=dict(real_KL=original['data_paths']['real_KL'],real_preferences=original['data_paths']['real_preferences'],
            synthetic_preferences={m:str(ROOT/m/'training_preferences.jsonl') for m in ['MAS','fixed']}),
        source_sampling_receipts=receipts,selections=selections,settings=original['settings'],gate=original['gate'],
        train_modes=['MAS','fixed'],mode_labels=dict(MAS='odd weak-semantic context',fixed='natural coherent context'),
        allocation='Prospective same matched slot order, round-robin24targetvisits plus8sharedanchors. Same real32/91pairs, KL256, foursteps. Different numbers of distinct wrong transcripts are averaged per audio, not counted as extra audio exposure.',
        semantic_review='Both new content arms use the versioned oddness-aware prompt. Shared historical anchors retain their original standard coarse review receipts.',
        sampling_caps='Actual real256 versus SG128 retained, separately recorded; model and remaining decoding parameters match.'))
    base.save(out/'expected_DPO_exposures.json',planned);base.save(out/'expected_common_KL_exposures.json',kl)
    base.verify();print(json.dumps(dict(summary=arm_summary,common_slots=len(common),training_target_visits=Counter(target_order)),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
