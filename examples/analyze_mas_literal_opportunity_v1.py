#!/usr/bin/env python3
"""Separate selector-wave evidence from held-out SG opportunities and training views."""
import argparse
from collections import Counter
import html
import json
from pathlib import Path

import mas_literal_opportunity_selector_v1 as run
from mas_contextual_recipe_allocation_v1 import signatures
from therapist_harness.preference_data import build_preference_pairs
from therapist_harness.retention_loop import training_pairs
from therapist_harness.dpo_training import preference_key

base, ROOT, PREVIOUS = run.base, run.ROOT, run.PREVIOUS
base.ROOT = ROOT
RECIPES = ['history','alternate']


def load_verified():
    p = base.verify(); data, sources = {}, []
    for recipe in RECIPES:
        out = ROOT / recipe; sampling = out / 'sampling'
        inputs, candidates, greedy, pairs = [base.rows(sampling/f'{n}.jsonl') for n in ['validated_inputs','candidates','greedy','preferences']]
        receipt = base.read(sampling/'sampling_receipt.json')
        assert receipt['status']=='complete' and receipt['current_asr_sha256']==p.initial_adapter.sha256
        for key, file in [('candidates_sha256','candidates.jsonl'),('validated_inputs_sha256','validated_inputs.jsonl'),('greedy_sha256','greedy.jsonl')]:
            assert base.sha256_file(sampling/file)==receipt[key]
        rebuilt, _ = build_preference_pairs(inputs,candidates,p.initial_adapter.sha256,expected_samples=4,greedy_candidates=greedy)
        assert pairs==rebuilt and all(base.sha256_file(Path(x['audio']))==x['audio_sha256'] for x in inputs)
        generation=base.read(out/'generation/generation_summary.json');assert len(generation['outputs'])==32
        metadata={x['sample_id']:x for x in base.read(out/'role_sidecar.json')['samples']}
        data[recipe]=dict(inputs=inputs,candidates=candidates,greedy=greedy,pairs=training_pairs(pairs),receipt=receipt,
            metadata=metadata,generation={x['sample_id']:x for x in generation['outputs']})
        sources += list(sampling.glob('*.json*')) + [out/'generation/generation_summary.json',out/'role_sidecar.json',out/'manifest.jsonl']
    config=['current_asr_sha256','base_model_sha256','samples_per_audio','seed','temperature','top_p','top_k','max_new_tokens','language','prompt']
    assert all(data['history']['receipt'][k]==data['alternate']['receipt'][k] for k in config)
    for sid, m in data['history']['metadata'].items():
        other=data['alternate']['metadata'][sid]
        assert all(m[k]==other[k] for k in ['speaker_id','wave','split','case_id','literal_carriers'])
        a,b=data['history']['generation'][sid],data['alternate']['generation'][sid]
        assert a['target_text']==b['target_text'] and a['assignment_seed']==b['assignment_seed']
        draws=[[x for x in data[r]['candidates'] if x['audio_id']==sid] for r in RECIPES]
        if a['admitted'] and b['admitted']: assert [x['seed'] for x in draws[0]]==[x['seed'] for x in draws[1]]
    return data, sources


def literal(pair, metadata):
    info=signatures(pair['chosen'],pair['rejected'])
    hits=[e for e in info['events'] if e['reference_unit']=='ng' and e['reference_character'] in metadata['literal_carriers']]
    return dict(component_ng_to_n=info['ng_to_n'],literal_ng_to_n=len(hits),literal_events=hits)


def select():
    data, sources=load_verified()
    views={}
    for recipe,d in data.items():
        selector_ids={sid for sid,m in d['metadata'].items() if m['split']=='selector'}
        view=[dict(pair=x,metadata=d['metadata'][x['audio_id']]) for x in d['pairs'] if x['audio_id'] in selector_ids]
        assert all(x['metadata']['wave']==0 for x in view)
        path=ROOT/f'selector_{recipe}_pair_view.jsonl';base.jsonl(path,view);sources.append(path)
        views[recipe]=view
    # Only wave0 pair views enter this policy. Full receipt hashes verify provenance, not held-out utility.
    choices=[]
    for v in run.VOICES:
        for i in range(2):
            case=f'litopp.{v}.t{i}'
            scores={r:sum(literal(x['pair'],x['metadata'])['literal_ng_to_n']>0 for x in views[r] if x['metadata']['case_id']==case) for r in RECIPES}
            chosen='alternate' if scores['alternate']>scores['history'] else 'history'
            choices.append(dict(case_id=case,voice_group=v,scores=scores,chosen_recipe=chosen))
    sources += [Path(__file__),Path(signatures.__code__.co_filename),ROOT/'plan.json',ROOT/'request_context.json']
    base.save(ROOT/'selection.json',dict(choices=choices,changed_cases=sum(x['chosen_recipe']=='alternate' for x in choices),
        actual_MAS_policy_source=str(PREVIOUS/'MAS_reflection/reflection.json'),
        rule='Predeclared MASpolicy: distinctactualwrongpairs withsamepatient realliteralreferencecharacter andconservativeng->n event onwave0; tiehistory. Onlyfilteredselectorpairviews enterpolicy; waves1-3notusedinselection.',
        source_sha256={str(f):base.sha256_file(f) for f in sources}))
    print(json.dumps(choices,ensure_ascii=False),flush=True)


def report():
    selection=base.read(ROOT/'selection.json')
    assert all(base.sha256_file(Path(f))==h for f,h in selection['source_sha256'].items())
    data,sources=load_verified();source_context=base.read(ROOT/'request_context.json')
    exact={x['voice_group']:{(e['event']['reference_character'],e['event']['hypothesis_character']) for e in x['actual_events']} for x in source_context['patients']}
    cases={};listen=[]
    for recipe,d in data.items():
        cases[recipe]={}
        for sid,m in d['metadata'].items():
            output=d['generation'][sid];pairs=[x for x in d['pairs'] if x['audio_id']==sid]
            draws=[x for x in d['candidates'] if x['audio_id']==sid]
            infos=[dict(key=preference_key(x),chosen=x['chosen'],rejected=x['rejected'],**literal(x,m)) for x in pairs]
            draw_info=[literal(dict(chosen=output['target_text'],rejected=x['prediction']),m) for x in draws]
            case=dict(sample_id=sid,metadata=m,target=output['target_text'],admitted=output['admitted'],pairs=infos,
                literal_pairs=sum(x['literal_ng_to_n']>0 for x in infos),
                literal_draws=sum(x['literal_ng_to_n']>0 for x in draw_info),
                exact_original_glyph_pair_count=sum((e['reference_character'],e['hypothesis_character']) in exact[m['voice_group']] for x in infos for e in x['literal_events']),
                actual_final_mode=output['generation_mode'],first_attempt_mode=output['attempt_records'][0]['mode'],
                attempts=len(output['attempt_records']),review_verdict=output['review']['verdict'])
            cases[recipe][sid]=case
            e=html.escape
            listen.append(f'<article><h3>{recipe}: {e(sid)} ({m["split"]})</h3><p>{e(output["target_text"])}</p><audio controls src="{e(output["output_wav"])}"></audio><p>Reference: {e(output["reference_text"])}</p><p>粗审：{e(output["review"]["asr_transcription"])}</p><p>候选：{e(" / ".join(x["prediction"] for x in draws))}</p><p>真实错误字上的ng→n偏好对：{case["literal_pairs"]}</p></article>')
    choices={x['case_id']:x['chosen_recipe'] for x in selection['choices']}
    shared_meta={x['sample_id']:x for x in base.read(PREVIOUS/'MAS/role_sidecar.json')['samples'] if x['role'] in ['protection','maintenance']}
    shared_sampling=PREVIOUS/'MAS/sampling';shared_receipt=base.read(shared_sampling/'sampling_receipt.json')
    assert shared_receipt['current_asr_sha256']==data['history']['receipt']['current_asr_sha256']
    old_inputs,old_candidates,old_greedy=[base.rows(shared_sampling/f'{x}.jsonl') for x in ['validated_inputs','candidates','greedy']]
    for key,file in [('candidates_sha256','candidates.jsonl'),('validated_inputs_sha256','validated_inputs.jsonl'),('greedy_sha256','greedy.jsonl')]:
        assert shared_receipt[key]==base.sha256_file(shared_sampling/file)
    old_pairs,_=build_preference_pairs(old_inputs,old_candidates,shared_receipt['current_asr_sha256'],expected_samples=4,greedy_candidates=old_greedy)
    assert training_pairs(old_pairs)==base.rows(PREVIOUS/'MAS/training_preferences.jsonl')
    shared_pairs=[x for x in training_pairs(old_pairs) if x['audio_id'] in shared_meta]
    assert all(base.sha256_file(Path(x['audio']))==x['audio_sha256'] for x in shared_pairs)
    sources += [shared_sampling/f'{x}.jsonl' for x in ['validated_inputs','candidates','greedy','preferences']]
    assert len(shared_meta)==8 and {x['audio_id'] for x in shared_pairs}==set(shared_meta)
    arms={};effective_keys={}
    for mode in ['MAS','fixed']:
        chosen_cases=[];target_pairs=[]
        for sid,m in data['history']['metadata'].items():
            if m['split']!='training':continue
            recipe=choices[m['case_id']] if mode=='MAS' else 'history'
            case=cases[recipe][sid];chosen_cases.append(dict(case,selected_recipe=recipe))
            target_pairs += [x for x in data[recipe]['pairs'] if x['audio_id']==sid]
        aggregate=dict(target_audio=len(chosen_cases),admitted=sum(x['admitted'] for x in chosen_cases),
            paired_audio=sum(bool(x['pairs']) for x in chosen_cases),pairs=sum(len(x['pairs']) for x in chosen_cases),
            literal_audio=sum(x['literal_pairs']>0 for x in chosen_cases),literal_pairs=sum(x['literal_pairs'] for x in chosen_cases),
            literal_draws=sum(x['literal_draws'] for x in chosen_cases))
        effective_keys[mode]={preference_key(x) for x in target_pairs}
        out=ROOT/mode;out.mkdir(exist_ok=False)
        base.jsonl(out/'training_preferences.jsonl',target_pairs+shared_pairs)
        base.save(out/'role_sidecar.json',dict(samples=[x['metadata'] for x in chosen_cases]+list(shared_meta.values())))
        arms[mode]=dict(label='opportunity' if mode=='MAS' else 'history',aggregate=aggregate,cases=chosen_cases,
            shared_maintenance_protection_audio=list(shared_meta))
        sources += [out/'training_preferences.jsonl',out/'role_sidecar.json']
    enough=all(x['aggregate']['paired_audio']>=3 for x in arms.values())
    literal_present=any(x['aggregate']['literal_audio'] for x in arms.values())
    distinct=effective_keys['MAS']!=effective_keys['fixed']
    # A collapsed selector cannot support a two-arm claim; a single new literal-data DPO can still test data utility.
    train_modes=['MAS','fixed'] if enough and literal_present and distinct else ['fixed'] if enough and literal_present else []
    sources += [Path(__file__),ROOT/'selection.json',PREVIOUS/'MAS/training_preferences.jsonl',PREVIOUS/'MAS/role_sidecar.json',
        PREVIOUS/'MAS/sampling/sampling_receipt.json',PREVIOUS/'results.json']
    base.save(ROOT/'delivery_report.json',dict(arms=arms,raw_recipe_cases=cases,
        selector=selection['choices'],changed_selector_cases=selection['changed_cases'],
        different_effective_pair_sets=distinct,train_modes=train_modes,
        opportunity_more_heldout_literal_audio=arms['MAS']['aggregate']['literal_audio']>arms['fixed']['aggregate']['literal_audio'],
        training_scope='Internalmode MAS meansopportunity; fixed meanshistory, not originalref0. Shared8priorMASprotection/maintenanceaudio haveoriginalfullsamplingreceipt retained; new64rawreceiptssplit into explicit views. Selectorwaveforms areexcluded from DPO.',
        decision='Proceed matchedDPO' if len(train_modes)==2 else 'Selectorcollapsed; single newliteraldataDPO versusoldRcontrol only' if train_modes else 'Close this selector experiment: insufficient actual literal training opportunities; no fabricatednegative or redundantDPO.',
        source_sha256={str(f):base.sha256_file(f) for f in sources}))
    base.write_immutable_bytes(ROOT/'listening.html',('<!doctype html><meta charset="utf-8"><title>Literal recipe opportunity</title><style>body{font:16px system-ui;max-width:1200px;margin:32px auto}main{display:grid;grid-template-columns:1fr 1fr;gap:12px}article{border:1px solid #ccc;padding:12px}</style><h1>独立音频上的患者错误字与reference选择</h1><p>wave0选配方；wave1–3供训练。三文本粗审不证明逐字保真。</p><main>'+''.join(listen)+'</main>').encode())
    print(json.dumps(dict(arms={a:x['aggregate'] for a,x in arms.items()},changed_selector_cases=selection['changed_cases'],train_modes=train_modes),ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['select','report'])
    args=parser.parse_args();globals()[args.stage]()
