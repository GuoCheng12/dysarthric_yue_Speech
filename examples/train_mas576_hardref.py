"""Retain the MAS576 content/patient assignments, replace references, and fit ASR."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

import train_real2117_hardtts576 as full
import tts_reference_policy as reference
from resume_tts_reference_policy import install_empty_token_guard
from tts_increment_control import (
    ROOT as SOURCE, REPO, SEEDS, TEST, read, rows, jsonl, sha256_file,
    training_command, write_immutable_json as save, write_immutable_bytes,
)

ROOT = SOURCE.parent / "real2117_mas576_hardref_v1"
OLD = SOURCE.parent / "real2117_mas576_fresh_lora_b8g4_v1"
ARMS = ("mas_old_reference", "mas_hard_reference")
GPUS = ("GPU-db50b43c-0d00-f586-a012-6676287d4dc1", "GPU-48fe1998-ad91-552c-bc6f-5f4e879c192d")
BAD_REFERENCE = "vlink228/6453001756197404615719"
GROUPS = ("overall", "vlink247", "other_patients", "easy", "medium", "hard")


def select_alternatives(candidates, old_row):
    """Prefer the old reference's duration bin; rank distinct alternatives by CER."""
    eligible = [r for r in candidates if not reference.same_text(r, old_row)]
    if not eligible:
        raise ValueError(f"No alternate train reference: {old_row['speaker_id']}")
    bucket = reference.duration_bin(old_row["seconds"])
    available = {r["duration_bin"] for r in eligible}
    chosen_bin = min(available, key=lambda b: (abs(b - bucket), b))
    ranked = sorted((r for r in eligible if r["duration_bin"] == chosen_bin),
                    key=lambda r: (-float(r["zero_shot_cer"]), r["utt_id"]))
    selected = []
    for row in ranked:
        if not any(reference.same_text(row, previous) for previous in selected):
            selected.append(row)
        if len(selected) == 3:
            break
    return selected


def prepare(root):
    reference.verify(reference.ROOT)
    full.verify(full.ROOT)
    assert not root.exists(), root
    old_cfg = read(OLD / "run_config.json")
    old_manifest = rows(OLD / "train.jsonl")
    old_index = read(OLD / "training_index.json")[2117:]
    assert sha256_file(OLD / "train.jsonl") == old_cfg["train_sha256"]
    assert len(old_index) == 576 and len({x['text'] for x in old_index}) == 36
    assert set(Counter(x['text'] for x in old_index).values()) == {16}
    old_records = {}
    inputs = {OLD / "run_config.json", OLD / "train.jsonl", OLD / "training_index.json",
              SOURCE / "real_q/train.jsonl", TEST,
              Path('/data/qwen3-asr/records/cosyvoice3_reference_sft_setting_d_v1/augmentation/speech_generator_highinfo60_2h_v1/plan/README.md')}
    expected_runtime = None
    for number in range(1, 7):
        path = Path(old_cfg['source_run']) / f'round_{number:03d}/tts_outputs.json'
        receipt = read(path)
        assert receipt['status'] == 'complete' and receipt['output_count'] == 96
        inputs.add(path)
        for record in receipt['outputs']:
            assert record['sample_id'] not in old_records
            old_records[record['sample_id']] = record
        runtime = {k: receipt['runtime'][k] for k in ('base_llm_sha256', 'speech_generator_sha256', 'lora', 'inference')}
        if expected_runtime is None:
            expected_runtime = runtime
        assert runtime == expected_runtime
    train_path = Path('/data/qwen3-asr/finetune/data_setting_d_v1/setting_d_train.jsonl')
    train = rows(train_path)
    inputs.add(train_path)
    by_audio = {r['audio']: r for r in train}
    patients = {r['speaker_id'] for r in old_index}
    assert len(patients) == 51
    candidates = defaultdict(list)
    old_by_patient = {}
    for row in old_index:
        old_by_patient.setdefault(row['speaker_id'], row['reference_audio'])
        assert old_by_patient[row['speaker_id']] == row['reference_audio']
    for row in train:
        if row['speaker_id'] not in patients or row['utt_id'] == BAD_REFERENCE:
            continue
        info = reference.audio_features(row['audio'])
        if info['rejection'] is None and 3 <= info['seconds'] <= 12:
            candidates[row['speaker_id']].append({**row, **info, 'duration_bin': reference.duration_bin(info['seconds'])})
    pools, audit = {}, {}
    for patient in sorted(patients):
        old_row = by_audio[old_by_patient[patient]]
        old_row = {**old_row, **reference.audio_features(old_row['audio'])}
        assert 3 <= old_row['seconds'] <= 12
        pools[patient] = select_alternatives(candidates[patient], old_row)
        audit[patient] = {'old_reference': old_row['utt_id'], 'old_duration_bin': reference.duration_bin(old_row['seconds']),
            'selected': [reference.reference(r) for r in pools[patient]],
            'new_duration_bins': [r['duration_bin'] for r in pools[patient]],
            'candidate_rows': len(candidates[patient])}
    allocation = {}
    rng = random.Random(20260907)
    for patient in sorted(patients):
        samples = [r for r in old_index if r['speaker_id'] == patient]
        rng.shuffle(samples)
        for i, row in enumerate(samples):
            allocation[row['sample_id']] = reference.reference(pools[patient][i % len(pools[patient])])
    from therapist_harness.text import identity_keys
    blocked = {item for text in {r['clean_gt'] for r in train + rows(TEST)}
               for item in identity_keys(text).items()}
    target_identities = {text: identity_keys(text) for text in {r['text'] for r in old_index}}
    pairs = []
    for i, row in enumerate(old_index):
        old = old_records[row['sample_id']]
        assert old['target_text'] == row['text'] and old['output_wav'] == row['audio']
        assert old_manifest[2117+i]['audio'] == old['output_wav']
        assert old_manifest[2117+i]['text'] == 'language Cantonese<asr_text>' + old['target_text']
        assert sha256_file(Path(old['output_wav'])) == old['output_wav_sha256']
        assert sha256_file(Path(old['reference_wav'])) == old['reference_wav_sha256']
        assert not blocked.intersection(target_identities[old['target_text']].items())
        new_ref = allocation[row['sample_id']]
        assert new_ref['speaker_id'] == row['speaker_id']
        assert new_ref['reference_wav'] != old['reference_wav']
        inputs.update([Path(old['output_wav']), Path(old['reference_wav']), Path(new_ref['reference_wav'])])
        pairs.append({'sample_id': f'mas{i:04d}', 'source_sample_id': row['sample_id'],
            'speaker_id': row['speaker_id'], 'target_text': old['target_text'], 'seed': old['seed'],
            'old_record': old, 'new_reference': new_ref})
    root.mkdir(parents=True)
    save(root/'pairs.json', pairs)
    save(root/'reference_audit.json', audit)
    source_paths = [Path(__file__), Path(full.__file__), Path(reference.__file__),
                    REPO/'examples/resume_tts_reference_policy.py', REPO/'examples/report_tts_increment_control.py',
                    REPO/'src/therapist_harness/evidence.py', REPO/'src/therapist_harness/text.py']
    baseline = read(SOURCE/'protocol.json')
    commands = {a: {str(s): training_command(root,a,s) for s in SEEDS} for a in ARMS}
    for arm in ARMS:
        for seed in SEEDS:
            old_command = read(SOURCE/f'real_q/seed_{seed}/execution.json')['command']
            old_opts = dict(zip(old_command[2::2],old_command[3::2]))
            new_command = commands[arm][str(seed)]
            new_opts = dict(zip(new_command[2::2],new_command[3::2]))
            assert old_command[:2] == new_command[:2]
            assert {k for k in new_opts if old_opts[k] != new_opts[k]} == {'--train_file','--output_dir'}
    for label, path in locations(root).items():
        if label in ARMS:
            continue
        for seed in SEEDS:
            inputs.update(path/f'seed_{seed}'/file for file in ('result.json','evaluation/merged_step_50/predictions.jsonl'))
    save(root/'protocol.json', {'created_utc': datetime.now(timezone.utc).isoformat(),
        'real_rows':2117,'synthetic_rows':576,'total_rows':2693,'texts':36,'patients':51,
        'arms':ARMS,'seeds':SEEDS,'commands':commands,'expected_tts_runtime':expected_runtime,
        'reference_policy':'Full real train candidates, qualified mono3-12s; exclude old reference text and known contaminated vlink228 currency clip; use old duration bin or nearest available bin; highest-CER up to3 distinct-text references per patient; seeded balanced rotation.',
        'content_policy':'Exactly preserve original MAS576 text/patient/row/seed mapping. No new Agent generation.',
        'quality_gate':'Unchanged technical waveform checks, duration0.25-30s; no ASR-CER filter; completed-empty-token guard; at most10 deterministic attempts, seed=old_seed+10000*attempt.',
        'training':'New Qwen3-ASR LoRA from base each seed, same batch8x4 lr2e-4 r16 alpha4 dropout.05; unchanged140-step scheduler horizon; stop and evaluate at50.',
        'comparison_limits':['Same real order and same MAS patient/text assignments between reference arms.',
            'Compared with Q-reconstruction576, both content and patient allocation change (vlink247122 to6).',
            'MAS content was previously generated using feedback from the reused511; exploratory development evaluation.',
            'Fixed50 optimizer updates, not fixed real exposure; references are fixed across ASR seeds.'],
        'source_sha256':{**baseline['source_sha256'],**{str(p):sha256_file(p) for p in source_paths}},
        'input_sha256':{str(p):sha256_file(p) for p in sorted(inputs)},
        'artifact_sha256':{str(root/n):sha256_file(root/n) for n in ('pairs.json','reference_audit.json')}})
    full.status(root,'prepared',stage='prepare')
    print(json.dumps({'prepared':576,'patients':51,'texts':36,'changed_references':576,
        'unique_references':len({p['new_reference']['utt_id'] for p in pairs}),
        'patients_with_duration_bin_change':[p for p,a in audit.items() if set(a['new_duration_bins'])!={a['old_duration_bin']}],
        'vlink247_rows':sum(p['speaker_id']=='vlink247' for p in pairs)},ensure_ascii=False),flush=True)


def verify(root, arm=None):
    reference.verify_sources()
    cfg=read(root/'protocol.json')
    for key in ('source_sha256','input_sha256','artifact_sha256'):
        for path,digest in cfg[key].items():
            assert sha256_file(Path(path))==digest,path
    if arm is not None:
        assembly=read(root/'assembly.json')
        assert sha256_file(root/arm/'train.jsonl')==assembly['train_sha256'][arm]
        assert len(rows(root/arm/'train.jsonl'))==2693
        return {**cfg,'commands':cfg['commands'][arm],'train_sha256':assembly['train_sha256'][arm]}
    return cfg


def validate(pair,item):
    assert item['pair']==pair and 0<=item['attempt']<10
    record=item['record']
    assert record['sample_id']==pair['sample_id'] and record['target_text']==pair['target_text']
    assert record['seed']==pair['seed']+10000*item['attempt']
    for key in ('reference_text','reference_wav','reference_wav_sha256'):
        assert record[key]==pair['new_reference'][key]
    assert sha256_file(Path(record['output_wav']))==record['output_wav_sha256']


def synthesize(root,worker):
    cfg=verify(root)
    from synthesis.cosyvoice3_patient_sft import synthesize_manifest as tts
    model,metadata=tts.load_fixed_runtime()
    for k,v in cfg['expected_tts_runtime'].items():
        assert metadata[k]==v,(k,metadata[k],v)
    install_empty_token_guard(model.model)
    save(root/f'tts_worker_{worker}_runtime_{time.time_ns()}.json',metadata)
    pairs=read(root/'pairs.json')[worker::2]
    for index,pair in enumerate(pairs,1):
        selected=root/'selected'/f"{pair['sample_id']}.json"
        if selected.exists():
            validate(pair,read(selected));continue
        for attempt in range(10):
            folder=root/'synthesis'/pair['sample_id']/f'attempt_{attempt}'
            receipt=folder/'receipt.json'
            if receipt.exists():
                outcome=read(receipt)
            else:
                folder.mkdir(parents=True,exist_ok=True)
                ref=pair['new_reference']
                sample=tts.ManifestSample(sample_id=pair['sample_id'],target_text=pair['target_text'],
                    reference_text=ref['reference_text'],reference_wav=Path(ref['reference_wav']),
                    reference_wav_sha256=ref['reference_wav_sha256'],output_wav=folder/'hard_reference.wav',
                    seed=pair['seed']+10000*attempt)
                assert not sample.output_wav.exists(),f'Incomplete attempt requires inspection: {folder}'
                try:
                    record,temporary=tts._stage_audio_once(sample,model)
                    try: os.link(temporary,sample.output_wav)
                    finally: temporary.unlink()
                    outcome={'attempt':attempt,'record':record,'error':None}
                except RuntimeError as error:
                    if not str(error).startswith('Speech Generator returned '):raise
                    outcome={'attempt':attempt,'error':str(error)}
                save(receipt,outcome)
            if outcome['error'] is None:
                item={'pair':pair,'attempt':attempt,'record':outcome['record']}
                validate(pair,item);save(selected,item);break
        else:raise RuntimeError(f"Ten technical attempts exhausted: {pair['sample_id']}")
        print(f"[synthesis] worker={worker} {index}/{len(pairs)} {pair['sample_id']} attempt={attempt}",flush=True)


def assemble(root):
    verify(root)
    pairs=read(root/'pairs.json')
    selected=[read(root/'selected'/f"{p['sample_id']}.json") for p in pairs]
    for p,s in zip(pairs,selected):validate(p,s)
    real=rows(SOURCE/'real_q/train.jsonl')
    old=rows(OLD/'train.jsonl')[2117:]
    new=[{**r,'audio':s['record']['output_wav']} for r,s in zip(old,selected)]
    for arm,extra in zip(ARMS,(old,new)):jsonl(root/arm/'train.jsonl',real+extra)
    assert [r['text'] for r in old]==[r['text'] for r in new]
    save(root/'assembly.json',{'real_rows':2117,'synthetic_rows':576,
        'train_sha256':{a:sha256_file(root/a/'train.jsonl') for a in ARMS},
        'selected_sha256':{str(root/'selected'/f"{p['sample_id']}.json"):sha256_file(root/'selected'/f"{p['sample_id']}.json") for p in pairs},
        'attempt_histogram':dict(Counter(s['attempt'] for s in selected)),
        'synthetic_seconds':sum(s['record']['duration_seconds'] for s in selected),
        'new_clips_under_one_second':sum(s['record']['duration_seconds']<1 for s in selected)})


def train(root,arm,seed):
    verify(root,arm)
    for path,digest in read(root/'assembly.json')['selected_sha256'].items():
        assert sha256_file(Path(path))==digest
        item=read(Path(path));validate(item['pair'],item)
    full.ARM=arm
    full.verify=lambda r:verify(r,arm)
    full.train(root,seed)


def locations(root):
    return {'real2117':SOURCE/'real_q',
        'real2117_q_hardtts576':full.ROOT/full.ARM,
        **{a:root/a for a in ARMS}}


def report(root):
    verify(root)
    from therapist_harness.evidence import evaluate_snapshot
    from report_tts_increment_control import edit_distance,text_normalize
    scope=rows(TEST);results={}
    for label,path in locations(root).items():
        results[label]=[]
        for seed in SEEDS:
            saved=read(path/f'seed_{seed}/result.json')
            pred=rows(path/f'seed_{seed}/evaluation/merged_step_50/predictions.jsonl')
            assert len(pred)==511 and (saved['seed'],saved['step'])==(seed,50)
            for a,b in zip(pred,scope):assert all(a[k]==b[k] for k in ('utt_id','speaker_id','audio','clean_gt','zero_shot_bucket'))
            metrics=evaluate_snapshot(pred,prediction_column='prediction',critical_cer_threshold=.5).model_dump(mode='json')
            for group in ('vlink247','other_patients'):
                subset=[r for r in pred if (r['speaker_id']=='vlink247')==(group=='vlink247')]
                metrics[group]=evaluate_snapshot(subset,prediction_column='prediction',critical_cer_threshold=.5).model_dump(mode='json')['overall']
            assert (metrics['vlink247']['sample_count'],metrics['other_patients']['sample_count'])==(106,405)
            values=[(edit_distance(text_normalize(r['clean_gt']),text_normalize(r['prediction'])),len(text_normalize(r['clean_gt']))) for r in pred]
            for group in GROUPS:
                indexes=[i for i,r in enumerate(pred) if group=='overall' or r['zero_shot_bucket']==group or (group=='vlink247' and r['speaker_id']=='vlink247') or (group=='other_patients' and r['speaker_id']!='vlink247')]
                assert sum(values[i][0] for i in indexes)==metrics[group]['total_edits']
                assert math.isclose(statistics.mean(values[i][0]/values[i][1] for i in indexes),metrics[group]['mean_utterance_cer'],abs_tol=1e-12)
                if group in saved['metrics']:
                    for k in ('mean_utterance_cer','pooled_cer','patient_macro_cer'):
                        assert math.isclose(metrics[group][k],saved['metrics'][group][k],abs_tol=1e-12)
            results[label].append({'seed':seed,'metrics':metrics})
    means={a:{g:statistics.mean(r['metrics'][g]['mean_utterance_cer'] for r in rs) for g in GROUPS} for a,rs in results.items()}
    comparisons={}
    for baseline in ('real2117','real2117_q_hardtts576',ARMS[0]):
        comparisons[baseline]={}
        for group in GROUPS:
            deltas=[100*(a['metrics'][group]['mean_utterance_cer']-b['metrics'][group]['mean_utterance_cer']) for a,b in zip(results[ARMS[1]],results[baseline])]
            comparisons[baseline][group]={'mean_delta_pp':statistics.mean(deltas),'seed_deltas_pp':deltas,'improved_seeds':sum(d<0 for d in deltas)}
    save(root/'results.json',{'results':results,'means':means,'mas_hard_minus_baseline':comparisons})
    lines=['# MAS576 换 hard reference：ASR结果','','全部组使用2117真实数据；固定step50，三个ASR seed；逐句平均CER（%）。','',
        '| 组别 | Overall | vlink247 | 其他患者 | Easy | Medium | Hard |','|---|---:|---:|---:|---:|---:|---:|']
    for label in means:lines.append(f'| {label} | '+' | '.join(f'{100*means[label][g]:.4f}' for g in GROUPS)+' |')
    lines+=['','## MAS两组逐seed','','| 组别 | Seed | Overall | vlink247 | 其他患者 | Easy | Medium | Hard |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        for result in results[arm]:lines.append(f'| {arm} | {result["seed"]} | '+' | '.join(f'{100*result["metrics"][g]["mean_utterance_cer"]:.4f}' for g in GROUPS)+' |')
    for baseline,comparison in comparisons.items():
        lines+=['',f'MAS hard − {baseline}，百分点，负值改善：','']
        lines.extend(f'- {g}: {comparison[g]["mean_delta_pp"]:+.4f}（{comparison[g]["improved_seeds"]}/3 seeds改善）' for g in GROUPS)
    lines+=['','MAS两组严格保留36个新文本、51位患者、576条配对和真实数据顺序；旧音频复用，新音频更换reference。',
        '与Q重建TTS相比，MAS同时改变了文本和患者分配，vlink247从122条变为6条，不能把差异唯一归因于文本。',
        '既有511条为开发反馈集合；固定更新预算未固定真实曝光；高CER reference不是临床严重度保证。','']
    write_immutable_bytes(root/'report.zh.md','\n'.join(lines).encode())
    save(root/'acceptance.json',{'status':'verified','new_runs':6,'reused_runs':6,'rows_per_evaluation':511,
        'independent_verification':'12 runs x6 groups','results_sha256':sha256_file(root/'results.json')})
    print('\n'.join(lines),flush=True)


def parallel_children(root,specifications):
    children=[];handles=[]
    try:
        for gpu,label,arguments in specifications:
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false')
            log=(root/f'{label}_{time.time_ns()}.log').open('x');handles.append(log)
            command=[sys.executable,'-u',str(Path(__file__)),'--root',str(root),*arguments]
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
            children.append((label,child))
            print(f'[child] {label} pid={child.pid} gpu={gpu} log={log.name}',flush=True)
        codes=[(label,child.wait()) for label,child in children]
        assert all(code==0 for _,code in codes),codes
    finally:
        for handle in handles:handle.close()


def run(root):
    verify(root)
    if not (root/'assembly.json').exists():
        full.status(root,'running',stage='synthesize')
        parallel_children(root,[(gpu,f'synthesis_worker_{i}',['--stage','synthesize','--worker',str(i)]) for i,gpu in enumerate(GPUS)])
        assemble(root)
    for seed in SEEDS:
        pending=[(gpu,f'{arm}_seed_{seed}',['--stage','train','--arm',arm,'--seed',str(seed)]) for arm,gpu in zip(ARMS,GPUS) if not (root/arm/f'seed_{seed}/result.json').exists()]
        full.status(root,'running',stage='train',seed=seed)
        if pending:parallel_children(root,pending)
    full.status(root,'running',stage='report')
    if not (root/'acceptance.json').exists():report(root)
    full.status(root,'complete',stage='report')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--stage',required=True,choices=('prepare','synthesize','assemble','train','report','run'))
    parser.add_argument('--worker',type=int,choices=(0,1))
    parser.add_argument('--arm',choices=ARMS)
    parser.add_argument('--seed',type=int,choices=SEEDS)
    args=parser.parse_args()
    if args.stage=='train' and (args.arm is None or args.seed is None):parser.error('train requires --arm and --seed')
    if args.stage=='synthesize' and args.worker is None:parser.error('synthesize requires --worker')
    try:
        if args.stage=='train':train(args.root,args.arm,args.seed)
        elif args.stage=='synthesize':synthesize(args.root,args.worker)
        else:globals()[args.stage](args.root)
    except Exception as error:
        if args.root.exists():full.status(args.root,'failed',stage=args.stage,worker=args.worker,arm=args.arm,seed=args.seed,error=f'{type(error).__name__}: {error}')
        raise
