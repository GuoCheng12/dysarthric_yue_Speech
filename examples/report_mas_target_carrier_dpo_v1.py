#!/usr/bin/env python3
"""Same-pair all-edit versus target-character scoring: real511 results."""
import gzip
import json
from pathlib import Path
import prepare_mas_target_carrier_dpo_v1 as run
import mas_evaluation_report as cer_report
from therapist_harness.evidence import compare_prediction_snapshots, _key_descriptor

base, ROOT = run, run.ROOT
base.save = base.write_immutable_json


def rows(path):
    path=Path(path);s=gzip.decompress(path.read_bytes()).decode() if path.suffix=='.gz' else path.read_text()
    return [json.loads(x) for x in s.splitlines() if x.strip()]


def main():
    plan=base.read(ROOT/'training_plan/plan.json');modes=plan['train_modes']
    paths=dict(initial=base.B/'parent_eval_check/predictions.jsonl',
        R_DPO=Path('/tmp/mas_deterministic_matched_DPO_v1/R_DPO_KL/evaluation_run/evaluation/predictions.jsonl.gz'))
    paths.update({m:ROOT/'training'/m/'evaluation/predictions.jsonl.gz' for m in modes})
    data={a:rows(p) for a,p in paths.items()};ids=[x['utt_id'] for x in data['initial']]
    assert len(ids)==len(set(ids))==511 and all([x['utt_id'] for x in rs]==ids for rs in data.values())
    summary={}
    for a,rs in data.items():
        groups=dict(overall=rs,hard=[x for x in rs if x['zero_shot_bucket']=='hard'],
            vlink247=[x for x in rs if x['speaker_id'].startswith('vlink247')],others=[x for x in rs if not x['speaker_id'].startswith('vlink247')])
        summary[a]={k:dict(errors=sum(x['edit_count'] for x in xs),characters=sum(x['normalized_reference_characters'] for x in xs)) for k,xs in groups.items()}
        for x in summary[a].values():x['CER_percent']=100*x['errors']/x['characters']
        if a in modes:
            result=base.read(ROOT/'training'/a/'result.json');assert result['status']=='training_and_evaluation_complete'
            summary[a]['passes_original_gate']=result['passes_original_gate']
    assert summary['initial']['overall']['errors']==1384 and summary['R_DPO']['overall']['errors']==1399
    contrasts=[('initial',m) for m in modes]+[('R_DPO',m) for m in modes]
    if len(modes)==2:contrasts.append(('fixed','MAS'))
    descriptors=[_key_descriptor(('substitution','coda',a,b,None,None)) for a,b in [('ng','n'),('n','ng')]]
    context_path=base.PREVIOUS/'request_context.json'
    real_context=base.read(context_path)
    literal_edges=sorted({(e['event']['reference_character'],e['event']['hypothesis_character']) for patient in real_context['patients'] for e in patient['actual_events'] if e['event']['reference_character'] in {'生','盲','框'}})
    descriptors += [_key_descriptor(('substitution','character',a,b,None,None)) for a,b in literal_edges]
    target={};paired={};cer_report.rows=rows
    for a,b in contrasts:
        comparison=compare_prediction_snapshots(data[a],data[b],before_prediction_column='prediction',after_prediction_column='prediction',
            critical_cer_threshold=.5,monitored_clusters=descriptors,off_target_max_per_level_per_direction=2)
        target[a+'_to_'+b]=[x.model_dump(mode='json') for x in comparison.monitored_cluster_changes]
        paired[a+'_to_'+b]=cer_report.compare(paths[a],paths[b])
    real_control=rows('/tmp/mas_deterministic_matched_DPO_v1/R_DPO_KL/training_exposures.jsonl')
    for m in modes:
        es=rows(ROOT/'training'/m/'training_exposures.jsonl')
        assert [e for e in es if e['source']=='real']==real_control
        assert len({(e['step'],e['occurrence']) for e in es if e['source']=='synthetic'})==32
    base.save(ROOT/'results.json',dict(summary=summary,mode_labels=plan['mode_labels'],paired_CER=paired,target_changes=target,
        scoring_checks=base.read(ROOT/'focus_token_plan.json')['checks'],
        scope='Identical actual odd-content5audio14targetpairs plus8sharedanchors28pairs in both arms. Same42SGpairs,13uniqueSGaudio,32SGvisits (24target+8anchor),real32DPO91pairs+KL256. ONLY14targetpairscoringpositions differ: all-edit+EOS versus target-character-overlap token+EOS. Full audio/text conditioning remains identical. SameinitialR-onlyparentandfrozenreference. Originalgateunchanged; repeated511development, notindependenttest. Previousodd1408usedadifferent19slotpoolandisnotthismatchedcontrol.',
        source_sha256={str(f):base.sha256_file(f) for f in [Path(__file__),Path(cer_report.__file__),context_path,*paths.values(),ROOT/'training_plan/plan.json',ROOT/'focus_token_plan.json',ROOT/'label_smoke/result.json']}))
    labels=dict(initial='起点R-only',R_DPO='已有R/DPO',MAS='目标字符＋EOS计分',fixed='全部变化token＋EOS计分')
    lines=['相同目标偏好对：全部变化位置与目标字符计分的DPO结果','', '|条件|CER|错误数|hard错误|原验收|','|---|---:|---:|---:|---|']
    for a,s in summary.items():lines.append(f'|{labels[a]}|{s["overall"]["CER_percent"]:.4f}%|{s["overall"]["errors"]}|{s["hard"]["errors"]}|{s.get("passes_original_gate","基准/已有控制")}|')
    lines+=['','每组相同真实32音频/91对、256次KL访问；相同5条古怪句目标音频/14对，以及8条共享保持音频/28对。相同合成32次访问，仅14个目标偏好对的计分位置不同。保持完整上下文及EOS；这是掩码偏好损失对照。前一轮1408来自不同数据子集，不能充当此处匹配控制。','', '音素签名来自ASR文字；粗审通过不证明逐字保真。511条为开发反馈；不自动提升checkpoint。']
    base.write_immutable_bytes(ROOT/'results.zh.md',('\n'.join(lines)+'\n').encode())
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
