#!/usr/bin/env python3
"""Matched semantic-context DPO real511 evaluation and target recovery."""
import gzip
import json
from pathlib import Path
import mas_semantic_context_probe_v1 as run
import mas_evaluation_report as cer_report
from therapist_harness.evidence import compare_prediction_snapshots, _key_descriptor

base, ROOT = run.base, run.ROOT


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
        content_delivery=base.read(ROOT/'delivery_report.json')['arms'],
        scope='Paired natural/odd sentences, same normalized character multiset and length, patients and initial generation recipes/seeds.48newaudio. Same new oddness-aware coarse prompt. Matched slots with genuine preferences on BOTH sides supply24targetvisits; same8historicalanchors,real32DPO/91pairs andKL256. Conditional matched-eligibility comparison. Originalgateunchanged; repeated511development, notindependenttest orautomaticpromotion.',
        source_sha256={str(f):base.sha256_file(f) for f in [Path(__file__),Path(cer_report.__file__),*paths.values(),ROOT/'training_plan/plan.json',ROOT/'proposal.json',ROOT/'delivery_report.json']}))
    labels=dict(initial='起点R-only',R_DPO='已有R/DPO',MAS='古怪语义句＋DPO',fixed='自然语义句＋DPO')
    lines=['同用字自然句与古怪句的匹配DPO结果','', '|条件|CER|错误数|hard错误|原验收|','|---|---:|---:|---:|---|']
    for a,s in summary.items():lines.append(f'|{labels[a]}|{s["overall"]["CER_percent"]:.4f}%|{s["overall"]["errors"]}|{s["hard"]["errors"]}|{s.get("passes_original_gate","基准/已有控制")}|')
    lines+=['','每个混合臂相同真实32音频/91对、256次KL访问，合成32次访问（24次匹配目标＋8次共享保持内容）。两边都有实际错误的音频位置才进入匹配训练；全部48音频的候选产出另行完整报告。相同用字、长度、患者、reference、初始生成seed及新版审查提示。','', '音素签名来自ASR文字；粗审通过不证明逐字保真。511条为开发反馈；不自动提升checkpoint。']
    base.write_immutable_bytes(ROOT/'results.zh.md',('\n'.join(lines)+'\n').encode())
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
