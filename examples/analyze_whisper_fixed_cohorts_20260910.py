"""Decompose CER changes on baseline-defined fixed cohorts; retain all 511."""
from pathlib import Path
import argparse,csv,hashlib,json,time
from datetime import datetime,timezone
RUN=Path('/data/qwen3-asr/records/therapist_mas_setting_d_v1/whisper_large_v3_real2117_lora_seed42_4configs_20260910_v2')
OUT=RUN/'fixed_cohort_analysis'
def read(p):return json.loads(p.read_text())
def write(p,x):
 q=p.with_suffix(p.suffix+'.pending');q.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');q.replace(p)
def events(p):
 if not p.exists():return []
 return [json.loads(s) for s in p.read_text().splitlines(keepends=True) if s.endswith('\n') and s.strip()]
def metric(rows):
 e=sum(x['errors'] for x in rows);c=sum(x['chars'] for x in rows)
 return dict(audio=len(rows),errors=e,chars=c,CER_percent=100*e/c)
def analyze():
 OUT.mkdir(exist_ok=True);basepath=RUN/'base_evaluation/predictions.json';base=read(basepath);by_id={x['utt_id']:x for x in base};assert len(by_id)==511
 original_cap={x['utt_id'] for x in base if x['hit_token_cap']};assert len(original_cap)==15
 signature={name:[e['step'] for e in events(RUN/name/'events.jsonl') if e['stage']=='evaluation_complete'] for name in ['W-A','W-B','W-C','W-D']}
 if (OUT/'results.json').exists() and read(OUT/'results.json')['signature']==signature:return
 rows=[];proof={str(basepath):hashlib.sha256(basepath.read_bytes()).hexdigest()};tot=metric(base)
 for name,steps in signature.items():
  for step in [0]+steps:
   p=basepath if step==0 else RUN/name/'evaluations'/f'step-{step}'/'predictions.json';pred=read(p)
   assert len(pred)==511 and {x['utt_id'] for x in pred}==set(by_id)
   proof[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
   groups={'base_cap15':[x for x in pred if x['utt_id'] in original_cap],'base_noncap496':[x for x in pred if x['utt_id'] not in original_cap]}
   current=metric(pred);current_cap={x['utt_id'] for x in pred if x['hit_token_cap']}
   item=dict(config=name,step=step,epoch=step//67+(step%67)*32/2117,**current,
    improved_audio=sum(x['errors']<by_id[x['utt_id']]['errors'] for x in pred),
    worsened_audio=sum(x['errors']>by_id[x['utt_id']]['errors'] for x in pred),
    unchanged_audio=sum(x['errors']==by_id[x['utt_id']]['errors'] for x in pred),
    exact_audio=sum(x['errors']==0 for x in pred),current_cap=len(current_cap),new_cap=len(current_cap-original_cap),resolved_cap=len(original_cap-current_cap),
    total_errors_reduced=tot['errors']-current['errors'])
   for label,xs in groups.items():
    m=metric(xs);base_errors=sum(by_id[x['utt_id']]['errors'] for x in xs)
    item[label+'_CER']=m['CER_percent'];item[label+'_errors']=m['errors'];item[label+'_errors_reduced']=base_errors-m['errors']
   assert item['base_cap15_errors_reduced']+item['base_noncap496_errors_reduced']==item['total_errors_reduced']
   assert item['improved_audio']+item['worsened_audio']+item['unchanged_audio']==511
   rows.append(item)
 result=dict(updated_utc=datetime.now(timezone.utc).isoformat(),signature=signature,baseline_cap_ids=sorted(original_cap),rows=rows,source_sha256=proof,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
  interpretation='Fixed baseline cohorts, not changing exclusions. Primary CER always uses all511. Improvement in baseline noncap496 measures more than resolving original15 cap cases, but can include new cap failures and is not a causal attribution to phonetic learning.')
 write(OUT/'results.json',result)
 with (OUT/'timeline.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 lines=['# Whisper 改善来自哪里：固定音频分组','',
  '固定以原始 Whisper 的结果分组：15条达到256生成token上限，另496条未达上限。以后所有checkpoint都使用这两组相同音频；主CER始终保留全部511条，不删除失败样本。',
  '原始15条贡献2569个字符编辑错误，另496条贡献3449个。两个固定分组减少的错误数之和，必须等于总体减少的错误数。',
  '“496条改善”可以支持改善并非只来自最初15条长循环，但仍不能独自证明学到了哪一种发音机制；这组也可能出现新的长循环。新达到上限和解除上限另列。','',
  '|配置|step|epoch|总体CER|固定496条CER|15条减少错误|496条减少错误|变好/变差/不变音频|完全正确|当前/新增/解除上限|',
  '|---|---:|---:|---:|---:|---:|---:|---|---:|---|']
 for x in rows:
  lines.append(f"|{x['config']}|{x['step']}|{x['epoch']:.3f}|{x['CER_percent']:.4f}%|{x['base_noncap496_CER']:.4f}%|{x['base_cap15_errors_reduced']}|{x['base_noncap496_errors_reduced']}|{x['improved_audio']}/{x['worsened_audio']}/{x['unchanged_audio']}|{x['exact_audio']}|{x['current_cap']}/{x['new_cap']}/{x['resolved_cap']}|")
 lines+=['',f'[机器可读结果]({OUT}/results.json) · [CSV]({OUT}/timeline.csv) · [主对照报告]({RUN}/comparison.zh.md)']
 p=OUT/'report.zh.md';q=p.with_suffix('.pending');q.write_text('\n'.join(lines)+'\n');q.replace(p)
 print(json.dumps({n:[x for x in rows if x['config']==n][-1] for n in signature}),flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--loop',action='store_true');args=parser.parse_args()
 while True:
  analyze()
  if not args.loop or (RUN/'complete.json').exists():break
  time.sleep(40)
