"""Compact live dashboard and incremental transcript audit; no GPU work."""
import argparse,csv,hashlib,json,sys,time
from collections import Counter
from datetime import datetime,timezone
from functools import lru_cache
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from therapist_harness.text import text_normalize,edit_distance
ROOT=Path('/data/qwen3-asr/records/therapist_mas_setting_d_v1/whisper_large_v3_real2117_lora_seed42_4configs_20260910_v2')
PREVIOUS=ROOT.parent/'qwen3_asr_fresh_lora_sft_seed42_EFGH_20260910_v1'
RONLY=ROOT.parent/'qwen3_asr_lora_sft_audit_20260910_v2/evaluations/000_continue_original/dev_merged.predictions.json'
CONFIGS=['W-A','W-B','W-C','W-D']
norm=lru_cache(10000)(text_normalize)
def read(p):return json.loads(p.read_text())
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.pending');q.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');q.replace(p)
def events(p):
 if not p.exists():return []
 return [json.loads(s) for s in p.read_text().splitlines(keepends=True) if s.endswith('\n') and s.strip()]
def metric(rows):
 def one(xs):return 100*sum(x['errors'] for x in xs)/sum(x['chars'] for x in xs)
 return dict(all=one(rows),**{b:one([x for x in rows if x['zero_shot_bucket']==b]) for b in ['easy','medium','hard']},vlink247=one([x for x in rows if x['speaker_id']=='vlink247']),other=one([x for x in rows if x['speaker_id']!='vlink247']))
def marker(target,pred):
 if target=='根據公告該繳件價格正受緊急監管' and ('票券價格' in pred or '票劵價格' in pred):return 'ticket'
 if target=='邊個鑲條裙褂喺畫框入面呢' and '將條裙掛' in pred and '畫框' in pred:return 'hang_skirt'
 if target=='徐太收藏及編輯一切有關秦朝嘅書冊' and '收藏' in pred and '書籍' in pred:return 'book_synonym'
 return ''
@lru_cache(20000)
def nearest_train(pred,texts):
 target=min(texts,key=lambda t:edit_distance(t,pred)/len(t));return target,edit_distance(target,pred)/len(target)
def audit(path,out,dev,train_texts):
 if out.exists():return read(out)
 predictions=read(path);assert len(predictions)==511 and {x['utt_id'] for x in predictions}==set(dev)
 result=[]
 for p in predictions:
  d=dev[p['utt_id']];gt=norm(d['clean_gt']);hyp=norm(p['prediction']);assert edit_distance(gt,hyp)==p['errors']
  near,distance=nearest_train(hyp,train_texts)
  result.append(dict(utt_id=p['utt_id'],speaker_id=p['speaker_id'],bucket=d['zero_shot_bucket'],target=d['clean_gt'],prediction=p['prediction'],normalized_prediction=hyp,
   errors=p['errors'],chars=p['chars'],exact=hyp==gt,marker=marker(d['clean_gt'],hyp),exact_training_sentence=hyp in train_texts,
   closest_training_sentence=near,closest_training_CER=distance,closer_to_train_than_target=distance<p['errors']/p['chars'],
   deletion_control=d['clean_gt']=='邊幾屆織頸巾嘅冠軍鞠躬敬禮呢' and '屆頸巾' in hyp))
 r=dict(prediction_path=str(path),prediction_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),rows=result,pattern_counts=dict(Counter(x['marker'] for x in result if x['marker'])),exact_train=sum(x['exact_training_sentence'] for x in result),closer_train_proxy=sum(x['closer_to_train_than_target'] for x in result))
 save(out,r);return r

def render():
 if not (ROOT/'protocol.json').exists():print('Whisper: 输入准备中');return
 dev={x['utt_id']:x for x in read(ROOT/'dev.json')};texts=tuple(sorted({norm(x['clean_gt']) for x in read(ROOT/'train.json')}))
 lines=['# Whisper large-v3：真实数据 SFT 对照','',
 '2117 条真实训练音频；511 条既有开发评估音频，已反复用于反馈。句子与训练集不重叠，患者有重叠。',
 'seed42；全新 LoRA r16/alpha32/dropout0.05；每次更新32条，末批5条；3 epoch，每约1/5轮评估。',
 '粤语 yue＋transcribe，greedy，无目标句提示、无跨音频上下文；官方30秒128-bin特征，不截断音频。',
 'GPU0/1 同时运行 Qwen 50epoch；GPU2 W-A→W-B，GPU3 W-C→W-D。',
 'Whisper 使用未合并 LoRA 评估并验证保存重载一致；Qwen 数值沿用历史 BF16 合并评估，不能把跨模型绝对差异全部归因于架构。','',
 '|模型/阶段|总体 CER|easy|medium|hard|vlink247|other|','|---|---:|---:|---:|---:|---:|---:|']
 for label,path in [('原始 Qwen',PREVIOUS/'base_evaluation/predictions.json'),('Qwen R-only',RONLY),('原始 Whisper',ROOT/'base_evaluation/predictions.json')]:
  if path.exists():
   rows=read(path);assert {x['utt_id'] for x in rows}==set(dev)
   m=metric(rows);lines.append('|'+label+'|'+'|'.join(f'{x:.4f}%' for x in m.values())+'|')
 print(datetime.now(timezone.utc).strftime('%H:%M:%S UTC')+' | Whisper large-v3 | seed42 | 2117 real / 511 dev')
 print('GPU2: W-A→W-B | GPU3: W-C→W-D | encoder / encoder+decoder LoRA | 3 epoch')
 if (ROOT/'base_evaluation/summary.json').exists():print(f"原始 Whisper CER {read(ROOT/'base_evaluation/summary.json')['CER_percent']:.4f}% | Qwen 25.1352% | R-only 20.7933%")
 else:print('原始 Whisper 完整511条评估中')
 detail=[];trajectory={};auditroot=ROOT/'error_analysis'
 basepath=ROOT/'base_evaluation/predictions.json'
 baseaudit=audit(basepath,auditroot/'base.json',dev,texts) if basepath.exists() else None
 for name in CONFIGS:
  es=events(ROOT/name/'events.jsonl');ev=[e for e in es if e['stage']=='evaluation_complete'];losses=[e for e in es if e['stage']=='loss'];fail=[e for e in es if e['stage']=='failure']
  state='完成' if (ROOT/name/'complete.json').exists() else '训练中' if losses else 'smoke' if es else '等待'
  if fail:state='失败 '+fail[-1]['error']
  param=read(ROOT/name/'smoke_passed.json')['trainable_parameters'] if (ROOT/name/'smoke_passed.json').exists() else None
  latest=losses[-1] if losses else None
  print(f"\n{name} {'encoder' if name in ['W-A','W-B'] else 'encoder+decoder'} | LR {'2e-5' if name in ['W-A','W-C'] else '5e-5'} | {state} | 参数 {param}")
  if latest:print(f"step {latest['step']}/201 | epoch {latest['epoch']:.3f} | loss {latest['loss']:.4f} | lr {latest['learning_rate']:.3g}")
  for e in ev[-2:]:print(f"评估 step{e['step']}: CER {e['metrics']['CER_percent']:.3f}% | easy {e['metrics']['easy']['CER_percent']:.2f}% hard {e['metrics']['hard']['CER_percent']:.2f}%")
  if ev:
   best=min(ev,key=lambda e:e['metrics']['CER_percent']);print(f"最佳 {best['metrics']['CER_percent']:.4f}% @step{best['step']} | 已评估{len(ev)}/15")
   for label,e in [('最佳',best),('最新',ev[-1])]:
    m=e['metrics'];nums=[m['CER_percent']]+[m[b]['CER_percent'] for b in ['easy','medium','hard','vlink247','other']]
    lines.append(f'|{name} {label} step{e["step"]}|'+'|'.join(f'{x:.4f}%' for x in nums)+'|')
  first_correct={x['utt_id']:dict(step=0,prediction=x['prediction']) for x in baseaudit['rows'] if x['exact']} if baseaudit else {}
  stages=[]
  if baseaudit:stages.append(dict(step=0,pattern_counts=baseaudit['pattern_counts'],exact_train=baseaudit['exact_train'],closer_train_proxy=baseaudit['closer_train_proxy']))
  for e in ev:
   s=e['step'];a=audit(ROOT/name/'evaluations'/f'step-{s}'/'predictions.json',auditroot/name/f'step-{s}.json',dev,texts)
   transitions=[]
   for x in a['rows']:
    if x['marker'] and x['utt_id'] in first_correct:transitions.append(dict(**x,earlier_correct=first_correct[x['utt_id']]))
    if x['exact']:first_correct.setdefault(x['utt_id'],dict(step=s,prediction=x['prediction']))
   stages.append(dict(step=s,epoch=e['epoch'],pattern_counts=a['pattern_counts'],exact_train=a['exact_train'],closer_train_proxy=a['closer_train_proxy'],earlier_correct_now_contextual_error=transitions))
   m=e['metrics'];detail.append(f"|{name}|{s}|{e['epoch']:.3f}|{m['CER_percent']:.4f}%|{a['pattern_counts'].get('ticket',0)}|{a['pattern_counts'].get('hang_skirt',0)}|{a['pattern_counts'].get('book_synonym',0)}|{len(transitions)}|{a['exact_train']}|{a['closer_train_proxy']}|")
  trajectory[name]=stages
 lines+=['','## 全程指标与错误追踪','',
 '复用上一轮三个已人工解释的特定替换模式：繳件→票券、鑲條裙褂→將條裙掛、書冊→書籍。这是有限词组追踪，不是完整语义错误率，也不证明语言 prior 的因果作用。',
 '“早对后错”要求同一音频在更早 checkpoint 转写完全正确，随后命中上述特定错误模式。“训练句命中”是完整转写等于48个训练目标之一；“更近训练句”仅为按目标长度归一化编辑距离的探索代理，不能单独证明记忆训练文本。',
 '基线也运行相同错误统计；每个阶段保存完整逐条分析及来源预测 SHA256。', '',
 '|组|step|epoch|CER|票券|將條裙掛|書籍|早对后语境错|完整训练句命中|更近训练句代理|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']+detail
 examples=[]
 for name,stages in trajectory.items():
  if not stages:continue
  for x in stages[-1].get('earlier_correct_now_contextual_error',[])[:5]:
   examples.append(f"|{name}|{x['utt_id']}|{x['target']}|step{x['earlier_correct']['step']}: {x['earlier_correct']['prediction']}|{x['prediction']}|")
 if examples:lines+=['','## 同一音频早对后错的实例','', '|组|音频|目标|更早的正确转写|最新的语境替换|','|---|---|---|---|---|']+examples
 if (ROOT/'complete.json').exists():
  final_figures(trajectory)
  lines+=['',f'![CER 与 loss 趋势]({ROOT}/training_trends.png)',f'![特定错误模式趋势]({ROOT}/error_pattern_trends.png)']
 lines+=['','模型与论文：[官方 large-v3](https://huggingface.co/openai/whisper-large-v3)；[英语 TORGO encoder-only 研究](https://aclanthology.org/2025.emnlp-main.1701/)。本文实验不是该论文的精确复现；其 small/medium 英语结果不能直接推出本次粤语 large-v3 LoRA 效果。','',f'[逐阶段错误轨迹]({auditroot}/trajectories.json)','', 'tmux：`tmux attach -t whisper-real-sft`']
 save(auditroot/'trajectories.json',trajectory)
 save(auditroot/'analysis_provenance.json',dict(script=str(Path(__file__)),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),codebook_sha256=hashlib.sha256((ROOT/'codebook.json').read_bytes()).hexdigest()))
 p=ROOT/'comparison.zh.md';q=p.with_suffix('.pending');q.write_text('\n'.join(lines)+'\n');q.replace(p)
 print('\nCtrl+b 后 0=监督 1=本页 2=W-A/B日志 3=W-C/D日志 4=GPU')
def final_figures(trajectory):
 if (ROOT/'training_trends.png').exists() and (ROOT/'error_pattern_trends.png').exists():return
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 import numpy as np
 fig,axes=plt.subplots(1,3,figsize=(15,4.2))
 for name in CONFIGS:
  es=events(ROOT/name/'events.jsonl');ev=[e for e in es if e['stage']=='evaluation_complete'];ls=[e for e in es if e['stage']=='loss']
  base=read(ROOT/'base_evaluation/summary.json')
  x=[0]+[e['epoch'] for e in ev]
  axes[0].plot(x,[base['CER_percent']]+[e['metrics']['CER_percent'] for e in ev],label=name)
  axes[1].plot(x,[base['hard']['CER_percent']]+[e['metrics']['hard']['CER_percent'] for e in ev],label=name)
  for i in range(0,len(ls),10):
   chunk=ls[i:i+10]
   axes[2].plot(np.mean([e['epoch'] for e in chunk]),np.mean([e['loss'] for e in chunk]),marker='.',color=f'C{CONFIGS.index(name)}')
  axes[2].plot([],[],label=name,color=f'C{CONFIGS.index(name)}')
 axes[0].axhline(20.793269,color='gray',linestyle='--',label='Qwen R-only')
 for ax,title,y in zip(axes,['All development audio','Hard development audio','Training loss (10-step mean)'],['CER (%)','CER (%)','Utterance CE']):
  ax.set(title=title,xlabel='Epoch',ylabel=y);ax.grid(alpha=.2);ax.legend(fontsize=8)
 fig.tight_layout();fig.savefig(ROOT/'training_trends.png',dpi=160);fig.savefig(ROOT/'training_trends.svg');plt.close(fig)
 fig,axes=plt.subplots(1,4,figsize=(16,3.8))
 for name,stages in trajectory.items():
  for ax,key in zip(axes[:3],['ticket','hang_skirt','book_synonym']):ax.plot([x.get('epoch',0) for x in stages],[x['pattern_counts'].get(key,0) for x in stages],label=name)
  axes[3].plot([x.get('epoch',0) for x in stages],[x['exact_train'] for x in stages],label=name)
 for ax,title in zip(axes,['Ticket substitution (n=46)','Hang-skirt substitution (n=43)','Book synonym (n=40)','Exact training-sentence matches']):
  ax.set(title=title,xlabel='Epoch',ylabel='Number of audio samples');ax.grid(alpha=.2);ax.legend(fontsize=8)
 fig.tight_layout();fig.savefig(ROOT/'error_pattern_trends.png',dpi=160);fig.savefig(ROOT/'error_pattern_trends.svg');plt.close(fig)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--loop',action='store_true');args=parser.parse_args()
 while True:
  print('\033[2J\033[H',end='');render()
  if not args.loop:break
  time.sleep(10)
