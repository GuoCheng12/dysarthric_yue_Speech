"""Incrementally apply the same transcript audit to the concurrent Qwen 50-epoch run."""
import argparse,csv,hashlib,json,time
from datetime import datetime,timezone
from pathlib import Path
import monitor_whisper_real2117_sft_20260910_v2 as common
RUN=Path('/data/qwen3-asr/records/therapist_mas_setting_d_v1/qwen3_asr_fresh_G_lr2em4_2gpu_seed42_50epochs_20260910_v1')
BRANCH=RUN/'G_audio_lr2em4_2gpu_50epochs'
OUT=RUN.parent/'qwen_g50_contextual_substitution_audit_20260910_v1'
def render():
 dev={x['utt_id']:x for x in common.read(RUN/'dev.json')};texts=tuple(sorted({common.norm(x['clean_gt']) for x in common.read(RUN/'train.json')}))
 events=common.events(BRANCH/'events.jsonl');evaluations=[e for e in events if e['stage']=='evaluation_complete']
 signature=[e['step'] for e in evaluations]
 if (OUT/'timeline.json').exists() and common.read(OUT/'timeline.json')['completed_evaluation_steps']==signature:return
 stages=[dict(step=0,epoch=0,path=RUN/'base_evaluation/predictions.json')]+[dict(step=e['step'],epoch=e['epoch'],path=BRANCH/'evaluations'/f"step-{e['step']}"/'predictions.json') for e in evaluations]
 correct={};timeline=[];examples=[]
 for stage in stages:
  a=common.audit(stage['path'],OUT/'stages'/f"step-{stage['step']}.json",dev,texts)
  hits=[]
  for x in a['rows']:
   if x['marker'] and x['utt_id'] in correct:hits.append(dict(**x,earlier_correct=correct[x['utt_id']]))
   if x['exact']:correct.setdefault(x['utt_id'],dict(step=stage['step'],prediction=x['prediction']))
  errors=sum(x['errors'] for x in a['rows']);chars=sum(x['chars'] for x in a['rows'])
  timeline.append(dict(step=stage['step'],epoch=stage['epoch'],CER_percent=100*errors/chars,
   **{k:a['pattern_counts'].get(k,0) for k in ['ticket','hang_skirt','book_synonym']},
   earlier_correct_now_contextual_error=len(hits),exact_training_sentence=a['exact_train'],closer_train_proxy=a['closer_train_proxy']))
  examples=hits
 result=dict(updated_utc=datetime.now(timezone.utc).isoformat(),completed_evaluation_steps=signature,timeline=timeline,latest_examples=examples,latest_step=stages[-1]['step'],
  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),shared_audit_helper_sha256=hashlib.sha256(Path(common.__file__).read_bytes()).hexdigest(),
  scope='Same 3 predefined lexical patterns as prior G10 and current Whisper; not a comprehensive semantic classifier. Nearest-training-text similarity is an exploratory proxy, not proof of memorization.')
 common.save(OUT/'timeline.json',result)
 with (OUT/'timeline.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(timeline[0]));writer.writeheader();writer.writerows(timeline)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(11,4.2))
 x=[r['epoch'] for r in timeline];axes[0].plot(x,[r['CER_percent'] for r in timeline],label='Qwen G LR2e-4 / 50 epochs')
 axes[0].axhline(20.793269,color='gray',linestyle='--',label='R-only');axes[0].set(ylabel='Development CER (%)',xlabel='Epoch');axes[0].legend(fontsize=8)
 for k,label in [('ticket','Ticket (n=46)'),('hang_skirt','Hang-skirt (n=43)'),('book_synonym','Book synonym (n=40)')]:axes[1].plot(x,[r[k] for r in timeline],label=label)
 axes[1].set(xlabel='Epoch',ylabel='Audio with the specified replacement');axes[1].legend(fontsize=8)
 for ax in axes:ax.grid(alpha=.2)
 fig.tight_layout();fig.savefig(OUT/'trends.png',dpi=150);fig.savefig(OUT/'trends.svg');plt.close(fig)
 lines=['# Qwen G、LR2e-4、50 epoch：语境替换追踪','',
  '本报告读取正在运行实验的已完成评估，不修改训练。和 Whisper 四组使用同一词组规则、同一511条开发音频。',
  '三个模式仅追踪繳件→票券、鑲條裙褂→將條裙掛、書冊→書籍，不能代表所有语义错误；命中不等于已经证明语言 prior 因果作用。',
  '早对后错要求同一音频在更早阶段完整正确，当前变成指定词组错误。完整训练句命中和更近训练句代理分开记录，后者不能证明记忆。','',f'![趋势]({OUT}/trends.png)','',
  '|step|epoch|CER|票券|將條裙掛|書籍|早对后语境错|完整训练句命中|更近训练句代理|','|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for r in timeline:lines.append(f"|{r['step']}|{r['epoch']:.3f}|{r['CER_percent']:.4f}%|{r['ticket']}|{r['hang_skirt']}|{r['book_synonym']}|{r['earlier_correct_now_contextual_error']}|{r['exact_training_sentence']}|{r['closer_train_proxy']}|")
 if examples:
  lines+=['','## 最新阶段同音频实例','', '|音频|目标|更早正确转写|当前转写|','|---|---|---|---|']
  for x in examples[:10]:lines.append(f"|{x['utt_id']}|{x['target']}|step{x['earlier_correct']['step']}: {x['earlier_correct']['prediction']}|{x['prediction']}|")
 lines+=['',f'[逐条阶段分析]({OUT}/stages) · [可下载CSV]({OUT}/timeline.csv)', '',f'[Whisper 同规则追踪]({common.ROOT}/comparison.zh.md)']
 p=OUT/'report.zh.md';pending=p.with_suffix('.pending');pending.write_text('\n'.join(lines)+'\n');pending.replace(p)
 print(json.dumps(timeline[-1]),flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--loop',action='store_true');args=parser.parse_args()
 while True:
  render()
  if not args.loop or (RUN/'complete.json').exists():break
  time.sleep(40)
