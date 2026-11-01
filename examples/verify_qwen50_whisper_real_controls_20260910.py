"""Verify saved outcomes against the actual fixed data and exposure orders."""
from pathlib import Path
import hashlib,json,sys
from datetime import datetime,timezone
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from therapist_harness.text import text_normalize,edit_distance
ROOT=Path('/data/qwen3-asr/records/therapist_mas_setting_d_v1')
Q=ROOT/'qwen3_asr_fresh_G_lr2em4_2gpu_seed42_50epochs_20260910_v1'
W=ROOT/'whisper_large_v3_real2117_lora_seed42_4configs_20260910_v2'
OUT=ROOT/'qwen50_whisper_real_controls_completion_20260910_v1'
def read(p):return json.loads(p.read_text())
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def events(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def metric(xs):return dict(audio=len(xs),errors=sum(x['errors'] for x in xs),chars=sum(x['chars'] for x in xs),CER_percent=100*sum(x['errors'] for x in xs)/sum(x['chars'] for x in xs))
def check_predictions(path,summary,dev):
 rows=read(path);assert len(rows)==len(dev)==511
 assert [x['utt_id'] for x in rows]==[x['utt_id'] for x in dev]
 for r,d in zip(rows,dev,strict=True):
  gt=text_normalize(d['clean_gt']);hyp=text_normalize(r['prediction'])
  assert r['errors']==edit_distance(gt,hyp) and r['chars']==len(gt)
  assert r['speaker_id']==d['speaker_id'] and r['zero_shot_bucket']==d['zero_shot_bucket']
 groups={'all':rows,'vlink247':[r for r in rows if r['speaker_id']=='vlink247'],'other':[r for r in rows if r['speaker_id']!='vlink247']}
 groups.update({b:[r for r in rows if r['zero_shot_bucket']==b] for b in ['easy','medium','hard']})
 for name,xs in groups.items():
  actual=metric(xs);saved=summary if name=='all' else summary[name]
  assert all(actual[k]==saved[k] for k in actual),(path,name,actual,saved)
 return dict(path=str(path),sha256=sha(path),audio=511,errors=summary['errors'],chars=summary['chars'])
def main():
 pending=[str(p) for p in [Q/'complete.json',W/'complete.json'] if not p.exists()]
 if pending:print(json.dumps(dict(status='pending',missing_completion=pending)));return
 qp,wp=read(Q/'protocol.json'),read(W/'protocol.json');assert read(Q/'complete.json')['passed'] and read(W/'complete.json')['passed']
 assert qp['config']['lr']==2e-4 and qp['epochs']==50 and qp['world_size']==2 and qp['global_batch']==32
 assert wp['epochs']==3 and wp['global_batch']==32 and wp['model_id']=='openai/whisper-large-v3'
 assert wp['configs']=={'W-A':{'decoder':False,'lr':2e-5},'W-B':{'decoder':False,'lr':5e-5},'W-C':{'decoder':True,'lr':2e-5},'W-D':{'decoder':True,'lr':5e-5}}
 for protocol in [qp,wp]:
  for field in ['source_sha256','input_sha256']:
   assert all(sha(Path(p))==h for p,h in protocol[field].items()),field
 assert all(sha(Path(p))==h for p,h in wp['base_sha256'].items())
 assert all(sha(Path(p))==h for p,h in qp['base_files_sha256'].items())
 qsmoke=read(Q/'smoke_passed.json')
 assert qsmoke['passed'] and qsmoke['updated_adapter_identical_across_ranks']
 assert [x['audio'] for x in qsmoke['gradient_checks']]==[32,5]
 assert all(x['passed'] for x in qsmoke['gradient_checks'])
 assert len(qsmoke['rank_receipts'])==2
 for rank in qsmoke['rank_receipts']:
  assert all(rank[k] for k in ['passed','base_frozen','language_lora_frozen','audio_lora_updated','post_smoke_restored_exactly'])
  assert rank['trainable_parameters']==2359296
 train,dev=read(W/'train.json'),read(W/'dev.json');assert len(train)==2117 and len(dev)==511
 assert {x['utt_id'] for x in train}=={x['utt_id'] for x in read(Q/'train.json')}
 assert [x['utt_id'] for x in dev]==[x['utt_id'] for x in read(Q/'dev.json')]
 assert not ({x['utt_id'] for x in train}&{x['utt_id'] for x in dev})
 assert not ({text_normalize(x['clean_gt']) for x in train}&{text_normalize(x['clean_gt']) for x in dev})
 assert all(x['token_ids'][:4]==[50258,50358,50360,50364] and x['token_ids'][-1]==50257 for x in train+dev)
 assert all(0<x['actual_samples_16k']<=480000 for x in train+dev)
 receipts=[];branches={Q/'G_audio_lr2em4_2gpu_50epochs':(50,qp,Q),**{W/n:(3,wp,W) for n in wp['configs']}}
 for branch,(epochs,protocol,run) in branches.items():
  complete=read(branch/'complete.json');assert complete['passed'] and complete['actual_audio_exposures']==epochs*2117
  tr,orders=read(run/'train.json'),read(run/'orders.json');es=events(branch/'events.jsonl');ls=[e for e in es if e['stage']=='loss']
  assert [e['step'] for e in ls]==list(range(1,67*epochs+1));assert sum(e['actual_batch'] for e in ls)==epochs*2117
  for i,order in enumerate(orders,1):
   record=read(branch/f'epoch-{i}-exposure.json');assert record['utt_ids']==[tr[k]['utt_id'] for k in order]
   assert len(record['utt_ids'])==len(set(record['utt_ids']))==2117
  ev=[e for e in es if e['stage']=='evaluation_complete'];assert [e['step'] for e in ev]==protocol['evaluation_steps']
  for e in ev:receipts.append(check_predictions(branch/'evaluations'/f"step-{e['step']}"/'predictions.json',e['metrics'],dev))
  if run==Q:
   assert all(complete[k] for k in ['base_unchanged','frozen_lora_unchanged','final_adapters_equal_across_ranks'])
   receipts.append(check_predictions(branch/'final_unmerged_evaluation/predictions.json',complete['final_unmerged_metrics'],dev))
   best=read(branch/'best-checkpoint/provenance.json')
   assert best['CER_percent']==min(e['metrics']['CER_percent'] for e in ev)
   assert any(e['step']==best['step'] and e['metrics']['CER_percent']==best['CER_percent'] for e in ev)
   latest=branch/'latest-checkpoint';pr=read(latest/'provenance.json')
   assert pr['step']==3350 and pr['epoch']==50 and pr['actual_audio_exposures']==105850
   assert sha(latest/'adapter_model.safetensors')==pr['adapter_sha256']
   assert sha(latest/'training_state.pt')==pr['training_state_sha256']
   assert sha(branch/'checkpoint-3350/adapter_model.safetensors')==pr['adapter_sha256']
   for epoch in [3,10,30,50]:
    ck=branch/f'checkpoint-{67*epoch}';cp=read(ck/'provenance.json')
    assert cp['step']==67*epoch and cp['actual_audio_exposures']==2117*epoch
    assert (ck/'adapter_model.safetensors').stat().st_size>0
  if run==W:
   smoke=read(branch/'smoke_passed.json')
   assert all(smoke[k] for k in ['passed','prefix_not_supervised','padding_ignored','eos_supervised','manual_native_loss_matches_builtin','fp32_training_CE_matches_explicit_NLL','frozen_parameters_unchanged','save_reload_token_ids_exact','initial_predictions_match_raw'])
   assert complete['frozen_parameters_unchanged']
   for e in ev:
    ck=branch/f"checkpoint-{e['step']}";pr=read(ck/'provenance.json');assert sha(ck/'adapter_model.safetensors')==pr['adapter_sha256']
   best=read(branch/'best_checkpoint.json');assert best['metrics']['CER_percent']==min(e['metrics']['CER_percent'] for e in ev)
 for run in [Q,W]:receipts.append(check_predictions(run/'base_evaluation/predictions.json',read(run/'base_evaluation/summary.json'),dev))
 smokes=[read(W/n/'smoke_passed.json') for n in wp['configs']]
 assert len({x['fresh_initial_adapter_sha256'] for x in smokes})==len({x['raw_base_tensor_sha256'] for x in smokes})==1
 assert [x['trainable_parameters'] for x in smokes]==[11796480,11796480,28835840,28835840]
 for i in range(1,4):assert len({sha(W/n/f'epoch-{i}-exposure.json') for n in wp['configs']})==1
 cleanup=ROOT/'storage_audit_20260910_v2';assert read(cleanup/'authorized_cleanup_receipt.json')['status']=='complete'
 pre=read(cleanup/'authorized_cleanup_predelete.json');assert all(not Path(x['path']).exists() for x in pre['files'])
 assert all(sha(Path(p))==h for p,h in pre['protected_file_sha256'].items())
 OUT.mkdir(exist_ok=True);result=dict(status='verified',utc=datetime.now(timezone.utc).isoformat(),prediction_receipts=receipts,total_prediction_rows=len(receipts)*511,full_training_coverage_verified=True,whisper_four_branch_common_initialization_and_orders_verified=True,cleanup_and_baselines_verified=True,verifier_sha256=sha(Path(__file__)))
 (OUT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='prediction_receipts'}))
if __name__=='__main__':main()
