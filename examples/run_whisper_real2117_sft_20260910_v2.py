"""Independent Whisper large-v3 LoRA controls on the existing real/dev split."""
import argparse, copy, gc, hashlib, json, math, os, random, shutil, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
ROOT=Path('/data/qwen3-asr/records/therapist_mas_setting_d_v1/whisper_large_v3_real2117_lora_seed42_4configs_20260910_v2')
BASE=Path('/data/qwen3-asr/models/whisper-large-v3')
PREVIOUS=ROOT.parent/'qwen3_asr_fresh_lora_sft_seed42_EFGH_20260910_v1'
RONLY=ROOT.parent/'qwen3_asr_lora_sft_audit_20260910_v2/evaluations/000_continue_original/dev_merged.predictions.json'
CODEBOOK=ROOT.parent/'g10_contextual_substitution_audit_20260910_v1/codebook.json'
os.environ['TMPDIR']=str(ROOT/'tmp')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
sys.path.insert(0,str(REPO/'src'))
import numpy as np
import torch
import torch.nn.functional as F
from transformers import WhisperProcessor, WhisperForConditionalGeneration, set_seed, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file
from therapist_harness.text import text_normalize, edit_distance
CONFIGS={'W-A':dict(decoder=False,lr=2e-5),'W-B':dict(decoder=False,lr=5e-5),
         'W-C':dict(decoder=True,lr=2e-5),'W-D':dict(decoder=True,lr=5e-5)}
EVAL_STEPS=[e*67+s for e in range(3) for s in (14,27,40,54,67)]
DECODE=dict(language='yue',task='transcribe',return_timestamps=False,condition_on_prev_tokens=False,
            do_sample=False,num_beams=1,max_new_tokens=256)
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.pending');q.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');q.replace(p)
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def event(out,stage,**x):
 out.mkdir(parents=True,exist_ok=True);r=dict(utc=datetime.now(timezone.utc).isoformat(),stage=stage,**x)
 with (out/'events.jsonl').open('a') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 print(json.dumps(r,ensure_ascii=False),flush=True)
def digest(model,predicate):
 h=hashlib.sha256()
 for n,p in model.named_parameters():
  if predicate(n,p):h.update(n.encode());h.update(p.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
 return h.hexdigest()
def processor():
 proc=WhisperProcessor.from_pretrained(str(BASE),language='cantonese',task='transcribe',local_files_only=True)
 # Fast-tokenizer JSON retains its original postprocessor unless reset explicitly.
 proc.tokenizer.set_prefix_tokens(language='cantonese',task='transcribe',predict_timestamps=False)
 return proc
def prepare():
 import soundfile as sf
 import librosa
 assert not (ROOT/'protocol.json').exists()
 proc=processor();train,dev=read(PREVIOUS/'train.json'),read(PREVIOUS/'dev.json')
 assert len(train)==2117 and len(dev)==511
 assert not ({x['utt_id'] for x in train}&{x['utt_id'] for x in dev})
 assert not ({text_normalize(x['clean_gt']) for x in train}&{text_normalize(x['clean_gt']) for x in dev})
 orders=read(PREVIOUS/'orders.json');assert len(orders)==3
 rows=train+dev;hashes=read(PREVIOUS/'audio_sha256.json')
 prefix=proc.tokenizer.prefix_tokens;assert proc.tokenizer.convert_ids_to_tokens(prefix)==['<|startoftranscript|>','<|yue|>','<|transcribe|>','<|notimestamps|>']
 features=np.lib.format.open_memmap(ROOT/'features.npy',mode='w+',dtype=np.float32,shape=(len(rows),128,3000))
 masks=np.lib.format.open_memmap(ROOT/'feature_masks.npy',mode='w+',dtype=np.int64,shape=(len(rows),3000))
 for i,row in enumerate(rows):
  assert sha(row['audio'])==hashes[row['audio']]
  audio,sr=sf.read(row['audio'],dtype='float32',always_2d=False)
  if audio.ndim==2:audio=audio.mean(axis=1)
  if sr!=16000:audio=librosa.resample(audio,orig_sr=sr,target_sr=16000)
  assert 0<len(audio)<=480000,(row['utt_id'],len(audio))
  enc=proc.feature_extractor(audio,sampling_rate=16000,return_attention_mask=True,return_tensors='np')
  assert enc.input_features.shape==(1,128,3000)
  features[i]=enc.input_features[0];masks[i]=enc.attention_mask[0]
  row['feature_index']=i;row['actual_samples_16k']=len(audio)
  row['token_ids']=proc.tokenizer(row['clean_gt']).input_ids
  assert row['token_ids'][:len(prefix)]==prefix and row['token_ids'][-1]==proc.tokenizer.eos_token_id
  assert len(row['token_ids'])<=448
  if i%128==0:event(ROOT,'feature_cache',completed=i+1,total=len(rows))
 features.flush();masks.flush();del features,masks
 save(ROOT/'train.json',train);save(ROOT/'dev.json',dev);save(ROOT/'orders.json',orders)
 save(ROOT/'codebook.json',read(CODEBOOK));save(ROOT/'audio_sha256.json',hashes)
 sources=[Path(__file__),REPO/'src/therapist_harness/text.py']
 for p in sources:
  dest=ROOT/'sources'/p.relative_to(REPO);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
 save(ROOT/'protocol.json',dict(model_id='openai/whisper-large-v3',revision=read(ROOT/'download.json')['revision'],
  configs=CONFIGS,seed=42,epochs=3,steps=201,train_audio=2117,dev_audio=511,
  dev_status='Repeatedly used development evaluation set; not an untouched test set; known speakers with unseen sentences.',
  lora=dict(rank=16,alpha=32,dropout=.05,target_modules=['q_proj','k_proj','v_proj','out_proj','fc1','fc2'],
   initialization='Identical full encoder+decoder adapter insertion with seed42; decoder LoRA frozen at zero-B for W-A/W-B.'),
  base_dtype='bfloat16',adapter_dtype='float32',gradient_checkpointing='non-reentrant',
  objective='Mean per-utterance CE over target transcript and EOS only; fixed language/task prefix and padding masked. Sum divided by actual optimizer-batch audio count.',
  global_batch=32,microbatch=4,last_batch=5,optimizer='AdamW beta0.9/0.999 eps1e-8 wd0 clip1',
  schedule='linear201 warmup5',evaluation_steps=EVAL_STEPS,decode=DECODE,evaluation_batch=4,
  checkpoint_evaluation='Unmerged adapters throughout; no BF16 merge/unmerge roundoff.',
  features='Official 128-bin 30s feature extractor; float32 on-disk cache; no truncation, augmentation or waveform exclusion.',
  resource_allocation={'GPU0/1':'Qwen LR2e-4 50epoch','GPU2':['Whisper base','W-A','W-B'],'GPU3':['W-C','W-D']},
  source_sha256={str(p):sha(p) for p in sources},base_sha256={str(p):sha(p) for p in BASE.iterdir() if p.is_file()},
  input_sha256={str(ROOT/n):sha(ROOT/n) for n in ['train.json','dev.json','orders.json','features.npy','feature_masks.npy','audio_sha256.json','codebook.json']},
  comparison_sha256={str(p):sha(p) for p in [PREVIOUS/'base_evaluation/predictions.json',RONLY]},
  references=['https://huggingface.co/openai/whisper-large-v3','https://aclanthology.org/2025.emnlp-main.1701/'],
  interpretation='English TORGO encoder-only findings motivate this Cantonese large-v3 LoRA test; no claim of exact paper replication or architecture-only causality.'))
 event(ROOT,'prepared',audio=2117,dev=511,epochs=3,configs=CONFIGS)
class Batches:
 def __init__(self):
  self.features=np.load(ROOT/'features.npy',mmap_mode='r');self.masks=np.load(ROOT/'feature_masks.npy',mmap_mode='r');self.proc=processor()
 def audio(self,rows):
  ids=[r['feature_index'] for r in rows]
  return dict(input_features=torch.from_numpy(self.features[ids].copy()).to('cuda',dtype=torch.bfloat16),attention_mask=torch.from_numpy(self.masks[ids].copy()).to('cuda'))
 def train(self,rows):
  ids=[r['token_ids'] for r in rows];length=max(map(len,ids));pad=self.proc.tokenizer.pad_token_id
  full=torch.full((len(rows),length),pad,dtype=torch.long,device='cuda');mask=torch.zeros_like(full)
  for i,t in enumerate(ids):full[i,:len(t)]=torch.tensor(t,device='cuda');mask[i,:len(t)]=1
  labels=full[:,1:].clone();labels[mask[:,1:]==0]=-100
  labels[:,:len(self.proc.tokenizer.prefix_tokens)-1]=-100
  result=self.audio(rows);result.update(decoder_input_ids=full[:,:-1],decoder_attention_mask=mask[:,:-1]);return result,labels

def build(config=None):
 set_seed(42)
 model=WhisperForConditionalGeneration.from_pretrained(str(BASE),torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).to('cuda')
 model.config.forced_decoder_ids=None;model.generation_config.forced_decoder_ids=None
 model.config.apply_spec_augment=False;model.config.use_cache=False
 if config is not None:
  model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,bias='none',target_modules=['q_proj','k_proj','v_proj','out_proj','fc1','fc2']))
  for n,p in model.named_parameters():
   if not config['decoder'] and '.decoder.' in n:p.requires_grad_(False)
  assert all('.lora_' in n for n,p in model.named_parameters() if p.requires_grad)
  assert all(config['decoder'] or '.encoder.' in n for n,p in model.named_parameters() if p.requires_grad)
  assert all(p.dtype==torch.float32 for n,p in model.named_parameters() if '.lora_' in n)
  assert all(torch.count_nonzero(p)==0 for n,p in model.named_parameters() if '.lora_B.' in n)
 return model

def metrics(rows):
 def one(xs):
  e=sum(r['errors'] for r in xs);c=sum(r['chars'] for r in xs)
  return dict(audio=len(xs),errors=e,chars=c,CER_percent=100*e/c)
 return dict(**one(rows),vlink247=one([r for r in rows if r['speaker_id']=='vlink247']),other=one([r for r in rows if r['speaker_id']!='vlink247']),**{b:one([r for r in rows if r['zero_shot_bucket']==b]) for b in ['easy','medium','hard']})
@torch.inference_mode()
def generate(model,batches,rows):
 model.eval();inner=model.get_base_model() if hasattr(model,'get_base_model') else model
 result=inner.generate(**batches.audio(rows),**DECODE)
 return result.cpu().tolist(),batches.proc.batch_decode(result,skip_special_tokens=True)
def evaluate(model,batches,rows,out):
 was=model.training;cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state();tick=time.monotonic();result=[]
 try:
  for start in range(0,len(rows),4):
   batch=rows[start:start+4];ids,preds=generate(model,batches,batch)
   for r,tokens,pred in zip(batch,ids,preds,strict=True):
    gt=text_normalize(r['clean_gt']);result.append(dict(utt_id=r['utt_id'],speaker_id=r['speaker_id'],zero_shot_bucket=r['zero_shot_bucket'],target=r['clean_gt'],prediction=pred,errors=edit_distance(gt,text_normalize(pred)),chars=len(gt),token_ids=tokens,postprocessed_eos_present=batches.proc.tokenizer.eos_token_id in tokens,hit_token_cap=len(tokens)==DECODE['max_new_tokens'] and batches.proc.tokenizer.eos_token_id not in tokens))
   if start%128==0:event(out,'decode',completed=len(result),total=len(rows))
  assert [r['utt_id'] for r in result]==[r['utt_id'] for r in rows]
  summary=dict(**metrics(result),seconds=time.monotonic()-tick,truncated=sum(r['hit_token_cap'] for r in result),termination_note='Whisper removes terminal EOS then repads; lack of EOS in returned tokens alone is not truncation.');save(out/'predictions.json',result);save(out/'summary.json',summary);return summary
 finally:
  model.train(was);torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(cuda_rng)
def sequence_losses(model,batches,rows):
 inputs,labels=batches.train(rows)
 with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(**inputs,use_cache=False).logits
 losses=F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),labels.reshape(-1),ignore_index=-100,reduction='none').reshape(labels.shape)
 counts=labels.ne(-100).sum(1);assert torch.all(counts>0)
 return losses.sum(1)/counts

def smoke(config,out,batches,train):
 tick=time.monotonic();sample=[train[0],min(train,key=lambda r:len(r['token_ids'])),max(train,key=lambda r:len(r['token_ids'])),max(train,key=lambda r:r['actual_samples_16k'])]
 model=build(config);base_hash=digest(model,lambda n,p:'.lora_' not in n);frozen=digest(model,lambda n,p:not p.requires_grad);initial=digest(model,lambda n,p:'.lora_' in n)
 baseline=read(ROOT/'base_evaluation/predictions.json');raw_expected=[r['prediction'] for r in baseline[:4]]
 _,fresh_pred=generate(model,batches,read(ROOT/'dev.json')[:4]);assert fresh_pred==raw_expected,(fresh_pred,raw_expected)
 inputs,labels=batches.train(sample);prefix=batches.proc.tokenizer.prefix_tokens
 assert torch.all(labels[:,:len(prefix)-1]==-100)
 for i,r in enumerate(sample):
  n=len(r['token_ids']);assert inputs['decoder_input_ids'][i,0].item()==prefix[0]
  assert labels[i,n-2].item()==batches.proc.tokenizer.eos_token_id
  assert torch.all(labels[i,n-1:]==-100)
  assert labels[i].ne(-100).sum().item()==n-len(prefix)
 model.eval()
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
  direct=model(**inputs,labels=labels,use_cache=False)
  native=F.cross_entropy(direct.logits.reshape(-1,direct.logits.shape[-1]),labels.reshape(-1),ignore_index=-100)
  fp32=F.cross_entropy(direct.logits.float().reshape(-1,direct.logits.shape[-1]),labels.reshape(-1),ignore_index=-100)
  log_probs=F.log_softmax(direct.logits.float(),dim=-1)
  valid=labels.ne(-100)
  explicit=-log_probs.gather(-1,labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)[valid].mean()
 assert torch.allclose(direct.loss.float(),native,atol=1e-6,rtol=1e-6)
 assert torch.allclose(fp32,explicit,atol=1e-6,rtol=1e-6)
 ce_check=dict(builtin_native_input=float(direct.loss),native_input_manual=float(native),fp32_ce=float(fp32),explicit_fp32_logsoftmax_nll=float(explicit),note='Native BF16-input CE and explicitly FP32-input CE can differ slightly; actual training consistently uses FP32 CE.')
 del direct,native,fp32,explicit,log_probs,valid,inputs,labels
 model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
 params=[p for p in model.parameters() if p.requires_grad];optim=torch.optim.AdamW(params,lr=config['lr'],weight_decay=0)
 checks=[]
 for step in [1,2]:
  model.train();loss=sequence_losses(model,batches,sample).mean();loss.backward()
  enc=sum(float(p.grad.float().square().sum()) for n,p in model.named_parameters() if p.grad is not None and '.encoder.' in n)
  dec=sum(float(p.grad.float().square().sum()) for n,p in model.named_parameters() if p.grad is not None and '.decoder.' in n)
  assert enc>0 and (dec>0 if config['decoder'] else dec==0)
  assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
  norm=torch.nn.utils.clip_grad_norm_(params,1,error_if_nonfinite=True);optim.step();optim.zero_grad(set_to_none=True)
  checks.append(dict(step=step,loss=float(loss),encoder_gradient_l2=enc**.5,decoder_gradient_l2=dec**.5,gradient_norm=float(norm)))
  event(out,'smoke_loss',**checks[-1]);del loss
 assert digest(model,lambda n,p:not p.requires_grad)==frozen
 model.save_pretrained(out/'smoke-checkpoint');before,_=generate(model,batches,sample)
 changed=digest(model,lambda n,p:'.lora_' in n);assert changed!=initial
 model.gradient_checkpointing_disable();del model,optim,params;gc.collect();torch.cuda.empty_cache()
 # Fresh official base, fresh adapter structure, then saved weights only.
 restored=build(config);set_peft_model_state_dict(restored,load_file(str(out/'smoke-checkpoint/adapter_model.safetensors')))
 after,_=generate(restored,batches,sample);assert before==after
 assert digest(restored,lambda n,p:'.lora_' not in n)==base_hash
 assert digest(restored,lambda n,p:'.lora_' in n)==changed
 receipt=dict(passed=True,label_prefix=prefix,prefix_not_supervised=True,padding_ignored=True,eos_supervised=True,manual_native_loss_matches_builtin=True,fp32_training_CE_matches_explicit_NLL=True,ce_check=ce_check,
  checks=checks,frozen_parameters_unchanged=True,encoder_gradients_through_frozen_decoder=True,save_reload_token_ids_exact=True,
  initial_predictions_match_raw=True,fresh_initial_adapter_sha256=initial,raw_base_tensor_sha256=base_hash,
  trainable_parameters=sum(p.numel() for p in restored.parameters() if p.requires_grad),
  trainable_names=[n for n,p in restored.named_parameters() if p.requires_grad],seconds=time.monotonic()-tick)
 save(out/'smoke_passed.json',receipt);event(out,'smoke_complete',passed=True,seconds=receipt['seconds'])
 del restored;gc.collect();torch.cuda.empty_cache();return receipt

def validate_sources():
 p=read(ROOT/'protocol.json');assert all(sha(f)==h for f,h in p['source_sha256'].items());return p

def baseline():
 validate_sources();b=Batches();model=build();metric=evaluate(model,b,read(ROOT/'dev.json'),ROOT/'base_evaluation');save(ROOT/'baseline_complete.json',dict(passed=True,metrics=metric));event(ROOT,'baseline_complete',metrics=metric)

def worker(name):
 protocol=validate_sources();cfg=CONFIGS[name];out=ROOT/name;assert not (out/'training_started.json').exists()
 train,dev,orders=read(ROOT/'train.json'),read(ROOT/'dev.json'),read(ROOT/'orders.json');b=Batches()
 receipt=smoke(cfg,out,b,train)
 model=build(cfg);assert digest(model,lambda n,p:'.lora_' in n)==receipt['fresh_initial_adapter_sha256']
 frozen=digest(model,lambda n,p:not p.requires_grad)
 model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
 optim=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg['lr'],weight_decay=0)
 scheduler=get_linear_schedule_with_warmup(optim,num_warmup_steps=5,num_training_steps=201)
 set_seed(42);model.train();start=time.monotonic();observed=[];best=float('inf')
 save(out/'training_started.json',dict(config=cfg,initial_adapter_sha256=receipt['fresh_initial_adapter_sha256'],gpu=os.environ.get('CUDA_VISIBLE_DEVICES')))
 event(out,'training_started',config=cfg,steps=201,epochs=3,trainable_parameters=receipt['trainable_parameters'])
 for epoch,order in enumerate(orders):
  epoch_ids=[]
  for offset in range(0,2117,32):
   step=epoch*67+offset//32+1;global_rows=[train[i] for i in order[offset:offset+32]];tick=time.monotonic();loss_sum=0.;actual=[]
   for local in range(0,len(global_rows),4):
    batch=global_rows[local:local+4];loss=sequence_losses(model,b,batch).sum()/len(global_rows);loss.backward();loss_sum+=float(loss.detach());actual.extend(r['utt_id'] for r in batch);del loss
   assert actual==[r['utt_id'] for r in global_rows];epoch_ids.extend(actual);observed.extend(actual)
   norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1,error_if_nonfinite=True)
   lr=optim.param_groups[0]['lr'];optim.step();scheduler.step();optim.zero_grad(set_to_none=True)
   event(out,'loss',step=step,epoch=epoch+len(epoch_ids)/2117,loss=loss_sum,learning_rate=lr,gradient_norm=float(norm),actual_batch=len(actual),actual_audio_exposures=len(observed),seconds=time.monotonic()-tick)
   if step in EVAL_STEPS:
    cp=out/f'checkpoint-{step}';model.save_pretrained(cp)
    save(cp/'provenance.json',dict(step=step,config=cfg,base_revision=protocol['revision'],initial_adapter_sha256=receipt['fresh_initial_adapter_sha256'],adapter_sha256=sha(cp/'adapter_model.safetensors'),actual_audio_exposures=len(observed)))
    event(out,'evaluation_started',step=step);m=evaluate(model,b,dev,out/'evaluations'/f'step-{step}')
    event(out,'evaluation_complete',step=step,epoch=epoch+len(epoch_ids)/2117,metrics=m)
    if m['CER_percent']<best:best=m['CER_percent'];save(out/'best_checkpoint.json',dict(step=step,path=str(cp),metrics=m))
  assert epoch_ids==[train[i]['utt_id'] for i in order] and len(set(epoch_ids))==2117
  save(out/f'epoch-{epoch+1}-exposure.json',dict(epoch=epoch+1,utt_ids=epoch_ids,unique_audio=2117,actual_collator_order_verified=True))
  event(out,'epoch_coverage_verified',epoch=epoch+1,audio=2117)
 assert len(observed)==6351 and digest(model,lambda n,p:not p.requires_grad)==frozen
 validate_sources();save(out/'complete.json',dict(passed=True,epochs=3,steps=201,actual_audio_exposures=6351,all_epoch_orders_verified=True,frozen_parameters_unchanged=True,seconds=time.monotonic()-start,best=read(out/'best_checkpoint.json')))
 event(out,'complete',steps=201,epochs=3)

def supervise():
 assert (ROOT/'protocol.json').exists() and not (ROOT/'supervisor.json').exists()
 save(ROOT/'supervisor.json',dict(pid=os.getpid(),gpus=[2,3],queues={'2':['W-A','W-B'],'3':['W-C','W-D']}))
 def launch(mode,gpu,name=None):
  cmd=[sys.executable,'-u',str(Path(__file__).resolve()),mode]
  if name:cmd+=['--config',name]
  label=name or mode;f=(ROOT/f'{label}.log').open('w');p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1'),start_new_session=True)
  event(ROOT,'process_started',mode=mode,config=name,gpu=gpu,pid=p.pid);return p,f
 p,f=launch('baseline',2)
 try:
  code=p.wait();assert code==0,('baseline failed',code)
 finally:
  if p.poll() is None:p.terminate();p.wait()
  f.close()
 queues={2:['W-A','W-B'],3:['W-C','W-D']};running={}
 try:
  while any(queues.values()) or running:
   for gpu in queues:
    if gpu not in running and queues[gpu]:
     name=queues[gpu].pop(0);p,f=launch('worker',gpu,name);running[gpu]=(p,f,name)
   for gpu,(p,f,name) in list(running.items()):
    if p.poll() is not None:
     f.close();assert p.returncode==0 and read(ROOT/name/'complete.json')['passed'],(name,p.returncode)
     event(ROOT,'branch_complete',config=name,gpu=gpu);del running[gpu]
   time.sleep(2)
  # Strong common initialization and exposure-order cross-branch checks.
  checks=[read(ROOT/n/'smoke_passed.json') for n in CONFIGS]
  assert len({x['fresh_initial_adapter_sha256'] for x in checks})==1
  assert len({x['raw_base_tensor_sha256'] for x in checks})==1
  for e in range(1,4):assert len({sha(ROOT/n/f'epoch-{e}-exposure.json') for n in CONFIGS})==1
  save(ROOT/'complete.json',dict(passed=True,configs=list(CONFIGS),common_initialization_verified=True,all_exposure_orders_identical=True))
  event(ROOT,'experiment_complete')
 finally:
  for p,f,name in running.values():
   if p.poll() is None:p.terminate()
  for p,f,name in running.values():p.wait();f.close()
if __name__=='__main__':
 torch.set_num_threads(2)
 parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','baseline','worker','supervise']);parser.add_argument('--config',choices=CONFIGS);args=parser.parse_args()
 try:
  if args.mode=='worker':worker(args.config)
  else:globals()[args.mode]()
 except Exception as exc:
  event(ROOT if not args.config else ROOT/args.config,'failure',error=repr(exc),mode=args.mode);raise
