import os,sys,json,time,subprocess,traceback
from pathlib import Path
R=Path(__file__).resolve().parent
OLD=R.parent/'ucg_uniform_vs_v8_seed42_590_4gpu_20260913_v2'
os.environ.update(TMPDIR=str(R/'tmp'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='3',TOKENIZERS_PARALLELISM='false')
read=lambda p:json.loads(p.read_text())
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def status(**kw):
 d=dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw);save(R/'sequence_status.json',d);print(json.dumps(d),flush=True)
def call(mode,phase):
 status(stage='running',mode=mode,phase=phase,branch='mas',gpus=[0,1,2,3])
 subprocess.run([sys.executable,'-u',str(R/'runtime/examples/run_setting_e_sampling_curriculum_20260911.py'),mode,'--protocol',str(R/phase/'protocol.json'),'--branch','mas','--gpus','0,1,2,3'],check=True)
def verify(phase,n):
 p=R/phase/'mas';d=read(p/'complete.json');assert d['passed'] and d['steps']==n and d['world_size']==4 and d['actual_audio_exposures']==n*32
 assert read(p/'initialization.json')==read(OLD/'formal/mas/initialization.json')==read(OLD/'formal/uniform/initialization.json')
 receipts=list(p.glob('stage-*/**/receipt.json'));assert len(receipts)==(10 if phase=='smoke' else 105)
 assert all(read(f)['validated'] and read(f)['response_model']=='gpt-5.6-sol' for f in receipts)
 packs=list(p.glob('stage-*/decision/candidates.json'))
 assert len(packs)==(2 if phase=='smoke' else 21)
 assert all('ucg_material_task' in read(f) and all('dev_patient_evidence' in x for x in read(f)['focuses']) for f in packs)
 save(p/'material_adaptation_verified.json',dict(passed=True,steps=n,actual_api_calls=len(receipts),same_initialization=True,ucg_material_context_present=True))
def report():
 result={}
 for name,p in [('ordinary',OLD/'formal/uniform'),('previous_V8',OLD/'formal/mas'),('UCG_material_V8',R/'formal/mas')]:
  ev=[json.loads(l) for l in (p/'events.jsonl').read_text().splitlines()];es=[e for e in ev if e['stage']=='evaluation_complete'];s=read(p/'selected_checkpoint.json')['step']
  result[name]=dict(source=str(p.resolve()),dev_selected_step=s,dev_selected=next(e for e in es if e['step']==s),endpoint=es[-1],all_evaluations=es,retrospective_test_minimum=min(es,key=lambda e:e['test']['CER_percent']),timing=read(p/'complete.json')['timing'])
 save(R/'comparison.json',result)
try:
 assert not (R/'started.json').exists();save(R/'started.json',dict(pid=os.getpid(),started_unix=time.time()))
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 for phase in ['smoke','formal']:call('prepare',phase);call('cpu-smoke',phase)
 call('smoke','smoke');verify('smoke',4)
 call('supervise','formal');verify('formal',590)
 report();status(stage='complete',results=str(R/'comparison.json'))
except BaseException as e:status(stage='failed',error_type=type(e).__name__,detail=str(e));traceback.print_exc();raise
