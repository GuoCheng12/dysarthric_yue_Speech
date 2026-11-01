"""Repeat the historical one-step S32 improvement from three matched R models."""
import argparse
from pathlib import Path
import sys

from tts_additive_control import training,Monitor,repeated_permutations,ROOT as ADDITIVE
from tts_increment_control import REPO,ROOT as CONTROL,TEST,rows,read,jsonl,sha256_file,write_immutable_json as save
from torch.utils.data import SequentialSampler

ROOT=ADDITIVE.parent/'post_small'
SOURCE=CONTROL.parent/'m0_absorption_factorial_v1/s032_r000'


def run(arm,seed):
    directory=ROOT/arm/f'seed_{seed}';assert not directory.exists()
    parent=ADDITIVE/'real_w0'/f'seed_{seed}/training/checkpoint-50'
    assert (parent/'adapter_model.safetensors').exists()
    if arm=='synthetic32':
        selected=rows(SOURCE/'train.jsonl');assert len(selected)==32
        old=read(SOURCE/'config.json')
        for path,digest in old['audio_sha256'].items():assert sha256_file(Path(path))==digest
        source=SOURCE/'train.jsonl';selection='same32 original audio and labels from historical one-step positive probe'
    else:
        source=CONTROL/'real_q/train.jsonl';real=rows(source)
        indexes=repeated_permutations(len(real),1632,seed)[1600:]
        selected=[real[i] for i in indexes]
        selection='next32 real examples after the matched R50 training sequence; one extra optimizer update'
    jsonl(directory/'train.jsonl',selected)
    command=read(SOURCE/'config.json')['command'].copy()
    changes={'--train_file':str(directory/'train.jsonl'),'--output_dir':str(directory/'training'),
        '--lora_init_adapter_path':str(parent),'--max_steps':'1','--seed':str(seed)}
    for key,value in changes.items():command[command.index(key)+1]=value
    paths=[Path(__file__),Path(training.__file__),REPO/'finetune/official/qwen3_asr_sft.py',
        REPO/'finetune/scripts/train_m0_small_dose_pair.py',source,TEST,
        parent/'adapter_model.safetensors',parent/'adapter_config.json',directory/'train.jsonl']
    cfg={'arm':arm,'seed':seed,'parent':str(parent),'steps':1,'lr':2e-5,'rows':32,
        'optimizer':'fresh AdamW in both continuations; parent weights retained',
        'selection':selection,'command':command,'interpretation':'historical recipe replication, fixed dataset and endpoint; no new checkpoint sweep',
        'hashes':{str(p):sha256_file(p) for p in paths},
        'audio_sha256':{r['audio']:sha256_file(Path(r['audio'])) for r in selected}}
    save(directory/'protocol.json',cfg)
    for name in ['hashes','audio_sha256']:
        for path,digest in cfg[name].items():assert sha256_file(Path(path))==digest

    class OneStepMonitor(Monitor):
        def on_step_end(self,args,state,control,**kwargs):
            control.should_save=state.global_step==1
            control.should_training_stop=state.global_step>=1
            return control

    monitor=OneStepMonitor(directory);monitor.end_step=1;monitor.state['endpoint_step']=1;monitor.publish()
    original=training.QwenASRLoRATrainer
    class OneStepTrainer(original):
        def _get_train_sampler(self,train_dataset=None):
            return SequentialSampler(self.train_dataset if train_dataset is None else train_dataset)
        def __init__(self,*args,**kwargs):
            kwargs['args'].save_only_model=True
            super().__init__(*args,**kwargs);self.add_callback(monitor);monitor.trainer=self
    training.QwenASRLoRATrainer=OneStepTrainer;sys.argv=command[1:]
    try:
        training.main();assert len(monitor.history)==1 and monitor.history[0]['step']==1
        save(directory/'result.json',{'arm':arm,'seed':seed,'parent':str(parent),'step':1,
            'metrics':monitor.history[0]['metrics']})
        monitor.state['status']='complete';monitor.publish()
    except Exception as error:
        monitor.state.update(status='failed',error=f'{type(error).__name__}: {error}');monitor.publish();raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['synthetic32','real32'],required=True)
    p.add_argument('--seed',type=int,required=True);args=p.parse_args();run(args.arm,args.seed)
