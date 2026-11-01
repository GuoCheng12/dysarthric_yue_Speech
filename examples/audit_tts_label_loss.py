"""CPU teacher-forcing audit: same recognized audio, only terminal period differs."""
from contextlib import nullcontext
from pathlib import Path
import json
import statistics

import torch
import torch.nn.functional as F
from peft import PeftModel

from tts_additive_control import training,rows,save,sha256_file
from diagnose_tts_utility import ROOT,ADAPTER


def run():
    torch.set_num_threads(8)
    source=[r for r in rows(ROOT/'screening/audit.jsonl') if r['variant']=='mas_hard' and r['cer']==0]
    selected=[];speakers=set();texts=set()
    for r in source:
        if r['speaker_id'] not in speakers and r['clean_gt'] not in texts:
            selected.append(r);speakers.add(r['speaker_id']);texts.add(r['clean_gt'])
        if len(selected)==16:break
    assert len(selected)==16
    save(ROOT/'label_loss/protocol.json',{'selection':'first16 zero-CER MAS-hard rows, distinct patients and texts',
        'purpose':'format-only loss decomposition, not a population estimate of all TTS quality',
        'adapter':str(ADAPTER),'rows':[r['utt_id'] for r in selected],
        'source_sha256':sha256_file(Path(__file__))})
    dtype=torch.bfloat16 if torch.cpu._is_avx512_bf16_supported() else torch.float32
    wrapper=training.Qwen3ASRModel.from_pretrained('/data/qwen3-asr/models/Qwen3-ASR-1.7B',
        dtype=dtype,device_map='cpu')
    training.patch_outer_forward(wrapper.model)
    model=PeftModel.from_pretrained(wrapper.model,str(ADAPTER),is_trainable=False).eval()
    processor=wrapper.processor
    collator=training.DataCollatorForQwen3ASRFinetuning(processor=processor,sampling_rate=16000)
    preprocess=training.make_preprocess_fn_prefix_only(processor)
    outputs=[]
    for i,row in enumerate(selected):
        for model_name in ['base','real_tuned']:
            for style in ['period','no_period']:
                target=row['clean_gt'] if style=='period' else row['clean_gt'].rstrip('。！？.!?')
                batch=collator([preprocess({'audio':row['audio'],'text':'language Cantonese<asr_text>'+target,'prompt':''})])
                labels=batch.pop('labels');batch={k:v.to(dtype=dtype) if torch.is_tensor(v) and v.is_floating_point() else v for k,v in batch.items()}
                context=model.disable_adapter() if model_name=='base' else nullcontext()
                with torch.inference_mode(),context:
                    logits=model(**batch)['logits'][:,:-1].float()
                    targets=labels[:,1:]
                    losses=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),targets.reshape(-1),ignore_index=-100,reduction='none').view_as(targets)
                    mask=targets.ne(-100)
                    tokens=targets[mask].tolist();values=losses[mask].tolist()
                pieces=[processor.tokenizer.decode([t]) for t in tokens]
                record={'utt_id':row['utt_id'],'model':model_name,'style':style,'target':target,
                    'mean_loss':statistics.mean(values),'loss_sum':sum(values),
                    'tokens':[{'id':t,'text':p,'loss':v} for t,p,v in zip(tokens,pieces,values)],
                    'period_loss':sum(v for p,v in zip(pieces,values) if '。' in p),
                    'eos_loss':sum(v for t,v in zip(tokens,values) if t==processor.tokenizer.eos_token_id)}
                outputs.append(record)
                del logits,losses,batch
        print(f'[label-loss] {i+1}/16',flush=True)
    groups={}
    for m in ['base','real_tuned']:
        for s in ['period','no_period']:
            rr=[r for r in outputs if r['model']==m and r['style']==s]
            groups[m+'_'+s]={k:statistics.mean(r[k] for r in rr) for k in ['mean_loss','loss_sum','period_loss','eos_loss']}
            groups[m+'_'+s]['period_fraction_of_loss']=sum(r['period_loss'] for r in rr)/sum(r['loss_sum'] for r in rr)
    save(ROOT/'label_loss/results.json',{'groups':groups,'rows':outputs})
    print(json.dumps(groups,indent=2),flush=True)


if __name__=='__main__':run()
