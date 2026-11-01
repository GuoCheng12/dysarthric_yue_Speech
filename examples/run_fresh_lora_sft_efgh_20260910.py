"""Fresh E/F/G/H SFT: two learning rates crossed with full versus audio-only LoRA."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parents[1]
ROOT = Path('/data/qwen3-asr/records/therapist_mas_setting_d_v1/qwen3_asr_fresh_lora_sft_seed42_EFGH_20260910_v1')
BASE = Path('/data/qwen3-asr/models/Qwen3-ASR-1.7B')
TRAIN = Path('/data/qwen3-asr/finetune/data_setting_d_v1/setting_d_train.jsonl')
DEV = Path('/data/qwen3-asr/finetune/data_setting_d_merged_test_v1/setting_d_merged_test.jsonl')
for name in ('src', 'finetune/scripts', 'finetune/official'):
    sys.path.insert(0, str(REPO / name))
os.environ.setdefault('TMPDIR', str(ROOT / 'tmp'))
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('OMP_NUM_THREADS', '3')
os.environ.setdefault('MKL_NUM_THREADS', '3')

import torch
import torch.nn.functional as F
from torch.utils.data import Sampler, SequentialSampler
from transformers import TrainingArguments, TrainerCallback, set_seed
from qwen3_asr_lora_sft import QwenASRLoRATrainer, DEFAULT_TARGET_MODULES
from qwen3_asr_sft import DataCollatorForQwen3ASRFinetuning, make_preprocess_fn_prefix_only, patch_outer_forward

CONFIGS = {
    'E_full_lr5em5': dict(alpha=4, lr=5e-5, audio_lora=True, language_lora=True, objective='utterance'),
    'F_full_lr2em5': dict(alpha=4, lr=2e-5, audio_lora=True, language_lora=True, objective='utterance'),
    'G_audio_lr5em5': dict(alpha=4, lr=5e-5, audio_lora=True, language_lora=False, objective='utterance'),
    'H_audio_lr2em5': dict(alpha=4, lr=2e-5, audio_lora=True, language_lora=False, objective='utterance'),
}

EVAL_STEPS = [epoch*67 + offset for epoch in range(3) for offset in (14,27,40,54,67)]


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    pending.replace(path)


def event(out, stage, **data):
    record = dict(utc=datetime.now(timezone.utc).isoformat(), stage=stage, **data)
    out.mkdir(parents=True, exist_ok=True)
    with (out/'events.jsonl').open('a') as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(json.dumps(record, ensure_ascii=False), flush=True)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while chunk := handle.read(2**20): h.update(chunk)
    return h.hexdigest()


def tensor_digest(items):
    h = hashlib.sha256()
    for name, value in sorted(items):
        h.update(name.encode())
        h.update(value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()


def base_digest(model):
    return tensor_digest((name.removeprefix('base_model.model.').replace('.base_layer.', '.'), p)
        for name,p in model.named_parameters() if '.lora_' not in name)


def build(config):
    from peft import LoraConfig, get_peft_model
    from qwen_asr import Qwen3ASRModel
    set_seed(42)
    wrapper = Qwen3ASRModel.from_pretrained(str(BASE), dtype=torch.bfloat16,
        device_map='cuda:0', max_inference_batch_size=8, max_new_tokens=256)
    wrapper.model.requires_grad_(False)
    patch_outer_forward(wrapper.model)
    base_hash = base_digest(wrapper.model)
    # No adapter is loaded. Every recipe starts with the raw pretrained model.
    wrapper.model = get_peft_model(wrapper.model, LoraConfig(r=16, lora_alpha=config['alpha'],
        lora_dropout=.05, bias='none', target_modules=DEFAULT_TARGET_MODULES.split(',')))
    if not config['language_lora']:
        # Keep identical initial audio A matrices and dropout RNG consumption.
        # Frozen language adapters remain B=0; gradients still flow through the
        # frozen language model to the audio adapters (no detach/no_grad here).
        for name,p in wrapper.model.named_parameters():
            if 'audio_tower' not in name: p.requires_grad_(False)
    assert any(p.requires_grad and 'audio_tower' in name for name,p in wrapper.model.named_parameters())
    assert all(config['language_lora'] or 'audio_tower' in name
               for name,p in wrapper.model.named_parameters() if p.requires_grad)
    assert all(torch.count_nonzero(p).item()==0 for n,p in wrapper.model.named_parameters() if '.lora_B.' in n)
    assert all('.lora_' in n for n,p in wrapper.model.named_parameters() if p.requires_grad)
    wrapper.model.config.use_cache = False
    return wrapper, base_hash


def collator_for(wrapper, trace=False):
    collator = DataCollatorForQwen3ASRFinetuning(wrapper.processor)
    preprocess = make_preprocess_fn_prefix_only(wrapper.processor)
    def collate(data):
        result = collator([preprocess(r) for r in data])
        if trace: result['_audit_ids'] = [r['utt_id'] for r in data]
        return result
    return collate


class EpochSampler(Sampler):
    def __init__(self, orders):
        self.orders, self.epoch = orders, 0
    def set_epoch(self, epoch): self.epoch = epoch
    def __len__(self): return len(self.orders[0])
    def __iter__(self): return iter(self.orders[self.epoch])


class RecipeTrainer(QwenASRLoRATrainer):
    def __init__(self, *args, objective='utterance', orders=None, **kwargs):
        self.objective, self.orders = objective, orders
        self.exposures = []
        super().__init__(*args, **kwargs)
    def _get_train_sampler(self, train_dataset=None):
        return EpochSampler(self.orders) if self.orders else SequentialSampler(self.train_dataset)
    def _get_num_items_in_batch(self, batches, device):
        if self.objective=='utterance': return super()._get_num_items_in_batch(batches, device)
        return sum(int(b['labels'][:,1:].ne(-100).sum()) for b in batches)
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.objective=='utterance':
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        labels = inputs['labels'][:,1:]
        outputs = model(**{k:v for k,v in inputs.items() if k!='labels'})
        logits = outputs['logits'][:,:-1].float()
        count = labels.ne(-100).sum() if num_items_in_batch is None else num_items_in_batch
        loss = F.cross_entropy(logits.reshape(-1,logits.shape[-1]), labels.reshape(-1),
            ignore_index=-100, reduction='sum') / count
        return (loss,outputs) if return_outputs else loss
    def training_step(self, model, inputs, num_items_in_batch=None):
        self.exposures.extend(inputs.pop('_audit_ids', []))
        return super().training_step(model, inputs, num_items_in_batch)


def decode(wrapper, data, out):
    from therapist_harness.text import text_normalize, edit_distance
    wrapper.model.eval()
    wrapper.model.config.use_cache = True
    result = []
    started = time.monotonic()
    for i in range(0,len(data),8):
        batch = data[i:i+8]
        decoded = wrapper.transcribe(audio=[r['audio'] for r in batch], language=['Cantonese']*len(batch))
        for row,pred in zip(batch,decoded,strict=True):
            gt = text_normalize(row['clean_gt'])
            result.append(dict(utt_id=row['utt_id'], speaker_id=row['speaker_id'],
                zero_shot_bucket=row['zero_shot_bucket'], prediction=pred.text,
                errors=edit_distance(gt,text_normalize(pred.text)), chars=len(gt)))
        if i%128==0: event(out,'decode',completed=len(result),total=len(data))
    errors,chars = sum(r['errors'] for r in result),sum(r['chars'] for r in result)
    def metric(xs):
        return dict(audio=len(xs),errors=sum(r['errors'] for r in xs),chars=sum(r['chars'] for r in xs),
            CER_percent=100*sum(r['errors'] for r in xs)/sum(r['chars'] for r in xs))
    summary = dict(**metric(result), seconds=time.monotonic()-started,
        vlink247=metric([r for r in result if r['speaker_id']=='vlink247']) if len(data)==511 else None,
        other=metric([r for r in result if r['speaker_id']!='vlink247']),
        **{bucket:metric([r for r in result if r['zero_shot_bucket']==bucket])
           for bucket in ('easy','medium','hard') if any(r['zero_shot_bucket']==bucket for r in result)})
    save(out/'predictions.json',result)
    save(out/'summary.json',summary)
    return summary


def merged_evaluation(wrapper, data, out):
    from peft.tuners.lora.layer import LoraLayer
    model = wrapper.model
    was_training, use_cache = model.training,model.config.use_cache
    layers = [m for m in model.modules() if isinstance(m,LoraLayer)]
    assert all(not m.merged for m in layers)
    # Subtraction during BF16 unmerge does not recover all original bits.
    # Preserve and restore each affected base weight exactly on the same GPU.
    originals = [(m.get_base_layer().weight,m.get_base_layer().weight.detach().clone()) for m in layers]
    try:
        model.merge_adapter()
        summary = decode(wrapper,data,out)
    finally:
        model.unmerge_adapter()
        with torch.no_grad():
            for weight,original in originals: weight.copy_(original)
        assert all(torch.equal(weight,original) for weight,original in originals)
        model.train(was_training)
        model.config.use_cache = use_cache
    del originals
    torch.cuda.empty_cache()
    save(out/'restore_verified.json',dict(base_weights_restored_exactly=True,
        evaluation='BF16 merged LoRA, greedy Cantonese, empty context, max_new_tokens=256'))
    return summary


def prepare():
    ROOT.mkdir(parents=True,exist_ok=False)
    (ROOT/'tmp').mkdir()
    train,dev = rows(TRAIN),rows(DEV)
    assert len(train)==2117 and len(dev)==511
    assert len({r['utt_id'] for r in train})==2117
    assert not {Path(r['audio']).resolve() for r in train} & {Path(r['audio']).resolve() for r in dev}
    assert all(r['text'].startswith('language Cantonese<asr_text>') and not r['prompt'] for r in train)
    audio_hashes = {r['audio']:sha(r['audio']) for r in train+dev}
    assert not {audio_hashes[r['audio']] for r in train} & {audio_hashes[r['audio']] for r in dev}
    orders = []
    for epoch in range(3):
        indices = list(range(2117));random.Random(42+epoch).shuffle(indices);orders.append(indices)
    sample = [train[i] for i in orders[0][:3]] + [max(train,key=lambda r:float(r['duration']))]
    panel = [train[i] for i in orders[0][:128]]
    for name,data in [('train',train),('dev',dev),('orders',orders),('smoke_audio',sample),('train_panel',panel)]:
        save(ROOT/f'{name}.json',data)
    save(ROOT/'audio_sha256.json',audio_hashes)
    sources = [Path(__file__), REPO/'finetune/scripts/qwen3_asr_lora_sft.py',
        REPO/'finetune/official/qwen3_asr_sft.py',REPO/'src/therapist_harness/text.py']
    for p in sources:
        dest = ROOT/'sources'/p.relative_to(REPO);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
    save(ROOT/'protocol.json',dict(base_model=str(BASE),initial_adapter=None,resume_from=None,
        initialization='Raw Qwen3-ASR-1.7B plus freshly initialized LoRA, B=0; never load or merge R-only.',
        seed=42,configs=CONFIGS,rank=16,dropout=.05,epochs=3,real_audio=2117,
        audio_exposures_per_config=6351,steps_per_epoch=67,total_steps=201,batch=4,grad_acc=8,
        schedule='linear',warmup_ratio=.02,weight_decay=0,max_grad_norm=1,
        evaluation_steps=EVAL_STEPS,dev_audio=511,dev_is_repeated_feedback=True,
        primary_evaluation='BF16 merged, greedy Cantonese, empty context, cap256; exact base restore before more training',
        secondary_evaluation='unmerged at final epoch only',
        selection='Report each full-epoch endpoint plus best checkpoint with at least one complete epoch (step>=67).',
        contrast='E/F and G/H change only learning rate; E/G and F/H change only whether language LoRA is trainable.',
        frozen_zero_language_adapters_in_GH='Inserted to match initial audio A matrices and dropout RNG; B remains zero and all language weights stay frozen.',
        previous_experiment='qwen3_asr_fresh_lora_sft_seed42_4configs_20260910_v1',
        disk_estimate_additional_GB=5.4,
        source_sha256={str(p):sha(p) for p in sources},
        input_sha256={str(p):sha(p) for p in [TRAIN,DEV,ROOT/'orders.json',ROOT/'audio_sha256.json']},
        base_files_sha256={str(p):sha(p) for p in BASE.iterdir() if p.is_file()}))
    event(ROOT,'prepared',configs=list(CONFIGS),audio=2117,epochs=3,total_steps=201)


def smoke(name):
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    cfg = CONFIGS[name]
    out = ROOT/name/'smoke'
    wrapper,digest = build(cfg)
    initial = tensor_digest((n,p) for n,p in wrapper.model.named_parameters() if '.lora_' in n)
    data = read(ROOT/'smoke_audio.json')
    frozen_lora_initial = tensor_digest((n,p) for n,p in wrapper.model.named_parameters()
        if '.lora_' in n and not p.requires_grad)
    audio_initial = tensor_digest((n,p) for n,p in wrapper.model.named_parameters()
        if '.lora_' in n and 'audio_tower' in n)
    gradient_checks = []
    class CheckGradients(TrainerCallback):
        def on_pre_optimizer_step(self,args,state,control,model=None,**kwargs):
            measured = {}
            for label,is_audio in [('audio',True),('language',False)]:
                params = [(n,p) for n,p in model.named_parameters()
                    if '.lora_' in n and ('audio_tower' in n)==is_audio]
                gradients = [p.grad for n,p in params if p.grad is not None]
                assert all(torch.isfinite(g).all() for g in gradients)
                norm2 = sum(float(g.float().square().sum()) for g in gradients)
                measured[label] = dict(tensors_with_gradient=len(gradients),gradient_l2=norm2**.5)
                if is_audio or cfg['language_lora']: assert norm2>0
                else: assert not gradients
            assert all(p.grad is None for n,p in model.named_parameters() if '.lora_' not in n)
            gradient_checks.append(measured)
    trainer = RecipeTrainer(model=wrapper.model, objective=cfg['objective'],
        args=TrainingArguments(output_dir=str(out),per_device_train_batch_size=4,gradient_accumulation_steps=1,
            max_steps=2,learning_rate=cfg['lr'],lr_scheduler_type='constant',warmup_ratio=0,
            bf16=True,save_strategy='no',logging_steps=1,report_to='none',remove_unused_columns=False,
            dataloader_num_workers=0,disable_tqdm=True,seed=42,data_seed=42),
        train_dataset=data,data_collator=collator_for(wrapper,trace=True),callbacks=[CheckGradients()])
    trainer.train()
    assert trainer.state.global_step==2 and len(gradient_checks)==2
    assert tensor_digest((n,p) for n,p in wrapper.model.named_parameters()
        if '.lora_' in n and not p.requires_grad)==frozen_lora_initial
    assert tensor_digest((n,p) for n,p in wrapper.model.named_parameters()
        if '.lora_' in n and 'audio_tower' in n)!=audio_initial
    assert tensor_digest((n,p) for n,p in wrapper.model.named_parameters() if '.lora_' in n)!=initial
    assert all(torch.isfinite(p).all() for n,p in wrapper.model.named_parameters() if '.lora_' in n)
    inputs = collator_for(wrapper)(data[:1])
    inputs = {k:v.to('cuda',dtype=torch.bfloat16 if v.is_floating_point() else v.dtype) for k,v in inputs.items()}
    wrapper.model.eval()
    def logits():
        with torch.no_grad():return wrapper.model(**{k:v for k,v in inputs.items() if k!='labels'}).logits.detach().cpu()
    before = logits()
    merged_evaluation(wrapper,data,out/'merged_decode')
    after = logits()
    assert torch.equal(before,after),'Temporary merged evaluation changed training function'
    assert base_digest(wrapper.model)==digest
    wrapper.model.save_pretrained(out/'adapter')
    loaded = load_file(out/'adapter/adapter_model.safetensors')
    expected = get_peft_model_state_dict(wrapper.model)
    assert set(loaded)==set(expected) and all(torch.equal(loaded[k],v.cpu()) for k,v in expected.items())
    save(ROOT/name/'smoke_passed.json',dict(passed=True,raw_base_tensor_sha256=digest,
        fresh_initial_adapter_sha256=initial,steps=2,base_frozen=True,
        gradient_checks=gradient_checks,frozen_lora_unchanged=True,audio_lora_updated=True,
        trainable_parameters=sum(p.numel() for p in wrapper.model.parameters() if p.requires_grad),
        merged_evaluation_restores_logits_exactly=True,adapter_serialization_exact=True,
        longest_audio_seconds=max(float(r['duration']) for r in data),
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated()))
    event(ROOT/name,'smoke_complete',passed=True)


def train(name):
    cfg = CONFIGS[name]
    out = ROOT/name
    protocol = read(ROOT/'protocol.json')
    assert all(sha(p)==h for p,h in protocol['source_sha256'].items())
    assert all(read(ROOT/b/'smoke_passed.json')['passed'] for b in CONFIGS)
    wrapper,digest = build(cfg)
    initial = tensor_digest((n,p) for n,p in wrapper.model.named_parameters() if '.lora_' in n)
    assert initial==read(out/'smoke_passed.json')['fresh_initial_adapter_sha256']
    assert digest==read(out/'smoke_passed.json')['raw_base_tensor_sha256']
    save(out/'initialization.json',dict(base_model=str(BASE),initial_adapter=None,zero_initial_B=True,
        raw_base_tensor_sha256=digest,initial_adapter_sha256=initial,
        trainable_parameters=sum(p.numel() for p in wrapper.model.parameters() if p.requires_grad),
        trainable_names=[n for n,p in wrapper.model.named_parameters() if p.requires_grad]))
    train_rows,orders = read(ROOT/'train.json'),read(ROOT/'orders.json')
    frozen_lora_initial = tensor_digest((n,p) for n,p in wrapper.model.named_parameters()
        if '.lora_' in n and not p.requires_grad)
    if name=='E_full_lr5em5':
        base_metrics=merged_evaluation(wrapper,read(ROOT/'dev.json'),ROOT/'base_evaluation')
        event(ROOT,'raw_base_evaluation',**base_metrics)
    class Monitor(TrainerCallback):
        trainer = None
        def on_log(self,args,state,control,logs=None,**kwargs):
            if logs:event(out,'loss',step=state.global_step,**logs)
        def on_step_end(self,args,state,control,**kwargs):
            if state.global_step not in EVAL_STEPS:return
            step = state.global_step
            checkpoint = out/f'checkpoint-{step}'
            wrapper.model.save_pretrained(checkpoint)
            save(checkpoint/'provenance.json',dict(config=cfg,base_model=str(BASE),initial_adapter=None,
                step=step,epoch=state.epoch,actual_audio_exposures=len(self.trainer.exposures),
                raw_base_tensor_sha256=digest,source_sha256=protocol['source_sha256']))
            cpu_rng=torch.random.get_rng_state()
            cuda_rng=torch.cuda.get_rng_state()
            start=time.monotonic()
            metrics=merged_evaluation(wrapper,read(ROOT/'dev.json'),out/'evaluations'/f'step-{step}')
            torch.random.set_rng_state(cpu_rng);torch.cuda.set_rng_state(cuda_rng)
            event(out,'evaluation_complete',step=step,epoch=state.epoch,
                actual_audio_exposures=len(self.trainer.exposures),metrics=metrics,seconds=time.monotonic()-start)
        def on_epoch_end(self,args,state,control,**kwargs):
            epoch=int(round(state.epoch))
            observed=self.trainer.exposures[(epoch-1)*2117:epoch*2117]
            expected=[train_rows[i]['utt_id'] for i in orders[epoch-1]]
            assert observed==expected and len(set(observed))==2117
            save(out/f'epoch-{epoch}-exposure.json',dict(epoch=epoch,audio=2117,unique_audio=2117,
                utt_ids=observed,shared_order_exact=True))
            event(out,'epoch_coverage_verified',epoch=epoch,audio=2117,unique_audio=2117)
    monitor=Monitor()
    trainer=RecipeTrainer(model=wrapper.model,objective=cfg['objective'],orders=orders,
        args=TrainingArguments(output_dir=str(out),per_device_train_batch_size=4,gradient_accumulation_steps=8,
            num_train_epochs=3,learning_rate=cfg['lr'],lr_scheduler_type='linear',warmup_ratio=.02,
            weight_decay=0,max_grad_norm=1,bf16=True,seed=42,data_seed=42,dataloader_num_workers=0,
            save_strategy='no',logging_steps=1,report_to='none',remove_unused_columns=False,disable_tqdm=True),
        train_dataset=train_rows,data_collator=collator_for(wrapper,trace=True),callbacks=[monitor])
    monitor.trainer=trainer
    event(out,'training_started',initial_model='raw Qwen3-ASR + fresh LoRA',steps=201,audio=2117,epochs=3)
    started=time.monotonic()
    trainer.train()
    assert trainer.state.global_step==201 and len(trainer.exposures)==6351
    assert base_digest(wrapper.model)==digest
    assert tensor_digest((n,p) for n,p in wrapper.model.named_parameters()
        if '.lora_' in n and not p.requires_grad)==frozen_lora_initial
    if not cfg['language_lora']:
        assert all(torch.count_nonzero(p).item()==0 for n,p in wrapper.model.named_parameters()
            if 'audio_tower' not in n and '.lora_B.' in n)
    torch.save(trainer.optimizer.state_dict(),out/'final_optimizer.pt')
    trainer.state.save_to_json(str(out/'trainer_state.json'))
    # One secondary final decode quantifies the previously observed merge effect.
    metrics=decode(wrapper,read(ROOT/'dev.json'),out/'final_unmerged_evaluation')
    assert all(sha(p)==h for p,h in protocol['source_sha256'].items())
    save(out/'complete.json',dict(passed=True,steps=201,epochs=3,actual_audio_exposures=6351,
        all_epoch_orders_verified=True,base_unchanged=True,frozen_lora_unchanged=True,seconds=time.monotonic()-started,
        final_unmerged_metrics=metrics))
    event(out,'complete',steps=201,epochs=3)


def supervise():
    assert ROOT.exists() and not (ROOT/'supervisor.json').exists()
    save(ROOT/'supervisor.json',dict(pid=os.getpid()))
    handles=[]
    def launch(mode,name,gpu):
        handle=(ROOT/f'{name}_{mode}.log').open('w');handles.append(handle)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),TMPDIR=str(ROOT/'tmp'),
            PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='3',MKL_NUM_THREADS='3')
        p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),mode,'--config',name],
            stdout=handle,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        event(ROOT,'process_started',mode=mode,config=name,gpu=gpu,pid=p.pid)
        return p
    for mode in ('smoke','train'):
        processes=[launch(mode,name,gpu) for gpu,name in enumerate(CONFIGS)]
        while any(p.poll() is None for p in processes):
            failed=[p for p in processes if p.poll() not in (None,0)]
            if failed:
                for p in processes:
                    if p.poll() is None:p.terminate()
                raise RuntimeError(f'{mode} worker failed; other grid workers stopped')
            time.sleep(3)
        assert all(p.returncode==0 for p in processes)
        if mode=='smoke':
            receipts=[read(ROOT/n/'smoke_passed.json') for n in CONFIGS]
            assert len({r['raw_base_tensor_sha256'] for r in receipts})==1
            assert len({r['fresh_initial_adapter_sha256'] for r in receipts})==1
            save(ROOT/'common_fresh_initialization_verified.json',dict(passed=True,
                raw_base_tensor_sha256=receipts[0]['raw_base_tensor_sha256'],
                fresh_initial_adapter_sha256=receipts[0]['fresh_initial_adapter_sha256']))
    save(ROOT/'complete.json',dict(passed=True,configs=list(CONFIGS),epochs=3))
    event(ROOT,'grid_complete')


def main():
    import accelerate.utils.other as other
    other.is_deepspeed_available=lambda:False
    torch.set_num_threads(3)
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['prepare','smoke','train','supervise'])
    parser.add_argument('--config',choices=CONFIGS)
    args=parser.parse_args()
    try:
        if args.mode=='prepare':prepare()
        elif args.mode=='supervise':supervise()
        elif args.mode=='smoke':smoke(args.config)
        else:train(args.config)
    except Exception as exc:
        event(ROOT,'failure',mode=args.mode,config=args.config,error=repr(exc))
        raise


if __name__=='__main__':main()
