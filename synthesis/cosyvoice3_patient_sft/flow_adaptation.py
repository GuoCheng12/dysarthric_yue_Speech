"""One bounded Setting D Flow-only experiment; original SG artifacts stay immutable."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from safetensors.torch import load_file, save_file

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / 'src'), str(REPO / 'examples')]
ROOT = Path('/data/qwen3-asr/records/cosyvoice3_flow_sft_setting_d_v1')
DATA = Path('/data/qwen3-asr/records/voxcpm2_reference_sft_setting_d_v1/data.json')
SG = Path('/data/qwen3-asr/records/cosyvoice3_reference_sft_setting_d_v1')
CONFIG = dict(epochs=3, seed=42, micro_batch=2, effective_batch=8, lr=1e-5,
              clip_norm=5., weight_decay=0., streaming=False, train_cfg_rate=.2,
              minimum_free_bytes=20 * 1024**3, checkpoint_each_epoch=True,
              train_scope='all_flow_parameters_only', reference_policy='same_patient_distinct_train',
              loss='native conditional flow matching; target frames only; frame-weighted effective batch',
              selection='lowest heldout24 reconstruction MCD; ties within 0.01 dB choose earlier epoch',
              acoustic_progress='selected reconstruction and text MCD both below epoch0, CAMPPlus neither falls more than 0.01',
              utility='if acoustic_progress, regenerate existing 576 pairs and run three fixed-step50 ASR seeds')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def emit(event, **values):
    obj = dict(event=event, time=time.time(), **values)
    print(json.dumps(obj, ensure_ascii=False), flush=True)
    with (ROOT / 'events.jsonl').open('a', encoding='utf-8') as out:
        out.write(json.dumps(obj, ensure_ascii=False) + '\n')
    write(ROOT / 'status.json', obj)


def space():
    free = shutil.disk_usage(ROOT).free
    assert free >= CONFIG['minimum_free_bytes'], free
    return free


def prepare():
    assert not ROOT.exists()
    data = read(DATA)
    train = [r for r in data if r['partition'] == 'train']
    test = [r for r in data if r['partition'] == 'merged_test']
    expected = dict(line.split(maxsplit=1) for line in (SG / 'data/train/wav.scp').read_text().splitlines())
    assert {r['uid']: r['audio'] for r in train} == expected
    assert len(train) == 2108 and len(test) == 511
    for key in ('uid', 'clean_gt', 'prompt_id', 'prompt_jyutping_sha16', 'wav_sha256'):
        assert not {r[key] for r in train} & {r[key] for r in test}, key
    for row in data:
        assert digest(row['audio']) == row['wav_sha256']
    # Stratified by fixed utterance difficulty, never by any TTS output.
    rng, selected, used = random.Random(42), [], set()
    for group in ('easy', 'medium', 'hard'):
        candidates = sorted([r for r in test if r['zero_shot_bucket'] == group], key=lambda r:r['uid'])
        rng.shuffle(candidates)
        subset = []
        for row in candidates:
            if row['speaker_id'] not in used:
                subset.append(row)
                used.add(row['speaker_id'])
            if len(subset) == 8:
                break
        assert len(subset) == 8
        selected.extend(subset)
    ROOT.mkdir()
    space()
    write(ROOT / 'config.json', CONFIG)
    write(ROOT / 'data.json', data)
    write(ROOT / 'samples.json', selected)
    write(ROOT / 'protocol.json', dict(config=CONFIG, source=digest(__file__), data=digest(DATA),
        train_rows=len(train), train_patients=len({r['speaker_id'] for r in train}),
        monitor_rows=len(test), acoustic_rows=24, acoustic_patients=24,
        target_content_unseen=True, evaluation_scope='existing merged511 feedback, not an untouched test',
        immutable_modules='epoch8 SG LLM, original tokenizer and HiFT',
        reference_conditioning='reference tokens/mel + target tokens; distinct same-patient reference; only target velocity supervised',
        no_augmentation_or_asr_changes_before_flow_results=True))
    emit('prepared', train=2108, merged_monitor=511, acoustic_monitor=24)


def runtime():
    from synthesis.cosyvoice3_patient_sft import synthesize_manifest as tts
    torch.set_num_threads(4)
    model, receipt = tts.load_fixed_runtime()
    for component in (model.model.llm, model.model.flow, model.model.hift):
        component.requires_grad_(False)
        component.eval()
    return model, receipt


def cache():
    model, receipt = runtime()
    model.model.llm.cpu()
    model.model.flow.cpu()
    model.model.hift.cpu()
    torch.cuda.empty_cache()
    write(ROOT / 'runtime.json', receipt)
    features, alignment, tokens, embeddings, sources = {}, [], {}, {}, {}
    for part in ('train', 'dev', 'test'):
        for name, target in (('utt2speech_token.pt', tokens), ('utt2embedding.pt', embeddings)):
            path = SG / 'data' / part / name
            target.update(torch.load(path, map_location='cpu', weights_only=True))
            sources[str(path)] = digest(path)
    data = read(ROOT / 'data.json')
    assert set(tokens) == set(embeddings) == {r['uid'] for r in data}
    checks = []
    # Verify the previous frozen features against the current native frontend.
    for row in read(ROOT / 'samples.json'):
        with torch.inference_mode():
            native_codes, _ = model.frontend._extract_speech_token(row['audio'])
            native_emb = model.frontend._extract_spk_embedding(row['audio'])
        uid = row['uid']
        assert native_codes[0].cpu().tolist() == tokens[uid], uid
        gap = float((native_emb[0].cpu()-torch.tensor(embeddings[uid])).abs().max())
        assert gap < 1e-5, (uid,gap)
        checks.append(dict(uid=uid, token_exact_match=True, embedding_max_error=gap))
    write(ROOT / 'feature_reuse_audit.json', dict(sources=sources, native_checks=checks))
    started = time.monotonic()
    for i, row in enumerate(data):
        uid = row['uid']
        with torch.inference_mode():
            mel, _ = model.frontend._extract_speech_feat(row['audio'])
        codes = torch.tensor([tokens[uid]], dtype=torch.int32)
        emb = torch.tensor([embeddings[uid]])
        n = min(codes.shape[1], mel.shape[1] // 2)
        assert n > 0
        uid = row['uid']
        features[uid + '.codes'] = codes[0, :n].cpu().to(torch.int16).contiguous()
        features[uid + '.mel'] = mel[0, :2*n].cpu().float().contiguous()
        features[uid + '.speaker'] = emb[0].cpu().float().contiguous()
        alignment.append(dict(uid=uid, raw_tokens=codes.shape[1], raw_mel_frames=mel.shape[1], aligned_tokens=n))
        if (i+1) % 200 == 0:
            emit('caching', completed=i+1, total=2619, elapsed=time.monotonic()-started)
    save_file(features, ROOT / 'features.safetensors')
    write(ROOT / 'alignment.json', alignment)
    emit('cache_complete', seconds=time.monotonic()-started, bytes=(ROOT / 'features.safetensors').stat().st_size)


def pair_epoch(rows, epoch):
    pools = defaultdict(list)
    for row in rows:
        pools[row['speaker_id']].append(row['uid'])
    rng = random.Random(42 + epoch)
    paired = [{**r, 'ref_uid':rng.choice([u for u in pools[r['speaker_id']] if u != r['uid']])} for r in rows]
    rng.shuffle(paired)
    return paired


def pack(rows, features, device):
    codes, mels, speakers, refs, lengths = [], [], [], [], []
    for row in rows:
        u, r = row['uid'], row['ref_uid']
        assert u != r
        a, b = features[r + '.codes'], features[u + '.codes']
        codes.append(torch.cat([a,b]).long())
        mels.append(torch.cat([features[r+'.mel'], features[u+'.mel']]))
        speakers.append(features[r+'.speaker'])
        refs.append(2*len(a)); lengths.append(2*(len(a)+len(b)))
    token = pad_sequence(codes, batch_first=True).to(device)
    x1 = pad_sequence(mels, batch_first=True).to(device).transpose(1,2)
    pos = torch.arange(x1.shape[-1], device=device)[None]
    valid = pos < torch.tensor(lengths, device=device)[:,None]
    target = valid & (pos >= torch.tensor(refs, device=device)[:,None])
    cond = x1 * (valid & ~target)[:,None]
    return token, x1, valid[:,None], target[:,None], cond, torch.stack(speakers).to(device)


def condition(flow, packed):
    token, _, valid, _, _, emb = packed
    token_mask = valid[:,0,::2,None]
    h = flow.pre_lookahead_layer(flow.input_embedding(token) * token_mask)
    h = h.repeat_interleave(2, dim=1).transpose(1,2).contiguous()
    emb = flow.spk_embed_affine_layer(F.normalize(emb,dim=1))
    return h, emb


def velocity_error(flow, packed, training):
    _, x1, valid, target, cond, _ = packed
    mu, emb = condition(flow, packed)
    batch = len(x1)
    t = torch.rand(batch,1,1,device=x1.device)
    noise = torch.randn_like(x1)
    sigma = flow.decoder.sigma_min
    y = (1-(1-sigma)*t)*noise + t*x1
    velocity = x1-(1-sigma)*noise
    if training:
        keep = torch.rand(batch,device=x1.device) > CONFIG['train_cfg_rate']
        mu, emb, cond = mu*keep[:,None,None], emb*keep[:,None], cond*keep[:,None,None]
    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=x1.is_cuda):
        pred = flow.decoder.estimator(y, valid, mu, t.flatten(), emb, cond, streaming=False)
    return ((pred.float()-velocity).square()*target).sum(), int(target.sum())*x1.shape[1]


def checkpoint_blocks(flow):
    from torch.utils.checkpoint import checkpoint
    for block in flow.decoder.estimator.transformer_blocks:
        native = block.forward
        def checked(*args, _native=native, **kwargs):
            if torch.is_grad_enabled():
                return checkpoint(_native, *args, use_reentrant=False, **kwargs)
            return _native(*args, **kwargs)
        block.forward = checked


def flow_for_training():
    model, _ = runtime()
    model.model.llm.cpu(); model.model.hift.cpu()
    flow = model.model.flow
    flow.requires_grad_(True)
    checkpoint_blocks(flow)
    # No LLM, HiFT or tokenizer parameters belong to this optimizer.
    assert all(not p.requires_grad for p in model.model.llm.parameters())
    assert all(not p.requires_grad for p in model.model.hift.parameters())
    return model, flow


def smoke():
    model, flow = flow_for_training()
    features = load_file(ROOT / 'features.safetensors')
    data = read(ROOT / 'data.json')
    paired = pair_epoch([r for r in data if r['partition']=='train'],1)
    longest = sorted(paired,key=lambda r:len(features[r['uid']+'.codes'])+len(features[r['ref_uid']+'.codes']),reverse=True)[:2]
    batch = pack(longest,features,'cuda')
    flow.eval()
    captured = {}
    native = flow.decoder.forward
    def capture(*args,**kwargs):
        captured.update(kwargs)
        return torch.zeros_like(kwargs['mu']), None
    flow.decoder.forward = capture
    row=longest[0]; u,r=row['uid'],row['ref_uid']
    with torch.no_grad():
        code=features[u+'.codes'][None].cuda().int(); ref=features[r+'.codes'][None].cuda().int(); mel=features[r+'.mel'][None].cuda()
        flow.inference(code,torch.tensor([code.shape[1]],device='cuda'),ref,torch.tensor([ref.shape[1]],device='cuda'),mel,
                       torch.tensor([mel.shape[1]],device='cuda'),features[r+'.speaker'][None].cuda(),False,True)
        p=pack([row],features,'cuda'); h,e=condition(flow,p)
        errors={k:float((a-b).abs().max()) for k,a,b in [('mu',h,captured['mu']),('speaker',e,captured['spks']),('cond',p[4],captured['cond'])]}
        assert max(errors.values())==0,errors
    flow.decoder.forward=native
    flow.train(); seed_all(42)
    opt=torch.optim.Adam(flow.parameters(),lr=CONFIG['lr'])
    before=flow.spk_embed_affine_layer.weight.detach().clone()
    started=time.monotonic()
    numerator,denominator=velocity_error(flow,batch,True)
    loss=numerator/denominator
    loss.backward()
    gradients={name:float(p.grad.norm()) for name,p in flow.named_parameters() if name in
               ['input_embedding.weight','spk_embed_affine_layer.weight','decoder.estimator.proj_out.weight']}
    assert all(math.isfinite(x) and x>0 for x in gradients.values()),gradients
    norm=torch.nn.utils.clip_grad_norm_(flow.parameters(),CONFIG['clip_norm'],error_if_nonfinite=True)
    opt.step(); torch.cuda.synchronize()
    change=float((before-flow.spk_embed_affine_layer.weight).abs().max()); assert change>0
    result=dict(native_condition_max_error=errors,parameters=sum(p.numel() for p in flow.parameters()),loss=float(loss),
                selected_gradients=gradients,clip_norm=float(norm),update_max_change=change,
                max_pair_frames=batch[1].shape[-1],micro_batch=2,seconds=time.monotonic()-started,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                update_discarded=True,no_optimizer_checkpoint=True)
    write(ROOT/'smoke.json',result); emit('smoke_passed',**result)


def heldout_loss(flow,features,rows):
    flow.eval(); numerator=denominator=0
    with torch.no_grad():
        for start in range(0,len(rows),2):
            seed_all(9000+start)
            n,d=velocity_error(flow,pack(rows[start:start+2],features,'cuda'),False)
            numerator+=float(n); denominator+=d
    return numerator/denominator


def evaluation(epoch):
    command=[sys.executable,'-u',str(REPO/'examples/flow_adaptation_evaluate.py'),'--epoch',str(epoch)]
    with (ROOT/f'evaluate_epoch_{epoch}.log').open('x') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)


def train():
    assert (ROOT/'smoke.json').exists()
    assert read(ROOT/'config.json') == CONFIG
    write(ROOT/'execution.json', dict(config=CONFIG, source_sha256={str(p):digest(p) for p in (
        Path(__file__),REPO/'examples/flow_adaptation_evaluate.py',REPO/'tests/test_flow_adaptation.py')},
        features_sha256=digest(ROOT/'features.safetensors'),
        data_sha256=digest(ROOT/'data.json'), samples_sha256=digest(ROOT/'samples.json')))
    model,flow=flow_for_training()
    features=load_file(ROOT/'features.safetensors')
    data=read(ROOT/'data.json')
    train_rows=[r for r in data if r['partition']=='train']
    test=[r for r in data if r['partition']=='merged_test']
    optimizer=torch.optim.Adam(flow.parameters(),lr=CONFIG['lr'],weight_decay=0.)
    step=0
    history=[]
    for epoch in range(CONFIG['epochs']+1):
        if epoch:
            flow.train(); paired=pair_epoch(train_rows,epoch)
            write(ROOT/f'pairs_epoch_{epoch}.json',[{'uid':r['uid'],'ref_uid':r['ref_uid']} for r in paired])
            seed_all(100+epoch)
            epoch_n=epoch_d=0
            started=time.monotonic()
            for start in range(0,len(paired),CONFIG['effective_batch']):
                group=paired[start:start+CONFIG['effective_batch']]
                denom=sum(len(features[r['uid']+'.mel'])*80 for r in group)
                optimizer.zero_grad(set_to_none=True)
                total=0.
                for j in range(0,len(group),CONFIG['micro_batch']):
                    n,d=velocity_error(flow,pack(group[j:j+CONFIG['micro_batch']],features,'cuda'),True)
                    (n/denom).backward(); total+=float(n.detach())
                norm=torch.nn.utils.clip_grad_norm_(flow.parameters(),CONFIG['clip_norm'],error_if_nonfinite=True)
                optimizer.step(); step+=1
                epoch_n+=total; epoch_d+=denom
                if step%20==0:
                    emit('training',epoch=epoch,step=step,seen_in_epoch=min(start+len(group),len(paired)),
                         loss=total/denom,grad_norm=float(norm),elapsed_epoch=time.monotonic()-started)
            space()
            path=ROOT/f'flow_epoch_{epoch}.safetensors'
            save_file({k:v.detach().cpu().contiguous() for k,v in flow.state_dict().items()},path)
            emit('epoch_saved',epoch=epoch,step=step,train_loss=epoch_n/epoch_d,sha256=digest(path))
        val=heldout_loss(flow,features,test)
        history.append(dict(epoch=epoch,step=step,heldout_conditional_cfm_loss=val))
        write(ROOT/'loss_history.json',history)
        emit('evaluating',epoch=epoch,step=step,heldout_loss=val)
        # Keep the training optimizer live; evaluations load each frozen checkpoint in a fresh process.
        evaluation(epoch)
    emit('training_complete',epochs=3,steps=step,free_gib=space()/1024**3)
    from flow_adaptation_evaluate import decision
    emit('acoustic_decision', **decision())


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','cache','smoke','train'])
    globals()[parser.parse_args().stage]()
