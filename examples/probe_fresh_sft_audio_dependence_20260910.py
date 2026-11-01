"""Fixed same-speaker audio mismatch probe for existing fresh SFT checkpoints."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RECORDS = Path('/data/qwen3-asr/records/therapist_mas_setting_d_v1')
FRESH = RECORDS / 'qwen3_asr_fresh_lora_sft_seed42_4configs_20260910_v1'
ROOT = RECORDS / 'fresh_sft_audio_dependence_probe_20260910_v1'
BASE = Path('/data/qwen3-asr/models/Qwen3-ASR-1.7B')
CONFIGS = dict(A_legacy_full=40, B_alpha32_full=27, C_alpha32_language=14, D_alpha32_token_ce=27)
for sub in ['src', 'finetune/scripts', 'finetune/official']:
    sys.path.insert(0, str(REPO / sub))
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('OMP_NUM_THREADS', '3')
os.environ.setdefault('MKL_NUM_THREADS', '3')


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        while chunk := f.read(2**20):
            h.update(chunk)
    return h.hexdigest()


def save(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_suffix(p.suffix + '.pending')
    q.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    q.replace(p)


def event(out, stage, **values):
    out.mkdir(parents=True, exist_ok=True)
    r = dict(utc=datetime.now(timezone.utc).isoformat(), stage=stage, **values)
    with (out / 'events.jsonl').open('a') as f:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(json.dumps(r, ensure_ascii=False), flush=True)


def prepare():
    from scipy.optimize import linear_sum_assignment
    from therapist_harness.text import text_normalize
    ROOT.mkdir(exist_ok=False)
    (ROOT / 'tmp').mkdir()
    full = dict(train=read(FRESH / 'train.json'), evaluation=read(FRESH / 'dev.json'))
    source_groups = {}
    for split, rs in full.items():
        groups = defaultdict(lambda: defaultdict(list))
        for r in rs:
            groups[r['speaker_id']][text_normalize(r['clean_gt'])].append(r)
        source_groups[split] = groups
    eligible = sorted(set.intersection(*[{s for s, ts in groups.items() if len(ts) >= 2}
                                          for groups in source_groups.values()]))
    panels, selection = {}, {}
    for split, groups in source_groups.items():
        panel = []
        for speaker in eligible:
            rng = random.Random('42:' + split + ':' + speaker)
            ts = sorted(groups[speaker])
            rng.shuffle(ts)
            panel.extend(rng.choice(groups[speaker][t]) for t in ts[:8])
        assert all(len({text_normalize(r['clean_gt']) for r in panel if r['speaker_id'] == s}) >= 2 for s in eligible)
        panels[split] = panel
        selection[split] = dict(audio=len(panel), speakers=len(eligible), texts=len({r['clean_gt'] for r in panel}),
                                difficulty=dict(Counter(r['zero_shot_bucket'] for r in panel)))
    items = []
    for split, panel in panels.items():
        for speaker in eligible:
            rs = [r for r in panel if r['speaker_id'] == speaker]
            n = len(rs)
            duration = np.log([float(r['duration']) for r in rs])
            cost = np.abs(duration[:, None] - duration[None, :])
            cost += np.array([[.5 * (a['zero_shot_bucket'] != b['zero_shot_bucket']) for b in rs] for a in rs])
            previous = None
            mappings = []
            for permutation in range(2):
                rng = np.random.default_rng(int(hashlib.sha256(f'42:{split}:{speaker}:{permutation}'.encode()).hexdigest()[:8], 16))
                current = cost + rng.uniform(0, .05, (n, n))
                np.fill_diagonal(current, 1e6)
                if previous is not None and n > 2:
                    current[np.arange(n), previous] = 1e6
                ix, iy = linear_sum_assignment(current)
                assert list(ix) == list(range(n)) and len(set(iy)) == n and np.all(ix != iy)
                if previous is not None and n > 2:
                    assert np.all(iy != previous)
                mappings.append(iy)
                previous = iy
            for i, r in enumerate(rs):
                donors = [rs[int(m[i])] for m in mappings]
                assert all(d['speaker_id'] == r['speaker_id'] and text_normalize(d['clean_gt']) != text_normalize(r['clean_gt']) for d in donors)
                items.append(dict(split=split, target=r, wrong_audio=donors))
    input_hashes = {str(FRESH / f'{name}.json'): sha(FRESH / f'{name}.json') for name in ['train', 'dev', 'audio_sha256']}
    sources = [Path(__file__), REPO / 'finetune/official/qwen3_asr_sft.py', REPO / 'src/therapist_harness/text.py']
    checkpoint_hashes = {}
    for cfg, early in CONFIGS.items():
        for step in [early, 67]:
            cp = FRESH / cfg / f'checkpoint-{step}'
            for name in ['adapter_config.json', 'adapter_model.safetensors']:
                checkpoint_hashes[str(cp / name)] = sha(cp / name)
    saved_hashes = read(FRESH / 'audio_sha256.json')
    selected_audio = sorted({i['target']['audio'] for i in items})
    assert all(sha(p) == saved_hashes[p] for p in selected_audio)
    save(ROOT / 'panel.json', items)
    save(ROOT / 'protocol.json', dict(seed=42, selection=selection, eligible_speakers=eligible,
        excluded_evaluation_speakers=sorted(set(source_groups['evaluation']) - set(eligible)),
        correct_and_wrong_use_identical_target_text=True,
        pairing='Two fixed bijective within-speaker derangements of distinct texts; minimize log-duration gap plus 0.5 difficulty mismatch. Second permutation forbids first donors when n>2.',
        configs=CONFIGS, late_step=67, modes=['unmerged_bf16_autocast', 'merged_bf16_autocast'],
        primary='Per-utterance mean transcription token NLL, excluding language header and EOS. Report correct, mean of two wrong, and gap wrong-correct.',
        secondary=['all supervised token NLL including language/EOS', 'first transcription token NLL', 'remaining transcription token NLL'],
        no_optimizer=True, evaluation_mode=True, fixed_ground_truth_prefix=True,
        inference_scope='Teacher-forced forward only; this does not alone prove reliance during autoregressive decoding.',
        patient_sampling='Same 34 eligible evaluation patients in both panels, up to 8 distinct target texts per patient; descriptive diagnostic panels, not representative pooled CER.',
        input_sha256=input_hashes, checkpoint_sha256=checkpoint_hashes,
        source_sha256={str(p): sha(p) for p in sources},
        base_files_sha256=read(FRESH / 'protocol.json')['base_files_sha256'],
        selected_audio_sha256={p: saved_hashes[p] for p in selected_audio},
        panel_sha256=sha(ROOT / 'panel.json')))
    event(ROOT, 'prepared', selection=selection, rows=len(items))


def load_model(checkpoint):
    import torch
    from peft import PeftModel
    from qwen_asr import Qwen3ASRModel
    from qwen3_asr_sft import patch_outer_forward
    start = time.monotonic()
    w = Qwen3ASRModel.from_pretrained(str(BASE), dtype=torch.bfloat16, device_map='cuda:0')
    patch_outer_forward(w.model)
    if checkpoint is not None:
        w.model = PeftModel.from_pretrained(w.model, str(checkpoint), is_trainable=False)
        assert all(p.dtype == torch.float32 for n, p in w.model.named_parameters() if '.lora_' in n)
    w.model.requires_grad_(False)
    w.model.eval()
    w.model.config.use_cache = False
    return w, time.monotonic() - start


def score(w, batch):
    import torch
    import torch.nn.functional as F
    from qwen3_asr_sft import DataCollatorForQwen3ASRFinetuning, make_preprocess_fn_prefix_only
    collator = DataCollatorForQwen3ASRFinetuning(w.processor)
    preprocess = make_preprocess_fn_prefix_only(w.processor)
    inputs = collator([preprocess(r) for r in batch])
    labels = inputs['labels'].to('cuda')[:, 1:]
    input_ids = inputs['input_ids']
    marker = w.processor.tokenizer.encode('<asr_text>', add_special_tokens=False)
    assert len(marker) == 1
    marker = marker[0]
    eos = w.processor.tokenizer.eos_token_id
    masks = []
    for i in range(len(batch)):
        ids = input_ids[i]
        positions = ((ids == marker) & inputs['labels'][i].ne(-100)).nonzero().flatten()
        assert len(positions) == 1
        start = int(positions[0])
        mask = labels[i].ne(-100) & (torch.arange(labels.shape[1], device='cuda') >= start) & labels[i].ne(eos)
        token_ids = labels[i][mask].tolist()
        target = batch[i]['text'].split('<asr_text>', 1)[1]
        decoded = w.processor.tokenizer.decode(token_ids)
        assert decoded == target, (decoded, target)
        assert len(token_ids) >= 2
        masks.append(mask)
    args = {k: v.to('cuda', dtype=torch.bfloat16 if v.is_floating_point() else v.dtype)
            for k, v in inputs.items() if k != 'labels'}
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        logits = w.model(**args).logits[:, :-1].float()
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100, reduction='none').view_as(labels)
    result = []
    for i, mask in enumerate(masks):
        v = loss[i][mask]
        values = dict(text_nll=float(v.mean()), first_nll=float(v[0]), continuation_nll=float(v[1:].mean()),
                      supervised_nll=float(loss[i][labels[i].ne(-100)].mean()), text_tokens=len(v))
        assert all(np.isfinite(x) for x in values.values())
        result.append(values)
    return result


def run_variant(w, items, out, name, mode):
    start = time.monotonic()
    results = []
    for i in range(0, len(items), 4):
        batch = items[i:i+4]
        condition_scores = {}
        for condition in ['correct', 'wrong0', 'wrong1']:
            records = [dict(x['target'], audio=x['target']['audio'] if condition == 'correct' else x['wrong_audio'][int(condition[-1])]['audio']) for x in batch]
            condition_scores[condition] = score(w, records)
        for j, item in enumerate(batch):
            results.append(dict(split=item['split'], utt_id=item['target']['utt_id'], speaker_id=item['target']['speaker_id'],
                                difficulty=item['target']['zero_shot_bucket'], text=item['target']['clean_gt'],
                                **{c: vs[j] for c, vs in condition_scores.items()}))
        if i % 64 == 0:
            event(out, 'forward', checkpoint=name, mode=mode, completed=len(results), total=len(items))
    save(out / f'{name}_{mode}.json', results)
    event(out, 'variant_complete', checkpoint=name, mode=mode, audio=len(results), seconds=time.monotonic()-start)


def worker(cfg):
    import gc
    import torch
    out = ROOT / cfg
    protocol = read(ROOT / 'protocol.json')
    assert all(sha(p) == h for p, h in protocol['source_sha256'].items())
    assert sha(ROOT / 'panel.json') == protocol['panel_sha256']
    items = read(ROOT / 'panel.json')
    variants = [('raw', None)] if cfg == 'A_legacy_full' else []
    variants += [(f'{cfg[0]}_step{s}', FRESH / cfg / f'checkpoint-{s}') for s in [CONFIGS[cfg], 67]]
    for name, checkpoint in variants:
        if checkpoint:
            assert all(sha(checkpoint / n) == protocol['checkpoint_sha256'][str(checkpoint / n)] for n in ['adapter_config.json', 'adapter_model.safetensors'])
        w, elapsed = load_model(checkpoint)
        event(out, 'loaded', checkpoint=name, seconds=elapsed)
        # Exact repeat, batched-vs-single, and unchanged weights on a real batch.
        smoke = [items[0]['target'], items[1]['target']]
        a = score(w, smoke)
        b = score(w, smoke)
        c = score(w, smoke[:1])
        assert a == b
        assert abs(a[0]['text_nll'] - c[0]['text_nll']) < .12, (a, c)
        weights = [(p, p.detach().clone()) for n, p in w.model.named_parameters() if '.lora_' in n]
        event(out, 'smoke_passed', checkpoint=name, repeat_exact=True,
              batch_vs_single_nll_delta=a[0]['text_nll']-c[0]['text_nll'], masks_decode_to_target=True, no_grad=True)
        run_variant(w, items, out, name, 'unmerged')
        assert all(torch.equal(p, old) for p, old in weights)
        del weights
        if checkpoint:
            w.model.merge_adapter()
            run_variant(w, items, out, name, 'merged')
        del w
        gc.collect()
        torch.cuda.empty_cache()
    assert all(sha(p) == h for p, h in protocol['source_sha256'].items())
    assert all(sha(p) == h for p, h in protocol['checkpoint_sha256'].items())
    save(out / 'complete.json', dict(passed=True, checkpoint_files_unchanged=True))
    event(out, 'complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'worker'])
    parser.add_argument('--config', choices=list(CONFIGS))
    args = parser.parse_args()
    prepare() if args.mode == 'prepare' else worker(args.config)
