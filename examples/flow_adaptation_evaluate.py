"""Paired Flow-only evaluation: hold even the SG-generated token sequence fixed."""
import argparse
from html import escape
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src'), str(REPO / 'examples')]
from synthesis.cosyvoice3_patient_sft.flow_adaptation import ROOT, read, write, digest

PYTHON = '/data/qwen3-asr/venvs/qwen3-asr/bin/python'
KEYS = ('mcd_dtw_db', 'campplus_to_real', 'duration_abs_error_pct',
        'quiet_fraction_abs_gap_pp', 'f0_median_abs_gap_st')


def generate(epoch):
    import numpy as np
    import soundfile as sf
    import torch
    from safetensors.torch import load_file
    from synthesis.cosyvoice3_patient_sft import synthesize_manifest as tts
    torch.set_num_threads(4)
    model, receipt = tts.load_fixed_runtime()
    if epoch:
        weights = ROOT / f'flow_epoch_{epoch}.safetensors'
        model.model.flow.load_state_dict(load_file(weights), strict=True)
        receipt['adapted_flow_sha256'] = digest(weights)
    model.model.flow.eval()
    folder = ROOT / f'evaluation/epoch_{epoch}'
    folder.mkdir(parents=True, exist_ok=False)
    write(folder / 'runtime.json', receipt)
    data = {r['uid']: r for r in read(ROOT / 'data.json')}
    records, decoded = [], []
    captured = []
    native = model.model.flow.inference

    def observed(*args, **kwargs):
        captured.append(kwargs['token'].detach().cpu().flatten().tolist())
        return native(*args, **kwargs)

    model.model.flow.inference = observed
    for row in read(ROOT / 'samples.json'):
        ref = data[row['ref_uid']]
        assert ref['partition'] == 'train' and ref['speaker_id'] == row['speaker_id']
        assert digest(ref['audio']) == ref['wav_sha256']
        previous = ROOT / f'evaluation/epoch_0/{row["uid"]}.json'
        generated = {}
        for variant in ('reconstruction', 'text'):
            captured.clear()
            start = time.monotonic()
            tts._set_seed(42)
            with torch.inference_mode():
                if epoch == 0 and variant == 'text':
                    chunks = list(model.inference_zero_shot(row['clean_gt'],
                        tts.PROMPT_INSTRUCTION + ref['clean_gt'], ref['audio'],
                        stream=False, text_frontend=False))
                else:
                    inputs = model.frontend.frontend_zero_shot(row['clean_gt'],
                        tts.PROMPT_INSTRUCTION + ref['clean_gt'], ref['audio'], model.sample_rate, '')
                    if variant == 'reconstruction':
                        token, length = model.frontend._extract_speech_token(row['audio'])
                    else:
                        token = torch.tensor([read(previous)['generated']['text']['tokens']], dtype=torch.int32)
                        length = torch.tensor([token.shape[1]], dtype=torch.int32)
                    inputs.update(source_speech_token=token, source_speech_token_len=length)
                    chunks = list(model.model.tts(**inputs, stream=False))
            wave = torch.cat([x['tts_speech'].detach().cpu().float() for x in chunks], dim=-1).flatten().numpy()
            assert wave.size and np.isfinite(wave).all()
            assert len(captured) == 1
            if epoch:
                assert captured[0] == read(previous)['generated'][variant]['tokens']
            path = folder / f'{row["uid"]}_{variant}.wav'
            sf.write(path, wave, model.sample_rate, subtype='FLOAT')
            generated[variant] = dict(audio=str(path), sha256=digest(path), tokens=captured[0],
                seconds=len(wave)/model.sample_rate, samples=len(wave), sample_rate=model.sample_rate,
                peak=float(np.max(np.abs(wave))), clipping_fraction=float(np.mean(np.abs(wave)>=.999)),
                wall_seconds=time.monotonic()-start)
        record = dict(target_uid=row['uid'], reference_uid=ref['uid'], reference_sha256=ref['wav_sha256'],
                      seed=42, epoch=epoch, generated=generated)
        write(folder / f'{row["uid"]}.json', record)
        records.append(record)
        for variant in ('real', 'reconstruction', 'text'):
            path = row['audio'] if variant == 'real' else generated[variant]['audio']
            decoded.append({**row, 'source_utt_id':row['utt_id'], 'utt_id':row['utt_id']+'::'+variant,
                            'variant':variant, 'audio':path, 'audio_sha256':digest(path)})
        print(f'epoch {epoch}: paired audio {len(records)}/24', flush=True)
    write(folder / 'receipts.json', records)
    with (folder / 'decode.jsonl').open('x', encoding='utf-8') as handle:
        for row in decoded:
            handle.write(json.dumps(row, ensure_ascii=False)+'\n')


def score(epoch):
    import voxcpm2_sft_comparison as comparison
    folder = ROOT / f'evaluation/epoch_{epoch}'
    comparison.acoustics(folder)
    scores = {r['utt_id']:r for r in read(folder / 'acoustics.json')}
    metrics = {}
    for variant in ('reconstruction', 'text'):
        metrics[variant] = {}
        for group in ('overall', 'easy', 'medium', 'hard'):
            rows = [scores[r['utt_id']+'::'+variant] for r in read(ROOT/'samples.json')
                    if group == 'overall' or r['zero_shot_bucket']==group]
            metric = {'n':len(rows)}
            for key in KEYS:
                values = [r[key] for r in rows if r[key] is not None]
                metric[key] = statistics.mean(values) if values else None
                metric[key+'_n'] = len(values)
            metrics[variant][group] = metric
    write(folder/'metrics.json',metrics)
    report()
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


def select_epoch(history):
    selected = 1
    for epoch in sorted(history):
        if epoch > 1 and history[epoch]['reconstruction']['overall']['mcd_dtw_db'] < history[selected]['reconstruction']['overall']['mcd_dtw_db'] - .01:
            selected = epoch
    return selected


def decision():
    from synthesis.cosyvoice3_patient_sft.flow_adaptation import CONFIG
    history = {e:read(ROOT/f'evaluation/epoch_{e}/metrics.json') for e in range(CONFIG['epochs']+1)}
    selected = select_epoch(history)
    checks = {}
    for variant in ('reconstruction','text'):
        a,b = history[0][variant]['overall'],history[selected][variant]['overall']
        checks[variant+'_mcd_improves'] = b['mcd_dtw_db'] < a['mcd_dtw_db']
        checks[variant+'_camp_preserved'] = b['campplus_to_real'] >= a['campplus_to_real']-.01
    result = dict(selected_epoch=selected, checks=checks, proceed_to_utility=all(checks.values()),
                  selection=CONFIG['selection'], acoustic_rule=CONFIG['acoustic_progress'], history=history)
    write(ROOT/'acoustic_decision.json',result)
    return result


def report():
    import soundfile as sf
    folder = ROOT/'listening'
    assets = folder/'audio'
    assets.mkdir(parents=True,exist_ok=True)
    epochs = sorted(int(p.parent.name.split('_')[1]) for p in (ROOT/'evaluation').glob('epoch_*/metrics.json'))
    labels = {'reconstruction':'真实 token 重建','text':'SG 文本生成（冻结同一 token 序列）'}
    html = ['<!doctype html><html lang="zh-HK"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>SG Flow 患者适配</title><style>body{font:16px/1.6 system-ui;margin:24px;background:#f4f6fa;color:#203044}'
        'main{max-width:1250px;margin:auto}section{background:white;border-radius:12px;padding:20px;margin:20px 0}'
        '.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:15px}audio{width:100%}'
        'td,th{padding:9px;border-bottom:1px solid #ccd}table{border-collapse:collapse;width:100%}summary{cursor:pointer}'
        'small{color:#567}.gt{font-size:21px}</style><main><h1>SG · Flow-only 患者适配</h1>'
        '<p>24 位患者、24 条训练未见 content，easy/medium/hard 各 8 条（原始 utterance 难度）。'
        '固定 epoch8 LLM、tokenizer、vocoder，仅训练 Flow；2108 条 train、57 位患者。'
        '3 epochs，LR 1e-5，microbatch 2 / effective batch 8。'
        'reference、文本、实际输入 Flow 的 tokens 均固定。全部音频保留完整输出，不裁切、不重试。</p>'
        '<p>MCD 越低表示频谱更接近原声，CAMPPlus 越高表示声音相似度更高；不把回译 CER 降低当作病理保留。'
        '这些样本来自已有 merged511 反馈集，不是新设立的独立测试集。</p>']
    for group in ('overall','easy','medium','hard'):
        html.append(f'<details {"open" if group=="overall" else ""}><summary>{group} 指标</summary><table>'
                    '<tr><th>Flow epoch</th><th>模式</th><th>MCD ↓</th><th>CAMPPlus ↑</th><th>时长误差 % ↓</th></tr>')
        for e in epochs:
            metrics=read(ROOT/f'evaluation/epoch_{e}/metrics.json')
            for v in labels:
                m=metrics[v][group]
                html.append(f'<tr><td>{e} {"（原 SG）" if e==0 else ""}</td><td>{labels[v]}</td>'
                    f'<td>{m["mcd_dtw_db"]:.3f}</td><td>{m["campplus_to_real"]:.3f}</td><td>{m["duration_abs_error_pct"]:.2f}</td></tr>')
        html.append('</table></details>')
    data={r['uid']:r for r in read(ROOT/'data.json')}
    def player(path):
        p=Path(path); name=digest(p)+'.wav'; asset=assets/name
        if not asset.is_symlink():
            asset.symlink_to(p)
        seconds=sf.info(p).duration
        return f'<audio controls preload="none" src="audio/{name}"></audio><small>完整 {seconds:.2f}s · <a href="audio/{name}" download>下载</a></small>'
    for row in read(ROOT/'samples.json'):
        ref=data[row['ref_uid']]
        html.append(f'<section><h2>{escape(row["speaker_id"])} · {row["zero_shot_bucket"]}</h2>'
                    f'<p class="gt">GT：{escape(row["clean_gt"])}</p><div class="grid"><div>患者原声{player(row["audio"])}</div>'
                    f'<div>Reference：{escape(ref["clean_gt"])}{player(ref["audio"])}</div></div>')
        for e in epochs:
            rec=read(ROOT/f'evaluation/epoch_{e}/{row["uid"]}.json')
            html.append(f'<details {"open" if e in (0,epochs[-1]) else ""}><summary>Flow epoch {e}</summary><div class="grid">')
            for v in labels:
                html.append(f'<div>{labels[v]}{player(rec["generated"][v]["audio"])}</div>')
            html.append('</div></details>')
        html.append('</section>')
    html.append('</main></html>')
    (folder/'index.html').write_text(''.join(html),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--epoch',type=int,required=True)
    parser.add_argument('--stage',choices=['generate','score','all'],default='all')
    args=parser.parse_args()
    if args.stage=='all':
        env={**os.environ,'PYTHONUTF8':'1','OMP_NUM_THREADS':'4'}
        subprocess.run([PYTHON,__file__,'--epoch',str(args.epoch),'--stage','generate'],env=env,check=True)
        env['PYTHONPATH']='/data/qwen3-asr/overlays/tts-comparison-metrics'
        subprocess.run([PYTHON,__file__,'--epoch',str(args.epoch),'--stage','score'],env=env,check=True)
    elif args.stage=='generate':
        generate(args.epoch)
    else:
        score(args.epoch)
