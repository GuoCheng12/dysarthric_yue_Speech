"""Reload every saved adapter, generate the fixed panel, score and publish locally."""
import argparse
from html import escape
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
ROOT = Path('/data/qwen3-asr/records/qwen3_tts_reference_sft_setting_d_v1')
PREVIOUS = Path('/data/qwen3-asr/records/voxcpm2_reference_sft_setting_d_v1/comparison_cosyvoice_v1')
ASR_PYTHON = '/data/qwen3-asr/venvs/qwen3-asr/bin/python'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def generate(step):
    import soundfile as sf
    import torch
    from safetensors.torch import load_file
    from train_setting_d import CONFIG, digest, load_pipeline, seed_all, write
    torch.set_num_threads(4)
    pipeline = load_pipeline()
    model = pipeline.model
    if step:
        path = ROOT / 'checkpoints' / f'step_{step:06d}' / 'lora_weights.safetensors'
        adapter = load_file(path)
        params = dict(model.named_parameters())
        assert set(adapter) == {n for n, p in params.items() if p.requires_grad}
        with torch.no_grad():
            for name, value in adapter.items():
                params[name].copy_(value)
        assert any(float(v.abs().max()) > 0 for n, v in adapter.items() if 'lora_B' in n)
    model.eval()
    folder = ROOT / 'listening' / f'step_{step:06d}'
    folder.mkdir(parents=True, exist_ok=False)
    by_id = {r['uid']: r for r in read(ROOT / 'data.json')}
    observed = {}
    original = model.generate
    def observed_generate(*args, **kwargs):
        value = original(*args, **kwargs)
        observed['frames'] = len(value[0][0])
        return value
    model.generate = observed_generate
    receipts, decode = [], []
    for row in read(ROOT / 'samples.json'):
        ref = by_id[row['ref_uid']]
        started = time.monotonic()
        seed_all(42)
        with torch.inference_mode():
            wavs, sr = pipeline.generate_voice_clone(text=row['clean_gt'], language='Auto',
                ref_audio=ref['audio'], ref_text=ref['clean_gt'], x_vector_only_mode=False,
                non_streaming_mode=CONFIG['non_streaming_mode'], max_new_tokens=CONFIG['max_new_tokens'])
        assert len(wavs) == 1 and len(wavs[0]) > 0
        wave = wavs[0]
        path = folder / (row['uid'] + '.wav')
        sf.write(path, wave, sr, subtype='FLOAT')
        receipt = dict(step=step, target_uid=row['uid'], speaker_id=row['speaker_id'],
            patient_group=row['patient_group'], text=row['clean_gt'], real_audio=row['audio'],
            reference_audio=ref['audio'], reference_text=ref['clean_gt'], reference_sha256=digest(ref['audio']),
            seed=42, audio=str(path), sha256=digest(path), samples=len(wave), sample_rate=sr,
            seconds=len(wave) / sr, generated_frames=observed['frames'],
            hit_length_limit=observed['frames'] >= CONFIG['max_new_tokens'] - 1,
            wall_seconds=time.monotonic() - started, conditioning='full_ICL')
        receipts.append(receipt)
        write(folder / 'receipts.json', receipts)
        for variant, audio in [('real', row['audio']), ('qwen', str(path))]:
            decode.append({**row, 'source_utt_id': row['utt_id'], 'utt_id': row['utt_id'] + '::' + variant,
                'variant': variant, 'audio': audio, 'audio_sha256': digest(audio)})
        print(json.dumps(receipt, ensure_ascii=False), flush=True)
    with (folder / 'decode.jsonl').open('x', encoding='utf-8') as handle:
        for row in decode:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    env = {**os.environ, 'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8',
           'PYTHONPATH': '/data/qwen3-asr/overlays/tts-comparison-metrics', 'OMP_NUM_THREADS': '4'}
    subprocess.run([ASR_PYTHON, __file__, 'score', '--step', str(step)], env=env, check=True)


def score(step):
    sys.path.insert(0, str(REPO / 'examples'))
    import voxcpm2_sft_comparison as comparison
    folder = ROOT / 'listening' / f'step_{step:06d}'
    comparison.acoustics(folder)
    acoustic = {r['utt_id']: r for r in read(folder / 'acoustics.json')}
    samples = read(ROOT / 'samples.json')
    metrics = {}
    for group in ('overall', 'easy', 'hard'):
        subset = [acoustic[r['utt_id'] + '::qwen'] for r in samples if group == 'overall' or r['patient_group'] == group]
        metrics[group] = {'n': len(subset)}
        for key in ('mcd_dtw_db', 'campplus_to_real', 'duration_abs_error_pct'):
            metrics[group][key] = statistics.mean(r[key] for r in subset)
    metrics['length_cap_hits'] = sum(r['hit_length_limit'] for r in read(folder / 'receipts.json'))
    comparison.write(folder / 'metrics.json', metrics)
    report()
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


def report():
    import soundfile as sf
    from hashlib import sha256
    folder = ROOT / 'listening'
    assets = folder / 'audio'
    assets.mkdir(exist_ok=True)
    previous = read(PREVIOUS / 'results.json')
    completed = sorted(folder.glob('step_*/metrics.json'))
    comparisons = [(0, read(path), path.parent) for path in completed]
    comparisons = [(int(path.name.split('_')[1]), metric, path) for _, metric, path in comparisons]
    html = ['<!doctype html><html lang="zh-HK"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Qwen3-TTS 患者微调监控</title><style>body{font:16px/1.65 system-ui;background:#f3f6fa;color:#1d3046;margin:0}'
        'main{max-width:1250px;margin:auto;padding:28px}section,.box{background:white;border:1px solid #dde3ec;border-radius:12px;padding:20px;margin:18px 0}'
        '.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}.voice{padding:14px;background:#f6f8fb;border-radius:8px}'
        'audio{width:100%}td,th{padding:10px;border-bottom:1px solid #dde3ec;text-align:left}table{border-collapse:collapse;width:100%;min-width:680px}'
        '.small{font-size:13px;color:#576675}.gt{font-size:22px}a{color:#176cad}summary{cursor:pointer}</style><main>'
        '<h1>Qwen3-TTS · Setting D 患者微调监控</h1><div class="box">相同 6 位患者（3 easy / 3 hard）、GT、reference、seed 42。'
        '每个保存点重新加载 adapter，生成完整音频，再计算 MCD-DTW 和 CAMPPlus。旧 Speech Generator 作为固定参照。'
        '<p>训练：2108 条 / 57 位患者；LoRA r16/alpha32，LR 2e-5，batch 4 × grad_acc 4，3 epochs / 396 steps，每 66 steps 保存。'
        '完整 ICL 条件：reference codes＋transcript＋speaker embedding；language=Auto。</p>'
        '<p>MCD 越低、CAMPPlus 越高，分别表示频谱与音色更接近同 GT 的患者原声；并非病理构音认证。'
        '所有 WAV 保留模型完整输出，无重试替换或裁切。页面仅在一个 checkpoint 的 6 条评分全部完成后更新；刷新查看新结果。</p></div>']
    for group in ('overall', 'easy', 'hard'):
        if group != 'overall':
            html.append(f'<details><summary>{group} 分组指标</summary>')
        html.append('<div style="overflow:auto"><table><tr><th>版本</th><th>MCD-DTW ↓</th><th>CAMPPlus ↑</th>'
            '<th>时长误差 ↓</th><th>生成触顶数</th></tr>')
        sg = previous['metrics']['cosy_sg'][group]
        html.append(f'<tr><td>旧 CosyVoice SG epoch 8</td><td>{sg["mcd_dtw_db"]:.3f}</td>'
            f'<td>{sg["campplus_to_real"]:.3f}</td><td>{sg["duration_abs_error_pct"]:.1f}%</td><td>—</td></tr>')
        for step, metrics, _ in comparisons:
            m = metrics[group]
            html.append(f'<tr><td>Qwen {"zero-shot" if step == 0 else "step " + str(step)}</td><td>{m["mcd_dtw_db"]:.3f}</td>'
                f'<td>{m["campplus_to_real"]:.3f}</td><td>{m["duration_abs_error_pct"]:.1f}%</td><td>{metrics["length_cap_hits"]}/6（全组）</td></tr>')
        html.append('</table></div>')
        if group != 'overall':
            html.append('</details>')
    def player(path):
        info = sf.info(path)
        name = sha256(Path(path).read_bytes()).hexdigest() + '.wav'
        asset = assets / name
        if not asset.is_symlink():
            asset.symlink_to(Path(path))
        return (f'<audio controls preload="none" src="audio/{name}" data-seconds="{info.duration:.6f}"></audio>'
                f'<p class="small">完整 {info.duration:.2f}s · <a href="audio/{name}" download>下载 WAV</a> · <span class="check">待载入核验</span></p>')
    latest_step = comparisons[-1][0]
    for row in read(ROOT / 'samples.json'):
        pair = previous['pairs'][row['utt_id']]
        html.append(f'<section><h2>{escape(row["speaker_id"])} · {row["patient_group"]}</h2>'
            f'<p class="gt">{escape(row["clean_gt"])}</p><h3>Reference</h3>'
            f'<p>{escape(pair["real"]["reference_text"])}</p>' + player(pair['real']['reference_audio']) + '<div class="grid">')
        for v, label in [('real', '患者原声'), ('cosy_sg', '旧 Speech Generator')]:
            html.append(f'<div class="voice"><strong>{label}</strong>' + player(pair[v]['audio']) + '</div>')
        for step, _, path in comparisons:
            audio = path / (row['uid'] + '.wav')
            metrics = next(r for r in read(path / 'acoustics.json') if r['utt_id'] == row['utt_id'] + '::qwen')
            label = 'Qwen 原始 zero-shot' if step == 0 else f'Qwen step {step}'
            content = f'<div class="voice"><strong>{label}</strong>' + player(audio) + f'<p class="small">MCD {metrics["mcd_dtw_db"]:.3f} · CAMPPlus {metrics["campplus_to_real"]:.3f}</p></div>'
            html.append(content if step in (0, latest_step) else f'<details><summary>{label}</summary>{content}</details>')
        html.append('</div></section>')
    html.append('<script>document.querySelectorAll("audio").forEach(a=>a.addEventListener("loadedmetadata",()=>{'
        'a.nextElementSibling.querySelector(".check").textContent=Math.abs(a.duration-Number(a.dataset.seconds))<0.1?"完整时长已核对":"时长不符，请下载 WAV";}));'
        'document.addEventListener("play",e=>{if(e.target.tagName==="AUDIO")document.querySelectorAll("audio").forEach(a=>{if(a!==e.target)a.pause();});},true);'
        '</script></main></html>')
    temp = folder / 'index.tmp.html'
    temp.write_text('\n'.join(html), encoding='utf-8')
    temp.replace(folder / 'index.html')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['generate', 'score'])
    parser.add_argument('--step', type=int, required=True)
    args = parser.parse_args()
    globals()[args.action](args.step)
