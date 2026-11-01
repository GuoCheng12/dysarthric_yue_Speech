"""Six fixed, held-out patient utterances: VoxCPM2 checkpoints vs Speech Generator.

Run cosy in a fresh ASR-env process (its loader selects Transformers 4.51.3).
Run acoustics with the isolated tts-comparison-metrics overlay on PYTHONPATH.
"""
import argparse
import hashlib
from html import escape
import json
import math
from pathlib import Path
import statistics
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src')]
VOX = Path('/data/qwen3-asr/records/voxcpm2_reference_sft_setting_d_v1')
ROOT = VOX / 'comparison_cosyvoice_v1'
SG = Path('/data/qwen3-asr/records/cosyvoice3_reference_sft_setting_d_v1')
CAMP = Path('/data/qwen3-asr/models/tts/Fun-CosyVoice3-0.5B-2512/campplus.onnx')
LABEL = {'real': '患者原声', 'cosy_sg': 'CosyVoice3 · Speech Generator / epoch 8',
         'vox_0': 'VoxCPM2 · 原始 zero-shot', 'vox_132': 'VoxCPM2 · step 132',
         'vox_330': 'VoxCPM2 · step 330', 'vox_396': 'VoxCPM2 · step 396'}
VARIANTS = tuple(LABEL)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]


def write(path, value):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare():
    selected = read(VOX / 'samples.json')
    data = read(VOX / 'data.json')
    by_id = {r['uid']: r for r in data}
    train = [r for r in data if r['partition'] == 'train']
    sg_audio = {line.split(maxsplit=1)[1] for line in
                (SG / 'data/train/wav.scp').read_text(encoding='utf-8').splitlines()}
    assert sg_audio == {r['audio'] for r in train}
    assert len(selected) == 6 and len({r['speaker_id'] for r in selected}) == 6
    assert [r['patient_group'] for r in selected].count('hard') == 3
    for row in selected:
        ref = by_id[row['ref_uid']]
        assert ref['partition'] == 'train' and ref['speaker_id'] == row['speaker_id']
        assert row['uid'] != ref['uid'] and row['partition'] == 'merged_test'
        for key in ('audio', 'clean_gt', 'prompt_jyutping_sha16', 'wav_sha256'):
            assert row[key] not in {r[key] for r in train}, (row['uid'], key)
        row['reference_audio'] = ref['audio']
        row['reference_text'] = ref['clean_gt']
        row['reference_sha256'] = digest(ref['audio'])
        assert digest(row['audio']) == row['wav_sha256']
    ROOT.mkdir(exist_ok=False)
    write(ROOT / 'selection.json', selected)
    write(ROOT / 'protocol.json', {
        'patients': 6, 'easy_patients': 3, 'hard_patients': 3, 'seed': 42,
        'tts_train_rows_both': len(train), 'same_train_set': True,
        'target_content_unseen_in_tts_train': True, 'reference_from_train_only': True,
        'variants': LABEL, 'selection': 'existing fixed checkpoint listening panel; no result-based filtering',
        'vox_mode': 'reference_only, no VAD trimming; 10 inference steps, cfg=2, max_len=512; no retry',
        'cosy_mode': 'native zero-shot API with reference waveform + reference transcript; generic instruction, implicit Cantonese; no retry',
        'conditioning_note': 'Same reference waveform, target text and seed; native conditioning mechanisms differ.',
        'asr': 'Original Qwen3-ASR-1.7B, no LoRA, Cantonese, batch 8, max_new_tokens 256',
        'audio': 'Full original output; no trimming, rejection, amplitude normalization or preview clips',
        'metrics': {
            'mcd_dtw_db': '16k mono WORLD CheapTrick powerspectrum; pysptk.sp2mc order24 alpha0.42; c1..c24, c0 excluded; 10ms; global Euclidean DTW, mean path cost times 10*sqrt(2)/ln(10). Quiet frames excluded by 25ms RMS < 35dB below each file RMS 95th percentile (floor 1e-7). Lower is closer, not a clinical impairment metric.',
            'campplus_to_real': 'Cosine of existing CAMPPlus embeddings to same-GT real speech; higher is closer in timbre. CAMPPlus also conditions CosyVoice: auxiliary, not an independent pathology judge.',
            'duration_abs_error_pct': '100*abs(synthetic full duration / real full duration - 1); lower is closer.',
            'quiet_fraction_abs_gap_pp': '100*abs(relative-energy quiet frame fraction difference); includes leading/trailing silence, not linguistic pause detection.',
            'f0_median_abs_gap_st': 'Absolute 12*log2(synthetic voiced F0 median / real voiced F0 median); WORLD DIO+StoneMask 60..500Hz; missing if no voiced frames.',
            'cer': 'Same frozen zero-shot ASR for generated AND real; corpus character-weighted CER. Lower measures recognized content, not impairment retention.',
            'asr_to_real_gap': 'Edit distance between synthetic and real ASR hypotheses / GT characters; diagnostic agreement, not a pathology score.',
            'aggregation': 'Acoustics: mean of six utterances (one per patient); CER: pooled character counts. Group labels are pre-existing patient groups, not utterance CER buckets.'},
        'sources': ['https://github.com/MattShannon/mcd',
                    'https://pysptk.readthedocs.io/en/latest/generated/pysptk.conversion.sp2mc.html',
                    'https://arxiv.org/html/2508.05102v1']})
    print('Prepared six paired patients', flush=True)


def cosy():
    import numpy as np
    import soundfile as sf
    import torch
    from synthesis.cosyvoice3_patient_sft import synthesize_manifest as tts
    torch.set_num_threads(4)
    model, meta = tts.load_fixed_runtime()
    folder = ROOT / 'cosy_audio'
    folder.mkdir(exist_ok=False)
    write(ROOT / 'cosy_runtime.json', meta)
    receipts = []
    for row in read(ROOT / 'selection.json'):
        start = time.monotonic()
        tts._set_seed(42)
        chunks = [x['tts_speech'].detach().cpu().float() for x in model.inference_zero_shot(
            row['clean_gt'], tts.PROMPT_INSTRUCTION + row['reference_text'], row['reference_audio'],
            stream=False, text_frontend=False)]
        wave = torch.cat(chunks, dim=-1).flatten().numpy()
        assert wave.size and np.isfinite(wave).all()
        path = folder / (row['uid'] + '.wav')
        sf.write(path, wave, model.sample_rate, subtype='FLOAT')
        receipt = {'target_uid': row['uid'], 'speaker_id': row['speaker_id'],
                   'text': row['clean_gt'], 'reference_audio': row['reference_audio'],
                   'reference_text': row['reference_text'], 'reference_sha256': row['reference_sha256'],
                   'seed': 42, 'audio': str(path), 'sha256': digest(path),
                   'seconds': len(wave) / model.sample_rate, 'samples': len(wave),
                   'sample_rate': model.sample_rate, 'wall_seconds': time.monotonic() - start}
        write(folder / (row['uid'] + '.json'), receipt)
        receipts.append(receipt)
        print(json.dumps(receipt, ensure_ascii=False), flush=True)
    write(ROOT / 'cosy_receipts.json', receipts)


def assemble():
    receipts = {'cosy_sg': read(ROOT / 'cosy_receipts.json')}
    for step in (0, 132, 330, 396):
        receipts[f'vox_{step}'] = read(VOX / f'listening/step_{step:06d}/receipts.json')
    lookup = {v: {x['target_uid']: x for x in items} for v, items in receipts.items()}
    output = []
    for row in read(ROOT / 'selection.json'):
        for variant in VARIANTS:
            path = row['audio']
            if variant != 'real':
                rec = lookup[variant][row['uid']]
                assert rec['seed'] == 42 and rec['text'] == row['clean_gt']
                for key in ('reference_audio', 'reference_text', 'reference_sha256'):
                    assert rec[key] == row[key]
                path = rec['audio']
                assert digest(path) == rec['sha256']
            output.append({**row, 'source_utt_id': row['utt_id'],
                           'utt_id': row['utt_id'] + '::' + variant, 'variant': variant,
                           'audio': path, 'audio_sha256': digest(path)})
    assert len(output) == 36
    with (ROOT / 'decode.jsonl').open('x', encoding='utf-8') as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    print('Assembled 36 full-waveform ASR inputs', flush=True)


def mcd_dtw(a, b):
    import librosa
    import numpy as np
    from scipy.spatial.distance import cdist
    cost = cdist(a, b, metric='euclidean')
    _, path = librosa.sequence.dtw(C=cost, backtrack=True)
    return float(np.mean(cost[path[:, 0], path[:, 1]]) * (10 * math.sqrt(2) / math.log(10)))


def duration_error(synthetic, real):
    return 100 * abs(synthetic / real - 1)


def acoustics(root=ROOT):
    import librosa
    import numpy as np
    import onnxruntime as ort
    import pysptk
    import pyworld
    import soundfile as sf
    import torch
    import torchaudio.compliance.kaldi as kaldi
    torch.set_num_threads(4)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    camp = ort.InferenceSession(str(CAMP), sess_options=options, providers=['CPUExecutionProvider'])
    features = {}
    decoded = rows(root / 'decode.jsonl')
    for i, row in enumerate(decoded):
        assert digest(row['audio']) == row['audio_sha256']
        raw, original_sr = sf.read(row['audio'], always_2d=True)
        assert np.isfinite(raw).all()
        y = librosa.resample(raw.mean(axis=1), orig_sr=original_sr, target_sr=16000).astype(np.float64)
        rms = librosa.feature.rms(y=y, frame_length=400, hop_length=160, center=True)[0]
        active = rms >= max(float(np.quantile(rms, .95)) * 10 ** (-35 / 20), 1e-7)
        f0, times = pyworld.dio(y, 16000, f0_floor=60, f0_ceil=500, frame_period=10.)
        f0 = pyworld.stonemask(y, f0, times, 16000)
        spectrum = pyworld.cheaptrick(y, f0, times, 16000)
        mc = pysptk.sp2mc(spectrum, order=24, alpha=.42)[:, 1:]
        assert len(active) == len(mc) and active.any() and np.isfinite(mc).all()
        feat = kaldi.fbank(torch.from_numpy(y.astype(np.float32))[None],
                           num_mel_bins=80, dither=0, sample_frequency=16000)
        feat -= feat.mean(dim=0, keepdim=True)
        emb = camp.run(None, {camp.get_inputs()[0].name: feat[None].numpy()})[0].flatten()
        emb /= np.linalg.norm(emb)
        voiced = f0[f0 > 0]
        features[row['utt_id']] = {'mc': mc[active], 'embedding': emb,
            'seconds': len(raw) / original_sr, 'frames': len(raw), 'sample_rate': original_sr,
            'quiet_fraction': float(np.mean(~active)),
            'f0_median': float(np.median(voiced)) if len(voiced) else None}
        print(f'Acoustic features {i+1}/{len(decoded)}', flush=True)
    output = []
    for row in decoded:
        a = features[row['source_utt_id'] + '::real']
        b = features[row['utt_id']]
        pitch = abs(12 * math.log2(b['f0_median'] / a['f0_median'])) if a['f0_median'] and b['f0_median'] else None
        output.append({'utt_id': row['utt_id'], 'variant': row['variant'],
                       'seconds': b['seconds'], 'frames': b['frames'], 'sample_rate': b['sample_rate'],
                       'mcd_dtw_db': mcd_dtw(a['mc'], b['mc']),
                       'campplus_to_real': float(np.dot(a['embedding'], b['embedding'])),
                       'duration_ratio': b['seconds'] / a['seconds'],
                       'duration_abs_error_pct': duration_error(b['seconds'], a['seconds']),
                       'quiet_fraction_abs_gap_pp': 100 * abs(b['quiet_fraction'] - a['quiet_fraction']),
                       'f0_median_abs_gap_st': pitch})
    write(root / 'acoustics.json', output)
    print('Acoustics complete', flush=True)


def summarize():
    from therapist_harness.text import edit_distance, text_normalize
    predictions = rows(ROOT / 'asr/predictions.jsonl')
    expected = {r['utt_id']: r['audio_sha256'] for r in rows(ROOT / 'decode.jsonl')}
    assert {r['utt_id']: r['audio_sha256'] for r in predictions} == expected
    acoustic = {r['utt_id']: r for r in read(ROOT / 'acoustics.json')}
    pairs = {}
    for row in predictions:
        assert digest(row['audio']) == expected[row['utt_id']]
        assert edit_distance(text_normalize(row['clean_gt']), text_normalize(row['prediction'])) == row['edit_count']
        pairs.setdefault(row['source_utt_id'], {})[row['variant']] = {**row, **acoustic[row['utt_id']]}
    assert len(pairs) == 6 and all(set(p) == set(VARIANTS) for p in pairs.values())
    for pair in pairs.values():
        for row in pair.values():
            row['asr_to_real_gap'] = edit_distance(text_normalize(row['prediction']),
                text_normalize(pair['real']['prediction'])) / row['normalized_reference_characters']
    metrics = {}
    for variant in VARIANTS:
        metrics[variant] = {}
        for group in ('overall', 'easy', 'hard'):
            subset = [p[variant] for p in pairs.values() if group == 'overall' or p['real']['patient_group'] == group]
            metric = {'n': len(subset), 'cer': sum(r['edit_count'] for r in subset) /
                      sum(r['normalized_reference_characters'] for r in subset)}
            for key in ('mcd_dtw_db', 'campplus_to_real', 'duration_abs_error_pct',
                        'quiet_fraction_abs_gap_pp', 'f0_median_abs_gap_st', 'asr_to_real_gap'):
                values = [r[key] for r in subset if r[key] is not None]
                metric[key] = statistics.mean(values) if values else None
                metric[key + '_n'] = len(values)
            metrics[variant][group] = metric
    write(ROOT / 'results.json', {'metrics': metrics, 'pairs': pairs,
          'campplus_sha256': digest(CAMP), 'protocol': read(ROOT / 'protocol.json')})
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


def report():
    result = read(ROOT / 'results.json')
    folder = ROOT / 'listening'
    (folder / 'audio').mkdir(parents=True, exist_ok=True)

    def player(path, seconds):
        asset = folder / 'audio' / (digest(path) + '.wav')
        if not asset.is_symlink():
            asset.symlink_to(Path(path))
        url = 'audio/' + asset.name
        return (f'<audio controls preload="none" src="{url}" data-seconds="{seconds:.6f}"></audio>'
                f'<div class="small">完整音频 {seconds:.2f}s · <a href="{url}" download>下载 WAV</a>'
                ' · <span class="check">载入后核验播放器时长</span></div>')

    def num(value, digits=2):
        return '—' if value is None else f'{value:.{digits}f}'

    html = ['<!doctype html><html lang="zh-HK"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>VoxCPM2 SFT × CosyVoice Speech Generator</title><style>'
        'body{font:16px/1.65 system-ui,sans-serif;background:#f4f6fa;color:#172535;margin:0}'
        'main{max-width:1250px;margin:auto;padding:28px}h1{font-size:28px}h2{margin-top:32px}'
        '.card,.intro{background:white;border:1px solid #d9e1eb;border-radius:14px;padding:20px;margin:18px 0}'
        '.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:12px}'
        '.variant{padding:15px;background:#f7f9fc;border-radius:10px}audio{width:100%;margin:8px 0}'
        '.small{font-size:13px;color:#566576}.gt{font-size:22px;color:#165d82}.scroll{overflow:auto}'
        'table{border-collapse:collapse;min-width:850px;width:100%;font-size:14px}'
        'td,th{padding:10px;border-bottom:1px solid #dce3ed;text-align:left}'
        'a{color:#126fb2}code{overflow-wrap:anywhere}summary{cursor:pointer}</style><main>',
        '<h1>VoxCPM2 微调 × 旧 Speech Generator</h1>',
        '<div class="intro">6 位固定患者：3 easy、3 hard；每位提供 reference、真实目标原声和 5 个合成版本。'
        '相同 GT、reference、seed 42；没有根据生成结果重抽、重试或删掉失败语音。'
        '<p>两套微调模型使用相同的 2108 条 Setting D TTS 训练语音。这里的目标内容未进入 TTS train；'
        'reference 来自同患者 train。VoxCPM2 使用 reference-only；CosyVoice 使用原生 reference 音频＋其 transcript。'
        '均不显式指定香港粤语。</p><p>音频是完整 WAV；如果模型漏读，播放器仍保留完整模型输出，不以裁剪掩盖。</p></div>',
        '<h2>同一批原声上的配对指标</h2><p>MCD-DTW 越低＝对齐后频谱越接近原声；CAMPPlus 越高＝音色越接近。'
        '时长误差与音高偏差越低越接近。CER 越低＝ASR 越容易读对，不代表越像患者。</p>']
    html.append('<div class="intro"><strong>这批样本的结论：</strong>VoxCPM2 step 330/396 的回译 CER 更低，'
                '而旧 Speech Generator 的平均频谱距离、音色相似度和时长匹配更好。'
                '“更容易识别”与“更像患者”并没有指向同一个模型；本次不因 CER 更低就替换旧 SG。'
                '这不是病理构音保留的单指标认证。</div>')
    for group, title in [('overall', '全部 6 位'), ('easy', '3 位 easy'), ('hard', '3 位 hard')]:
        if group != 'overall':
            html.append(f'<details><summary>{title} · 展开分组指标</summary>')
        html.append(f'<h3>{title}</h3><div class="scroll"><table><tr><th>版本</th><th>CER ↓</th>'
                    '<th>MCD-DTW ↓ (dB)</th><th>CAMPPlus ↑</th><th>时长绝对误差 ↓</th>'
                    '<th>低能量占比差 ↓ (pp)</th><th>F0 中位数差 ↓ (半音)</th></tr>')
        for v in VARIANTS:
            m = result['metrics'][v][group]
            html.append(f'<tr><td>{LABEL[v]}</td><td>{100*m["cer"]:.2f}%</td>'
                f'<td>{m["mcd_dtw_db"]:.2f}</td><td>{m["campplus_to_real"]:.3f}</td>'
                f'<td>{m["duration_abs_error_pct"]:.1f}%</td><td>{m["quiet_fraction_abs_gap_pp"]:.1f}</td>'
                f'<td>{num(m["f0_median_abs_gap_st"])} ({m["f0_median_abs_gap_st_n"]}/{m["n"]})</td></tr>')
        html.append('</table></div>')
        if group != 'overall':
            html.append('</details>')
    html.append('<h2>试听导航</h2><p>' + ' · '.join(
        f'<a href="#{escape(p["real"]["speaker_id"])}">{escape(p["real"]["speaker_id"])} ({p["real"]["patient_group"]})</a>'
        for p in result['pairs'].values()) + '</p>')
    import soundfile as sf
    for pair in result['pairs'].values():
        real = pair['real']
        ref = sf.info(real['reference_audio'])
        html.append(f'<section class="card" id="{escape(real["speaker_id"])}"><h2>{escape(real["speaker_id"])} · {real["patient_group"]}</h2>'
                    f'<p class="gt">目标 GT：{escape(real["clean_gt"])}</p><h3>固定 reference speech</h3>'
                    f'<p>{escape(real["reference_text"])}</p>' + player(real['reference_audio'], ref.duration) + '<div class="grid">')
        for v in VARIANTS:
            r = pair[v]
            html.append(f'<div class="variant"><strong>{LABEL[v]}</strong>' + player(r['audio'], r['seconds']) +
                f'<p>ASR 回译：{escape(r["prediction"])}</p><div class="small">CER {100*r["cer"]:.1f}% · '
                f'MCD {r["mcd_dtw_db"]:.2f} · CAMPPlus {r["campplus_to_real"]:.3f}<br>'
                f'时长比 {r["duration_ratio"]:.2f}×原声 · ASR 与原声回译差 {100*r["asr_to_real_gap"]:.1f}%</div></div>')
        html.append('</div></section>')
    html.append('<details class="card"><summary>指标定义与实验协议</summary><p>这些是声学、内容和韵律的分项诊断，'
        '不是经临床验证的病理构音总分，不拼成一个任意加权分数。easy/hard 是既有患者分组，不是这句话的 CER 分类。</p><pre style="white-space:pre-wrap">' +
        escape(json.dumps(result['protocol'], ensure_ascii=False, indent=2)) + '</pre>'
        '<p>方法来源：<a href="https://github.com/MattShannon/mcd">MCD 作者实现</a> · '
        '<a href="https://pysptk.readthedocs.io/en/latest/generated/pysptk.conversion.sp2mc.html">SPTK mel-cepstrum 定义</a> · '
        '<a href="https://arxiv.org/html/2508.05102v1">构音障碍语音克隆多维评估研究</a></p></details>'
        '<script>document.querySelectorAll("audio").forEach(a=>a.addEventListener("loadedmetadata",()=>{'
        'const ok=Math.abs(a.duration-Number(a.dataset.seconds))<0.1;'
        'a.nextElementSibling.querySelector(".check").textContent=ok?"完整时长已核对":"时长不符，请下载完整 WAV";}));'
        'document.addEventListener("play",e=>{if(e.target.tagName==="AUDIO")document.querySelectorAll("audio").forEach(a=>{if(a!==e.target)a.pause();});},true);'
        '</script></main></html>')
    (folder / 'index.html').write_text('\n'.join(html), encoding='utf-8')
    print(folder / 'index.html', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'cosy', 'assemble', 'acoustics', 'summarize', 'report'])
    globals()[parser.parse_args().stage]()
