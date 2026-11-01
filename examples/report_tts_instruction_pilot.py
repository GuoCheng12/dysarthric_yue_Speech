"""Summarize all paired diagnostic generations and export a small listening page."""
import base64
from collections import defaultdict
import html
import json
from pathlib import Path
import statistics

from therapist_harness.text import text_normalize
from tts_instruction_pilot import ROOT,MODES,rows,save


def run():
    cfg=json.loads((ROOT/'protocol.json').read_text());ids={p['sample_id'] for p in cfg['pairs']}
    results={};cases=defaultdict(dict)
    for judge,oldfile,newfile in [
        ('real_asr',ROOT.parent/'screening/asr/predictions.jsonl',ROOT/'real_asr/predictions.jsonl'),
        ('base_asr',ROOT.parent/'screening/base_asr/predictions.jsonl',ROOT/'base_asr/predictions.jsonl')]:
        old=[r for r in rows(oldfile) if r['variant']=='mas_hard' and r['pair_id'] in ids]
        new=rows(newfile);assert len(old)==19 and len(new)==57
        for r in old:r['variant']='zero_shot_generic'
        by_mode=defaultdict(list)
        for r in old+new:
            assert r['pair_id'] in ids
            gt,hyp,ref=map(text_normalize,[r['clean_gt'],r['prediction'],r['reference_text']])
            matches=sorted({ref[i:i+4] for i in range(max(0,len(ref)-3)) if ref[i:i+4] in hyp and ref[i:i+4] not in gt})
            r={**r,'reference_fragment_matches':matches}
            by_mode[r['variant']].append(r);cases[r['pair_id']][judge+'__'+r['variant']]=r
        results[judge]={}
        for mode,rr in by_mode.items():
            assert len(rr)==19 and len({x['pair_id'] for x in rr})==19
            results[judge][mode]={'n':len(rr),'cer_pct':100*statistics.mean(r['cer'] for r in rr),
                'perfect':sum(r['cer']==0 for r in rr),'over_50pct':sum(r['cer']>.5 for r in rr),
                'reference_fragment_clues':sum(bool(r['reference_fragment_matches']) for r in rr)}
    save(ROOT/'results.json',{'selection':'19 targeted known failures; not a population estimate',
        'judges':'same architecture, base and R-only tuned; manual listening remains useful',
        'metrics':results,'cases':cases})
    selected=['mas0042','mas0058','mas0022','mas0333']
    parts=['<!doctype html><meta charset="utf-8"><title>TTS 生成方式诊断试听</title>',
        '<style>body{font:16px system-ui;max-width:1050px;margin:35px auto;padding:0 20px}article{border-top:1px solid #ccc;padding:20px 0}.row{padding:12px;background:#f5f6f8;margin:10px 0}audio{width:100%}small{color:#555}</style>',
        '<h1>TTS 生成方式诊断试听</h1><p>同一患者 reference 与目标文本。下方文字是两个 ASR 的预测，不是人工确认转录。完整定量分析包括全部 19 个预先选定的问题案例。</p>']
    def player(path):
        data=base64.b64encode(Path(path).read_bytes()).decode('ascii')
        return f'<audio controls preload="none" src="data:audio/wav;base64,{data}"></audio>'
    names={'zero_shot_generic':'原始 zero-shot','zero_shot_cantonese':'zero-shot + 粤语指令',
        'instruct2_generic':'instruct2 + 通用指令','instruct2_cantonese':'instruct2 + 粤语指令'}
    for key in selected:
        c=cases[key];old=c['real_asr__zero_shot_generic']
        parts.append(f'<article><h2>{key} · {html.escape(old["speaker_id"])}</h2><p>目标：{html.escape(old["clean_gt"])}</p>')
        parts.append('<details><summary>患者 reference：'+html.escape(old['reference_text'])+'</summary>'+player(old['reference_audio'])+'</details>')
        for mode,name in names.items():
            a,b=c['real_asr__'+mode],c['base_asr__'+mode]
            parts.append(f'<div class="row"><strong>{name}</strong>'+player(a['audio'])+f'<p>R-only ASR：{html.escape(a["prediction"])}</p><small>Base ASR：{html.escape(b["prediction"])}</small></div>')
        parts.append('</article>')
    with (ROOT/'listening.html').open('x') as f:f.write('\n'.join(parts))
    print(json.dumps(results,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':run()
