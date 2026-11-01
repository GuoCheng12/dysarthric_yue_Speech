"""Two independently attested orthographic equivalences, applied to every arm."""
from collections import defaultdict
import json
from pathlib import Path
import statistics

from tts_increment_control import ROOT as CONTROL,TEST,read,rows,sha256_file
from therapist_harness.text import text_normalize,edit_distance
from therapist_harness.text_scoring_v2 import text_normalize_v2 as normalize_variants

ROOT=CONTROL.parent/'tts_utility_diagnosis_v1'
def run():
    fixed={
        'old_real2117':CONTROL/'real_q',
        'old_r1541':CONTROL.parent/'tts_increment_r_only1541_v1/r_only',
        'old_real2117_qhard':CONTROL.parent/'real2117_hardtts576_v1/real_plus_hard_tts',
        'old_real2117_mas_old':CONTROL.parent/'real2117_mas576_hardref_v1/mas_old_reference',
        'old_real2117_mas_hard':CONTROL.parent/'real2117_mas576_hardref_v1/mas_hard_reference',
    }
    fixed.update({p.name:p for p in (ROOT/'additive').glob('*') if p.is_dir()})
    scope=rows(TEST);results=defaultdict(dict)
    for arm,root in fixed.items():
        for path in sorted(root.glob('seed_*/result.json')):
            result=read(path);pred=rows(path.parent/f"evaluation/merged_step_{result['step']}/predictions.jsonl")
            assert [r['utt_id'] for r in pred]==[r['utt_id'] for r in scope]
            old=[];new=[]
            for r in pred:
                old.append(edit_distance(text_normalize(r['clean_gt']),text_normalize(r['prediction']))/len(text_normalize(r['clean_gt'])))
                new.append(edit_distance(normalize_variants(r['clean_gt']),normalize_variants(r['prediction']))/len(normalize_variants(r['clean_gt'])))
            assert abs(statistics.mean(old)-result['metrics']['overall']['mean_utterance_cer'])<1e-12
            groups={}
            for g in ['overall','vlink247','other_patients']:
                ii=[i for i,r in enumerate(pred) if g=='overall' or (r['speaker_id']=='vlink247')==(g=='vlink247')]
                groups[g]={'legacy_cer':statistics.mean(old[i] for i in ii)*100,
                    'variant_cer':statistics.mean(new[i] for i in ii)*100}
            groups['changed_utterances']=sum(a!=b for a,b in zip(old,new))
            results[arm][result['seed']]=groups
    output={'rule':{'鋭':'銳','枴':'柺','拐':'柺'},'scope':'orthographic audit only; legacy metric retained',
        'sources':['https://dict.variants.moe.edu.tw/dictView.jsp?ID=47272&la=0',
            'https://words.hk/zidin/%E6%8B%90%E6%9D%96'],
        'source_sha256':sha256_file(Path(__file__)),
        'normalizer_sha256':sha256_file(Path(__file__).resolve().parents[1]/'src/therapist_harness/text_scoring_v2.py'),
        'results':results}
    out=ROOT/'scoring';out.mkdir(exist_ok=True)
    (out/'variants.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    lines=['# CER 异体字计分审计','',
        '仅统一有字典依据的銳/鋭和拐/枴/柺；对全部GT及全部模型预测应用相同规则，不修改原评测结果。',
        '鑲/將、褂/掛等不能仅因接近同音就统一。','',
        '| 组别 | Seeds | 原CER | 异体字统一后CER | 差值 |','|---|---|---:|---:|---:|']
    for arm,ss in results.items():
        a=statistics.mean(r['overall']['legacy_cer'] for r in ss.values())
        b=statistics.mean(r['overall']['variant_cer'] for r in ss.values())
        lines.append(f'| {arm} | '+','.join(map(str,ss))+f' | {a:.4f} | {b:.4f} | {b-a:+.4f} |')
    (out/'variants.zh.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)


if __name__=='__main__':run()
