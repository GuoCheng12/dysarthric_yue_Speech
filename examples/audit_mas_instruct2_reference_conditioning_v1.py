#!/usr/bin/env python3
"""Inspect actual fixed SG frontend inputs; no audio generation or ASR update."""
import ast
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'src')]
from therapist_harness.store import sha256_file,write_immutable_json
from therapist_harness.generation_design import INSTRUCTIONS
from synthesis.cosyvoice3_patient_sft.synthesize_manifest import load_fixed_runtime

B=Path('/data/qwen3-asr/records/therapist_mas_setting_d_v1/mas_design_exploration_20260908_v1')
ROOT=B/'instruct2_reference_conditioning_audit_v1'


def tensor_record(value):
    x=value.detach().cpu().contiguous()
    return dict(shape=list(x.shape),dtype=str(x.dtype),sha256=hashlib.sha256(x.numpy().tobytes()).hexdigest())


def main():
    ROOT.mkdir(exist_ok=False)
    start=time.time();model,runtime=load_fixed_runtime();load_seconds=time.time()-start
    print(json.dumps(dict(stage='fixed_SG_loaded',seconds=load_seconds)),flush=True)
    cases={arm:json.loads((B/'literal_matched_SG24_v1'/arm/'manifest.jsonl').read_text().splitlines()[0]) for arm in ['A','B']}
    assert cases['A']['target_text']==cases['B']['target_text']
    assert cases['A']['reference_wav_sha256']!=cases['B']['reference_wav_sha256']
    observations={}
    for mode in ['instruct2','zero_shot']:
        observations[mode]={}
        for arm,row in cases.items():
            assert sha256_file(Path(row['reference_wav']))==row['reference_wav_sha256']
            instruction=INSTRUCTIONS['cantonese_zh']
            if mode=='instruct2':
                inputs=model.frontend.frontend_instruct2(row['target_text'],instruction,row['reference_wav'],model.sample_rate,'')
            else:
                inputs=model.frontend.frontend_zero_shot(row['target_text'],instruction+row['reference_text'],row['reference_wav'],model.sample_rate,'')
            observations[mode][arm]={k:tensor_record(v) for k,v in inputs.items()}
    live_function=model.model.llm.inference
    source=inspect.getsource(live_function)
    import textwrap
    tree=ast.parse(textwrap.dedent(source))
    embedding_reads=[n.id for n in ast.walk(tree) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load) and n.id=='embedding']
    assert model.model.llm.__class__.__name__=='CosyVoice3LM'
    assert live_function.__qualname__=='Qwen2LM.inference' and not embedding_reads
    a,b=observations['instruct2']['A'],observations['instruct2']['B']
    assert 'llm_prompt_speech_token' not in a and 'llm_prompt_speech_token' not in b
    assert all(a[k]==b[k] for k in ['text','text_len','prompt_text','prompt_text_len'])
    assert a['flow_prompt_speech_token']!=b['flow_prompt_speech_token']
    z=observations['zero_shot']
    assert all('llm_prompt_speech_token' in z[arm] and z[arm]['llm_prompt_speech_token']['shape'][1]>0 for arm in ['A','B'])
    assert z['A']['prompt_text']!=z['B']['prompt_text']
    files=[Path(__file__),Path(inspect.getfile(model.frontend.__class__)),Path(inspect.getfile(live_function)),
        Path(inspect.getfile(model.model.__class__)),REPO/'src/therapist_harness/generation_design.py',
        REPO/'synthesis/cosyvoice3_patient_sft/synthesize_manifest.py']
    result=dict(status='passed_actual_fixed_SG_frontend_conditioning_audit',load_seconds=load_seconds,
        total_seconds=time.time()-start,model_class=model.__class__.__name__,LM_class=model.model.llm.__class__.__name__,
        actual_LM_inference=live_function.__qualname__,speaker_embedding_argument_read_in_LM=bool(embedding_reads),
        frontend_observations=observations,instruct2_effective_LM_conditioning_equal_across_references=True,
        instruct2_flow_conditioning_differs=True,zero_shot_retains_reference_speech_tokens_and_reference_text=True,
        generated_audio=0,ASR_updates=0,runtime=runtime,
        interpretation='For this fixed CosyVoice3LM runtime and same target/instruction, instruct2 removes reference speech tokens from the LM path; inherited Qwen2LM.inference does not consume its embedding argument. The flow path still consumes reference speech features/tokens and speaker embedding, so references can change acoustics and ASR outputs. This is an input/dependency audit, not a proof that generated impairment cannot transfer, that audio tokens are bitwise identical, or that zero_shot improves real CER.',
        source_sha256={str(f):sha256_file(f) for f in files})
    write_immutable_json(ROOT/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['frontend_observations','runtime','source_sha256']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
