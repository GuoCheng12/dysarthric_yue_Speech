#!/usr/bin/env python3
"""Read-only base-ASR comparison on completed synthetic audio; no review decisions."""

import argparse
from pathlib import Path
import time

from mas_selection_dpo_probe import read,save
from transcribe_sg_review import QwenGreedyTranscriber,DECODE_CONFIG,EOS_IDS,MAX_NEW_TOKENS
from therapist_harness.store import sha256_file,sha256_directory


def main(source,root):
    import torch
    from qwen_asr import Qwen3ASRModel
    from transformers import set_seed
    root.mkdir(exist_ok=False)
    generation=read(source/'generation/generation_summary.json')
    base=Path('/data/qwen3-asr/models/Qwen3-ASR-1.7B');digest=sha256_directory(base)
    set_seed(42);start=time.time()
    decoder=object.__new__(QwenGreedyTranscriber)
    decoder.wrapper=Qwen3ASRModel.from_pretrained(str(base),dtype=torch.bfloat16,
        device_map='cuda:0',max_inference_batch_size=1,max_new_tokens=MAX_NEW_TOKENS)
    decoder.wrapper.model.eval().requires_grad_(False)
    loaded=time.time();records=[]
    for item in generation['outputs']:
        path=Path(item['output_wav']);audio_hash=sha256_file(path);assert audio_hash==item['output_wav_sha256']
        begin=time.time();decoded=decoder.transcribe(str(path))
        assert sha256_file(path)==audio_hash
        tokens=decoded['generated_token_ids']
        complete=bool(tokens and tokens[-1] in EOS_IDS and len(tokens)<MAX_NEW_TOKENS)
        record=dict(audio_id=item['sample_id'],audio=str(path),audio_sha256=audio_hash,**decoded,
            complete=complete,actual_transcription=True,elapsed_seconds=time.time()-begin)
        records.append(record);save(root/f'{len(records):03d}.transcription.json',record)
    assert sha256_directory(base)==digest
    save(root/'results.json',dict(status='complete',base_model=str(base),base_model_sha256=digest,
        adapter_applied=False,decode_config=DECODE_CONFIG,records=records,
        model_load_seconds=loaded-start,total_seconds=time.time()-start,
        source_generation=str(source/'generation/generation_summary.json'),
        source_generation_sha256=sha256_file(source/'generation/generation_summary.json'),source_sha256=sha256_file(Path(__file__)),
        scope='Read-only base-ASR diagnostic; recognizer receives audio only. No changes to frozen R-only review, admission, chosen targets or preference candidates. Neither recognizer proves word fidelity.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('source',type=Path);parser.add_argument('root',type=Path)
    args=parser.parse_args();main(args.source.resolve(),args.root.resolve())
