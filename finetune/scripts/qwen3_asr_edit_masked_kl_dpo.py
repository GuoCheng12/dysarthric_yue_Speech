#!/usr/bin/env python3
"""Compare full-sequence and edit+EOS scores with unchanged real retention."""

import argparse
import json
from pathlib import Path
import sys

import qwen3_asr_kl_dpo_storage_v2 as engine
import qwen3_asr_residual_kl_dpo as residual
from qwen3_asr_patient_weighted_kl_dpo import weighted_window_type
from therapist_harness.preference_token_mask import collator_type


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--preference_token_mask',choices=['full','edit_eos'],required=True)
    parser.add_argument('--real_loss_factors',type=Path,required=True)
    args,remaining=parser.parse_known_args()
    output=Path(remaining[remaining.index('--output_dir')+1]);mode=args.preference_token_mask
    weights=json.loads(args.real_loss_factors.read_text());audit={}
    files=[Path(__file__),args.real_loss_factors,Path(__file__).with_name('qwen3_asr_patient_weighted_kl_dpo.py'),
        Path(__file__).parents[2]/'src/therapist_harness/preference_token_mask.py']
    hashes={str(p.resolve()):engine.sha256(p) for p in files};original_write=engine.write_json

    def write_record(path,value):
        if Path(path).name in {'trainer_state.json','residual_execution.json','residual_composition.json'}:
            value=dict(value,preference_token_mask=mode,modified_preference_objective=mode=='edit_eos')
        if Path(path).name=='run_protocol.json':
            value=dict(value,objective='patient-weighted real symmetric KL + synthetic audio preference loss',
                preference_token_mask=mode,
                score_mask='full transcript + EOS' if mode=='full' else 'token edit positions + EOS; sum, no length normalization',
                scope='Explicit score-mask ablation; edit_eos is a modified preference objective, not unchanged full-sequence DPO. Audio and all teacher-forced inputs remain unchanged. Real KL uses full transcript+EOS.',
                real_loss_factors_path=str(args.real_loss_factors.resolve()),
                real_loss_factors_sha256=hashes[str(args.real_loss_factors.resolve())],
                source_sha256={**value['source_sha256'],**hashes})
        if Path(path).name=='reference_log_probs.json':
            original_write(output/'preference_token_masks.json',dict(mode=mode,pairs=audit,
                meaning='Actual tokenizer positions recorded during frozen-reference scoring and checked on each policy scoring call.'))
            value=dict(value,preference_token_mask=mode,mask_manifest_sha256=engine.sha256(output/'preference_token_masks.json'))
        original_write(path,value)

    engine.write_json=write_record
    engine.AudioPreferenceCollator=collator_type(mode,audit)
    engine.SourceBalancedWindows=weighted_window_type(weights['factors'])
    residual.trainer=engine
    sys.argv=[sys.argv[0]]+remaining
    residual.main()
    assert all(engine.sha256(Path(p))==h for p,h in hashes.items())
    assert json.loads((output/'preference_token_masks.json').read_text())['pairs']==audit


if __name__=='__main__':
    main()
