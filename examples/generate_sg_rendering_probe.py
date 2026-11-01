#!/usr/bin/env python3
"""Explicit TTS input rendering while preserving the intended three-text target."""

import json
from pathlib import Path
import generate_reviewed_sg_recovery_v2 as recovery


def main():
    base=recovery.base;args=base.parse_args()
    from synthesis.cosyvoice3_patient_sft import synthesize_manifest as tts
    choices={x['sample_id']:x for x in json.loads(args.generation_plan.read_text())['choices']}
    original=tts._stage_audio_once
    def stage(sample,model):
        choice=choices[sample.sample_id];rendered=choice['rendered_target_text']
        class RenderedInput:
            sample_rate=model.sample_rate
            def inference_zero_shot(self,target,prompt,reference,**kwargs):
                assert target==sample.target_text
                return model.inference_zero_shot(rendered,prompt,reference,**kwargs)
        record,temporary=original(sample,RenderedInput())
        record.update(intended_target_text=sample.target_text,actual_tts_input_text=rendered,
            text_rendering=choice['text_rendering'],rendering_caveat=choice['rendering_caveat'])
        return record,temporary
    tts._stage_audio_once=stage
    variant=dict(source=str(Path(__file__).resolve()),sha256=base.sha256_file(Path(__file__)),
        scope='Only the actual TTS text argument is rendered according to the explicit plan. Generation records contain both the intended target and actual input. ASR receives audio only; the unchanged Agent compares intended target, reference text and transcription. Diagnostic audio is not automatically training-eligible.')
    writer=base.write_immutable_json
    def write(path,payload):
        if Path(path).name in {'generation_config.json','runtime.json','generation_summary.json','generation_failure.json'}:
            payload=dict(payload,text_rendering_execution=variant)
        writer(path,payload)
    base.write_immutable_json=write
    return recovery.main()


if __name__=='__main__':raise SystemExit(main())
