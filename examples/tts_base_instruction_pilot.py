"""Disable only SG LoRA to isolate whether adaptation caused novel-text failures."""
import argparse
from pathlib import Path

import tts_instruction_pilot as pilot

ORIGINAL=pilot.ROOT
ROOT=ORIGINAL.parent/'base_instruction_pilot'
MODES=('base_zero_shot_generic','base_instruct2_cantonese')


class BaseProxy:
    def __init__(self,model,mode):self.model=model;self.mode=mode;self.sample_rate=model.sample_rate
    def inference_zero_shot(self,target,prompt,reference,**kwargs):
        if self.mode=='base_zero_shot_generic':
            return self.model.inference_zero_shot(target,prompt,reference,**kwargs)
        assert self.mode=='base_instruct2_cantonese'
        return self.model.inference_instruct2(target,pilot.CANTONESE,reference,**kwargs)


def prepare():
    import json
    cfg=json.loads((ORIGINAL/'protocol.json').read_text())
    cfg.update(modes=list(MODES),intervention='Disable all SG LoRA layers; frozen base weights and same hard references, targets and seeds',
        comparison='Base zero-shot vs SG zero-shot; base instruct2 Cantonese vs SG instruct2 Cantonese')
    cfg['source_sha256'][str(Path(__file__))]=pilot.digest(__file__)
    pilot.save(ROOT/'protocol.json',cfg)


def run():
    from synthesis.cosyvoice3_patient_sft import synthesize_manifest as tts
    original_load=tts.load_fixed_runtime
    def load():
        model,runtime=original_load()
        from peft.tuners.tuners_utils import BaseTunerLayer
        layers=[m for m in model.model.llm.modules() if isinstance(m,BaseTunerLayer)]
        assert layers
        for layer in layers:layer.enable_adapters(False)
        assert all(layer.disable_adapters for layer in layers)
        runtime['experimental_intervention']={'sg_lora_disabled':True,'disabled_layers':len(layers),
            'base_weights':'unchanged; adapter residuals bypassed at inference'}
        return model,runtime
    tts.load_fixed_runtime=load
    pilot.ROOT=ROOT;pilot.MODES=MODES;pilot.ModeProxy=BaseProxy
    pilot.synthesize()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','synthesize'],required=True)
    args=p.parse_args();prepare() if args.stage=='prepare' else run()
