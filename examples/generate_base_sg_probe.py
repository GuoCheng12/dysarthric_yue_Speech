#!/usr/bin/env python3
"""Independent SG probe with the loaded patient LoRA explicitly disabled."""

from pathlib import Path
import generate_reviewed_sg_recovery_v2 as recovery


def main():
    base=recovery.base
    from synthesis.cosyvoice3_patient_sft import synthesize_manifest as tts
    variant=dict(name='base_cosyvoice3_with_patient_lora_disabled',patient_adapter_active=False,
        source=str(Path(__file__).resolve()),source_sha256=base.sha256_file(Path(__file__)),
        scope='Independent generation ablation. Original base files and patient checkpoint are hash-validated and loaded; every patient LoRA module is then disabled. Reference, target and decoding recipe are unchanged.')
    original_load=tts.load_fixed_runtime
    def load():
        model,runtime=original_load()
        # Import only after the existing SG loader installs its Transformers overlay.
        from peft.tuners.tuners_utils import BaseTunerLayer
        modules=[m for m in model.model.llm.modules() if isinstance(m,BaseTunerLayer)]
        if not modules:raise RuntimeError('expected loaded patient LoRA layers for this controlled ablation')
        for module in modules:module.enable_adapters(enabled=False)
        assert all(m.disable_adapters for m in modules)
        runtime=dict(runtime,generation_variant={**variant,'disabled_lora_modules':len(modules)})
        runtime['lora']=dict(runtime['lora'],active=False)
        return model,runtime
    tts.load_fixed_runtime=load
    writer=base.write_immutable_json
    def write(path,payload):
        if Path(path).name in {'generation_config.json','generation_summary.json','generation_failure.json'}:
            payload=dict(payload,generation_variant=variant)
        writer(path,payload)
    base.write_immutable_json=write
    return recovery.main()


if __name__=='__main__':raise SystemExit(main())
