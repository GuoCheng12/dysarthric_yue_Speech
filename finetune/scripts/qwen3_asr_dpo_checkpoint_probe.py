#!/usr/bin/env python3
"""Save steps 3/6/9 while replaying the unchanged 12-step DPO trainer."""

from pathlib import Path

import torch
import qwen3_asr_audio_dpo as trainer


def main():
    args=trainer.parse_args()
    if args.max_steps!=12 or args.objective!='dpo':
        raise ValueError('This bounded probe requires the original 12-step DPO configuration')
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=False)
    files=[Path(__file__),Path(trainer.__file__),Path(trainer.__file__).parents[2]/'src/therapist_harness/dpo_training.py']
    sources={str(p.resolve()):trainer.sha256(p) for p in files}
    trainer.write_json(output/'checkpoint_probe_execution.json',dict(source_sha256=sources,
        save_steps=[3,6,9],final_step=12,scope='Observation-only post-step saves; loss, data, schedule and optimizer unchanged'))
    load=trainer.load_adapter;state={}
    def capture(*a,**kw):
        model,processor=load(*a,**kw)
        if any(p.requires_grad for p in model.parameters()):state['policy']=model
        return model,processor
    trainer.load_adapter=capture
    adam=torch.optim.AdamW;step=0
    def save_after_step(optimizer,args_,kwargs):
        nonlocal step
        step+=1
        if step not in [3,6,9]:return
        from qwen3_asr_sft import copy_required_hf_files_for_qwen_asr
        checkpoint=output/f'checkpoint-{step}'
        state['policy'].save_pretrained(checkpoint,safe_serialization=True)
        copy_required_hf_files_for_qwen_asr(args.model_path,str(checkpoint))
        trainer.write_json(checkpoint/'trainer_state.json',dict(global_step=step,
            max_steps=12,objective='dpo',scope='Intermediate adapter for evaluation; optimizer continues in the same invocation'))
        print(f'[checkpoint probe] saved step {step}',flush=True)
    class CheckpointAdamW(adam):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw)
            self.register_step_post_hook(save_after_step)
    torch.optim.AdamW=CheckpointAdamW
    try:trainer.main()
    finally:
        torch.optim.AdamW=adam;trainer.load_adapter=load
    assert all(trainer.sha256(p)==digest for p,digest in sources.items())


if __name__=='__main__':main()
