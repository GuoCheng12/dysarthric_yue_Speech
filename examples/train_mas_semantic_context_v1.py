#!/usr/bin/env python3
"""Frozen matched trainer, directed to the new semantic-context execution."""
import mas_semantic_context_probe_v1 as experiment
import train_mas_contextual_recipe_dpo_v1 as trainer

trainer.DATA=experiment.ROOT
trainer.ROOT=experiment.ROOT/'training'
trainer.PLAN=experiment.ROOT/'training_plan'

if __name__=='__main__':trainer.main()
