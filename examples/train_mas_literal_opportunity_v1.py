#!/usr/bin/env python3
"""Bind the verified fresh-reference DPO trainer to the new selector experiment."""
import mas_literal_opportunity_selector_v1 as experiment
import train_mas_contextual_recipe_dpo_v1 as trainer

trainer.DATA = experiment.ROOT
trainer.ROOT = experiment.ROOT / 'training'
trainer.PLAN = experiment.ROOT / 'training_plan'

if __name__ == '__main__':
    trainer.main()
