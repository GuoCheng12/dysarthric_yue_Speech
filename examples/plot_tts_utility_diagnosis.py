"""Standalone scientific plot of paired seed effects, without winner selection."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('/data/qwen3-asr/records/therapist_mas_setting_d_v1/tts_utility_diagnosis_v1')
LABELS = {
    'mas_hard_w025': 'Raw MAS hard',
    'q_hard_w025': 'Q hard (original text)',
    'mas_hard_language_w025': 'Synthetic updates: language only',
    'mas_hard_audio_w025': 'Synthetic updates: audio only',
    'mas_hard_protected_w025': 'Gradient projection + norm cap',
    'consistency_shuffled_w025': 'Matched shuffled reference choice',
    'consistency_best_w025': 'ASR consistency reference choice',
    'mas_labels_clean_w025': 'Remove synthetic terminal period',
    'mas_hard_decay20': 'Synthetic weight decay, steps 1-20',
    'strict_qc_clean_w005': 'Strict QC + label cleanup + weight .05',
    'strict_qc_room_w005': 'Above + room/noise/level processing',
    'mas_period_w025': 'Both labels with period [vs R-period]',
    'room_full72_w005': 'Full coverage / LR72 [vs R72]',
}


def run():
    data = json.loads((ROOT / 'current_results.json').read_text())
    assert data['verified_runs'] == 48
    fig, ax = plt.subplots(figsize=(11, 7.3))
    y = np.arange(len(LABELS))
    for offset, seed, color in [(-.14, '42', '#3b70a0'), (0, '43', '#be7137'), (.14, '44', '#6b9272')]:
        values = [data['paired_vs_real'][arm]['seed_deltas_pp'][seed] for arm in LABELS]
        ax.scatter(values, y + offset, color=color, s=31, label=f'Seed {seed}', zorder=3)
    means = [data['paired_vs_real'][arm]['mean_delta_pp'] for arm in LABELS]
    ax.scatter(means, y, color='#222222', marker='D', s=27, label='Mean', zorder=4)
    ax.axvline(0, color='#777777', linestyle='--', linewidth=1)
    ax.set_yticks(y, list(LABELS.values()))
    ax.invert_yaxis()
    ax.set_xlabel('TTS minus its matched real-only CER (percentage points; negative is better)', labelpad=12)
    ax.grid(axis='x', alpha=.18)
    ax.set_axisbelow(True)
    for side in ['top', 'right', 'left']:
        ax.spines[side].set_visible(False)
    ax.tick_params(axis='y', length=0, pad=12)
    ax.set_title('TTS utility: all matched augmentation configurations', loc='left', pad=30, weight='bold', fontsize=14)
    ax.legend(loc='upper center', bbox_to_anchor=(.48, 1.06), ncol=4, frameon=False, fontsize=9)
    fig.text(.02, .025, '511 repeatedly used development recordings, 36 patients, 12 texts. Points show fitted-seed results, not confidence intervals.\n50-step controls share real exposure; the final row uses its own full-coverage 72-step control.', fontsize=9, color='#555555')
    fig.tight_layout(rect=(0, .075, 1, 1))
    fig.savefig(ROOT / 'paired_effects.png', dpi=180, facecolor='white')
    fig.savefig(ROOT / 'paired_effects.svg', facecolor='white')
    plt.close(fig)
    print(ROOT / 'paired_effects.png')


if __name__ == '__main__':
    run()
