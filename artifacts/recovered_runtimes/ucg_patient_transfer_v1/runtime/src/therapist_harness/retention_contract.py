"""Explicit auxiliary real-KL facts alongside the primary ASR preference table."""

import json
import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RetentionModel(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, frozen=True)


class RetentionDesign(RetentionModel):
    schema_version: Literal['real_KL_edit_DPO_v1']
    real_manifest: str
    real_manifest_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    real_loss_factors: str
    real_loss_factors_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    frozen_reference_asr_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    expected_frozen_base_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    real_per_step: int = Field(ge=1)
    kl_weight: float = Field(gt=0)
    preference_token_mask: Literal['edit_eos'] = 'edit_eos'
    chosen_text_transform: Literal['strip_terminal_sentence_punctuation_v2'] = 'strip_terminal_sentence_punctuation_v2'


class RealKLRetention(RetentionModel):
    objective: Literal['symmetric_kl_to_frozen_initial_R_only'] = 'symmetric_kl_to_frozen_initial_R_only'
    teacher_asr_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    manifest_file_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    loss_factors_file_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    teacher_cache_metadata_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    audio_count: int = Field(ge=1)
    unique_audio_exposed: int = Field(ge=1)
    training_exposures: int = Field(ge=1)
    total_loss_weight: float = Field(gt=0, allow_inf_nan=False)
    conditional_history: Literal['real_training_GT_transcript_plus_EOS'] = 'real_training_GT_transcript_plus_EOS'
    real_cross_entropy: Literal[False] = False
    semantics: Literal['auxiliary_real_audio_not_real_preference_pairs'] = 'auxiliary_real_audio_not_real_preference_pairs'

    @model_validator(mode='after')
    def coverage(self):
        if self.unique_audio_exposed > min(self.audio_count, self.training_exposures):
            raise ValueError('real-KL coverage exceeds its actual pool or exposure count')
        return self


def load_retention_design(protocol):
    from .store import sha256_file

    config = protocol.preference_loop
    path = Path(config.retention_design_path)
    if sha256_file(path) != config.retention_design_sha256:
        raise ValueError('frozen real-KL design changed')
    design = RetentionDesign.model_validate_json(path.read_text())
    if design.frozen_reference_asr_sha256 != protocol.initial_adapter.sha256:
        raise ValueError('real KL and DPO must retain the initial R-only ASR reference')
    for name in ['real_manifest', 'real_loss_factors']:
        if sha256_file(Path(getattr(design, name))) != getattr(design, name + '_sha256'):
            raise ValueError('real-KL input changed: ' + name)
    return design


def validate_retention_rows(protocol, design):
    from .store import sha256_file
    from .text import text_normalize

    def rows(path):
        return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

    pool = {x['utt_id']: x for x in rows(protocol.real_train.path)}
    selected = rows(design.real_manifest)
    factors = json.loads(Path(design.real_loss_factors).read_text())['factors']
    ids = {x['audio_id'] for x in selected}
    if len(ids) != len(selected) or set(factors) != ids:
        raise ValueError('real-KL audio identities/factors must cover the same unique pool')
    for row in selected:
        source = pool.get(row['audio_id'])
        if source is None or row['source'] != 'real' or 'rejected' in row:
            raise ValueError('real KL requires actual training-pool audio, without preference rejections')
        if Path(row['audio']).resolve() != Path(source['audio']).resolve():
            raise ValueError('real-KL identity points to different training audio')
        if row['audio_sha256'] != sha256_file(Path(row['audio'])):
            raise ValueError('real-KL audio changed')
        if text_normalize(row['chosen']) != text_normalize(source['clean_gt']):
            raise ValueError('real-KL history differs from real training ground truth')
        if row.get('prompt') != '' or row.get('language') != 'Cantonese':
            raise ValueError('real-KL audio must use empty context and Cantonese')
        if not isinstance(factors[row['audio_id']], (float, int)) or not math.isfinite(factors[row['audio_id']]) or factors[row['audio_id']] <= 0:
            raise ValueError('real-KL factors must be positive')
    return selected, factors
