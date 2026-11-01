from __future__ import annotations

from pathlib import Path

import pytest

from test_therapist_harness import evidence, proposal, protocol
from therapist_harness.evidence import extract_evidence
from therapist_harness.grounding import stimulus_cluster_grounding
from therapist_harness.schema import EvidencePack, Stimulus


def _surface_pack(tmp_path: Path) -> EvidencePack:
    payload = evidence(protocol(tmp_path)).model_dump(mode="json")
    payload["error_clusters"][0].update(
        level="character",
        operation="substitution",
        reference_unit="無",
        hypothesis_unit="有",
        reference_surface_units=["冇"],
        hypothesis_surface_units=["有"],
        rate_semantics="deterministic_character_alignment_rate",
    )
    return EvidencePack.model_validate(payload)


def test_character_grounding_uses_attested_spoken_carrier(tmp_path: Path) -> None:
    pack = _surface_pack(tmp_path)
    payload = proposal(pack).stimuli[0].model_dump(mode="json")

    payload["text"] = "我今日冇時間返公司。"
    spoken = stimulus_cluster_grounding(Stimulus.model_validate(payload), pack)
    assert spoken["e1"] == (1, False)

    payload["text"] = "我今日無時間返公司。"
    scoring_alias = stimulus_cluster_grounding(Stimulus.model_validate(payload), pack)
    assert scoring_alias["e1"] == (0, False)

    payload["text"] = "我有啲錢，但冇時間。"
    contrast = stimulus_cluster_grounding(Stimulus.model_validate(payload), pack)
    assert contrast["e1"] == (1, True)


def test_legacy_character_evidence_requires_explicit_regeneration(tmp_path: Path) -> None:
    pack = _surface_pack(tmp_path)
    payload = pack.model_dump(mode="json")
    del payload["error_clusters"][0]["reference_surface_units"]
    del payload["error_clusters"][0]["hypothesis_surface_units"]
    legacy = EvidencePack.model_validate(payload)

    with pytest.raises(ValueError, match="regenerate legacy evidence"):
        stimulus_cluster_grounding(proposal(legacy).stimuli[0], legacy)


@pytest.mark.parametrize(
    ("surface", "hypothesis", "normalized", "actionable"),
    [("冇", "有", "無", True), ("咩", "孭", "乜", False)],
)
def test_evidence_preserves_surface_carriers_and_their_actionability(
    surface: str, hypothesis: str, normalized: str, actionable: bool
) -> None:
    rows = [
        dict(
            utt_id=f"u{index}",
            speaker_id=f"p{index}",
            prompt_id=f"c{index}",
            zero_shot_bucket="hard",
            clean_gt=f"{prefix}{surface}時間。",
            prediction=f"{prefix}{hypothesis}時間。",
        )
        for index, prefix in enumerate(("我", "佢"))
    ]
    extracted = extract_evidence(
        rows,
        rows,
        rows,
        current_prediction_column="prediction",
        m0_prediction_column="prediction",
        zero_shot_prediction_column="prediction",
    )
    cluster = next(item for item in extracted.error_clusters if item.level == "character")
    assert cluster.reference_unit == normalized
    assert cluster.reference_surface_units == [surface]
    assert cluster.hypothesis_surface_units == [hypothesis]
    assert cluster.tts_actionable is actionable
