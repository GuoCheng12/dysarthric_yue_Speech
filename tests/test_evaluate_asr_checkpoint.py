from __future__ import annotations

import json
from pathlib import Path

import pytest

from finetune.scripts.evaluate_asr_checkpoint import read_split


def _row(audio: Path, *, utt_id: str = "utt-1") -> dict[str, str]:
    return {
        "utt_id": utt_id,
        "speaker_id": "speaker-1",
        "prompt_id": "prompt-1",
        "clean_gt": "我今日返屋企",
        "zero_shot_bucket": "easy",
        "audio": str(audio),
    }


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_read_split_accepts_the_strict_setting_d_contract(tmp_path: Path) -> None:
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"non-empty-placeholder")
    split = tmp_path / "split.jsonl"
    _write_jsonl(split, [_row(audio)])

    assert read_split(split) == [_row(audio)]


def test_read_split_rejects_duplicate_utterances(tmp_path: Path) -> None:
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"non-empty-placeholder")
    split = tmp_path / "split.jsonl"
    _write_jsonl(split, [_row(audio), _row(audio)])

    with pytest.raises(ValueError, match="duplicate utt_id"):
        read_split(split)


def test_read_split_rejects_missing_audio(tmp_path: Path) -> None:
    split = tmp_path / "split.jsonl"
    _write_jsonl(split, [_row(tmp_path / "missing.wav")])

    with pytest.raises(FileNotFoundError, match="invalid audio"):
        read_split(split)
