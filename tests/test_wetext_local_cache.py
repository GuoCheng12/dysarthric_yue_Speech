from __future__ import annotations

from pathlib import Path

import pytest

from synthesis.cosyvoice3_patient_sft import wetext_local_cache as local_cache
from therapist_harness.schema import canonical_sha256


def _cache(root: Path, marker: bytes = b"local-wetext-v1") -> Path:
    for index, relative_path in enumerate(local_cache.WETEXT_CRITICAL_FILES):
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(marker + b":" + str(index).encode())
    return root


def test_binding_makes_normalizer_use_only_verified_local_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import wetext
    import wetext.wetext as wetext_module

    cache_dir = _cache(tmp_path / "wetext")
    network_calls = []
    opened_fsts = []

    def forbidden_network(*args, **kwargs):
        network_calls.append((args, kwargs))
        raise AssertionError("network snapshot_download must not be called")

    monkeypatch.setattr(wetext_module, "snapshot_download", forbidden_network)
    monkeypatch.setattr(
        wetext_module,
        "normalizer",
        lambda path: opened_fsts.append(path) or (lambda text: text),
    )
    evidence = local_cache.bind_local_wetext_snapshot(cache_dir)
    wetext.Normalizer()

    assert network_calls == []
    assert evidence["cache_dir"] == str(cache_dir.resolve())
    assert evidence["network_allowed"] is False
    assert len(opened_fsts) == 4
    assert all(path.startswith(str(cache_dir.resolve())) for path in opened_fsts)
    assert local_cache.wetext_cache_evidence(cache_dir) == evidence
    with pytest.raises(RuntimeError, match="network access is disabled"):
        wetext_module.snapshot_download("some/other-online-repository")

    installation = local_cache.wetext_installation_evidence()
    assert installation["wetext_version"] == "0.0.4"
    assert installation["kaldifst_version"] == "1.8.0"
    assert {row["relative_path"] for row in installation["python_sources"]} == {
        "wetext/wetext.py",
        "wetext/token_parser.py",
    }


def test_missing_cache_fails_before_importing_wetext(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    imported = False

    def unexpected_import(_name: str):
        nonlocal imported
        imported = True
        raise AssertionError("wetext import must occur only after cache validation")

    monkeypatch.setattr(local_cache.importlib, "import_module", unexpected_import)
    with pytest.raises(FileNotFoundError, match="cache directory is missing"):
        local_cache.bind_local_wetext_snapshot(tmp_path / "missing")
    assert imported is False


def test_wetext_content_change_changes_protocol_fingerprint(tmp_path: Path) -> None:
    cache_dir = _cache(tmp_path / "wetext")
    first = local_cache.wetext_cache_evidence(cache_dir)
    (cache_dir / "zh/tn/tagger.fst").write_bytes(b"changed-critical-fst")
    second = local_cache.wetext_cache_evidence(cache_dir)

    assert first["critical_content_sha256"] != second["critical_content_sha256"]
    assert canonical_sha256({"wetext_cache": first}) != canonical_sha256(
        {"wetext_cache": second}
    )
