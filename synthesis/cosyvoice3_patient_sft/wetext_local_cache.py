"""Strict local-only binding for the wetext text-normalization snapshot."""

from __future__ import annotations

import hashlib
import importlib
import json
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any


DEFAULT_WETEXT_CACHE = Path(
    "/home/devbox/.cache/modelscope/hub/models/pengzhendong/wetext"
)
WETEXT_LOCAL_CACHE_HELPER_SOURCE = Path(__file__).resolve()
WETEXT_REPOSITORY_ID = "pengzhendong/wetext"
WETEXT_CRITICAL_FILES = (
    "configuration.json",
    "en/tn/tagger.fst",
    "en/tn/verbalizer.fst",
    "zh/tn/tagger.fst",
    "zh/tn/verbalizer.fst",
)
WETEXT_PYTHON_SOURCE_FILES = (
    "wetext/wetext.py",
    "wetext/token_parser.py",
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def wetext_cache_evidence(cache_dir: Path) -> dict[str, Any]:
    resolved = cache_dir.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(
            f"required local wetext cache directory is missing: {resolved}"
        )
    files = []
    for relative_path in WETEXT_CRITICAL_FILES:
        path = resolved / relative_path
        if not path.is_file():
            raise FileNotFoundError(
                f"required local wetext cache file is missing: {path}"
            )
        size = path.stat().st_size
        if size <= 0:
            raise ValueError(f"required local wetext cache file is empty: {path}")
        files.append(
            {
                "relative_path": relative_path,
                "size": size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "version": 1,
        "repository_id": WETEXT_REPOSITORY_ID,
        "cache_dir": str(resolved),
        "network_allowed": False,
        "critical_files": files,
        "critical_content_sha256": canonical_hash(files),
    }


def wetext_installation_evidence() -> dict[str, Any]:
    distribution = importlib_metadata.distribution("wetext")
    sources = []
    for relative_path in WETEXT_PYTHON_SOURCE_FILES:
        path = Path(distribution.locate_file(relative_path)).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"installed wetext source is missing: {path}")
        sources.append(
            {
                "relative_path": relative_path,
                "path": str(path),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "wetext_version": importlib_metadata.version("wetext"),
        "kaldifst_version": importlib_metadata.version("kaldifst"),
        "python_sources": sources,
        "python_source_fingerprint": canonical_hash(sources),
    }


def bind_local_wetext_snapshot(
    cache_dir: Path,
    *,
    expected_critical_content_sha256: str | None = None,
) -> dict[str, Any]:
    """Bind wetext's module-global downloader to one verified local directory."""

    evidence = wetext_cache_evidence(cache_dir)
    if (
        expected_critical_content_sha256 is not None
        and evidence["critical_content_sha256"] != expected_critical_content_sha256
    ):
        raise RuntimeError(
            "local wetext cache content hash differs from the evaluation protocol: "
            f"expected {expected_critical_content_sha256}, found "
            f"{evidence['critical_content_sha256']}"
        )

    wetext_module = importlib.import_module("wetext.wetext")
    resolved = evidence["cache_dir"]

    def local_snapshot_download(repository_id: str, *_args: Any, **_kwargs: Any) -> str:
        if repository_id != WETEXT_REPOSITORY_ID:
            raise RuntimeError(
                "network access is disabled; local wetext binding refuses "
                f"repository {repository_id!r}"
            )
        # Recheck the files at every Normalizer construction so deletion after
        # initial binding cannot silently fall through to an online download.
        current = wetext_cache_evidence(Path(resolved))
        if current["critical_content_sha256"] != evidence["critical_content_sha256"]:
            raise RuntimeError("local wetext cache changed after offline binding")
        return resolved

    wetext_module.snapshot_download = local_snapshot_download
    return evidence
