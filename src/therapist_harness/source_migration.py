"""Verify historical source bytes through an explicit immutable source archive."""

import json
from pathlib import Path

from .store import sha256_file


def verify_historical_sources(source_sha256, archive_path):
    archive_path = Path(archive_path)
    archive = json.loads(archive_path.read_text())
    if archive['status'] != 'complete' or archive['old_sources_json_rewritten']:
        raise ValueError('historical source archive is not an intact recorded snapshot')
    archived = {}
    for name, expected in source_sha256.items():
        current = Path(name)
        if current.is_file() and sha256_file(current) == expected:
            continue
        item = archive['entries'].get(name)
        if item is None or item['sha256'] != expected or sha256_file(Path(item['archive_path'])) != expected:
            raise ValueError('historical input/source changed without a matching archived version: ' + name)
        archived[name] = dict(expected_sha256=expected, verified_path=item['archive_path'])
    return dict(status='passed', source_count=len(source_sha256), archive_path=str(archive_path),
        archive_sha256=sha256_file(archive_path), historical_source_resolutions=archived,
        old_sources_json_rewritten=False,
        scope='Historical inputs and exact producer source bytes verified. Changed current source uses the recorded old bytes, not a rewritten source hash or a claim of new execution.')
