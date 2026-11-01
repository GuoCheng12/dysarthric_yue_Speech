#!/usr/bin/env python3
"""Versioned SG entry accepting deliberate semantic oddness in BOTH arms."""
from pathlib import Path
import generate_reviewed_sg_recovery_v2 as recovery
import therapist_harness.transcript_review as review

PROMPT = Path(review.__file__).with_name('prompts') / 'transcript_review_semantic_probe_v1.md'


class SemanticProbeReviewer(review.TranscriptReviewer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prompt = PROMPT.read_text('utf-8')


def main():
    review.TranscriptReviewer = SemanticProbeReviewer
    writer = recovery.base.write_immutable_json
    def write(path, payload):
        if Path(path).name in {'generation_config.json', 'runtime.json', 'generation_summary.json', 'generation_failure.json'}:
            payload = dict(payload, semantic_review_extension=dict(
                prompt_path=str(PROMPT), prompt_sha256=recovery.base.sha256_file(PROMPT),
                entry_path=str(Path(__file__).resolve()), entry_sha256=recovery.base.sha256_file(Path(__file__)),
                rule='Both arms allow deliberate odd targets. Same brief three-text breakdown review; no new score or word audit.'))
        writer(path, payload)
    recovery.base.write_immutable_json = write
    return recovery.main()


if __name__ == '__main__':
    raise SystemExit(main())
