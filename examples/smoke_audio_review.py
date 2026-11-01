"""Run the real configured audio endpoint before admitting SG preference examples.

Pass credentials only through the named environment variable. This command never
prints audio base64 or credentials, and never substitutes a text-only reviewer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from therapist_harness.audio_review import AudioReviewer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--speech-audio", type=Path, required=True,
                        help="Known speech recording for the audio-access sanity probe.")
    parser.add_argument("--candidate-audio", type=Path, required=True)
    parser.add_argument("--target-text", required=True)
    parser.add_argument("--reference-audio", type=Path, required=True)
    parser.add_argument("--reference-text", default="")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    payload = {"model": args.model, "base_url": args.base_url,
               "smoke_kind": "live_audio_capability_and_candidate_review"}
    agent = None
    try:
        agent = AudioReviewer(model=args.model, base_url=args.base_url,
                              api_key_env=args.api_key_env)
        payload["capability"] = agent.verify_audio_capability(args.speech_audio)
        review = agent.review(args.candidate_audio, args.target_text,
                              args.reference_audio, args.reference_text)
        payload["review"] = review.to_dict()
        payload["status"] = "passed_live_audio_smoke"
        code = 0
    except Exception as exc:
        payload["status"] = "live_audio_smoke_not_passed"
        payload["error_type"] = type(exc).__name__
        payload["http_status"] = getattr(exc, "status_code", None)
        if agent is not None and agent.capability_evidence is not None:
            payload["capability"] = agent.capability_evidence
        # Do not persist upstream request bodies, headers, or credential values.
        payload["action"] = "Check configured audio model, endpoint, credential environment, and audio access."
        code = 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", "utf-8")
    print(json.dumps({"status": payload["status"], "output": str(args.output)}, ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
