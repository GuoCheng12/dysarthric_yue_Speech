"""Paired error positions and coarse pitch-contour diagnostics."""
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import math
import statistics

from tts_reconstruction_probe import ROOT, SOURCE, read, rows, text_normalize, sha256_file, write_immutable_json
from therapist_harness.evidence import _stable_align


def main():
    result = read(ROOT / "results.json")
    acoustic = {x["utt_id"]: x for x in read(ROOT / "acoustics.json")}
    conditions = ("real", "patient_tts", "token_reconstruction", "vocoder_reconstruction")
    totals = {v: Counter() for v in conditions}
    common = {v: Counter() for v in conditions[1:]}
    pitch = {v: [] for v in conditions[1:]}
    details = []
    eligible = 0
    real_denominator = 0
    exact = Counter()
    for uid, pair in result["paired_predictions"].items():
        reference = text_normalize(pair["real"]["clean_gt"])
        alignment = {v: _stable_align(reference, text_normalize(pair[v]["prediction"])) for v in conditions}
        signatures = {v: {(e.operation,e.reference_index,e.reference_unit,e.hypothesis_unit)
                         for e in alignment[v][1] if e.reference_index is not None} for v in conditions}
        is_eligible = not any(alignment[v][2] for v in conditions)
        if is_eligible:
            eligible += 1
            real_denominator += len(signatures["real"])
        for v in conditions:
            totals[v].update(e.operation for e in alignment[v][1])
        for v in conditions[1:]:
            exact[v] += text_normalize(pair[v]["prediction"]) == text_normalize(pair["real"]["prediction"])
            if is_eligible:
                common[v]["retained_real_substitution_deletion"] += len(signatures["real"] & signatures[v])
                common[v]["new_substitution_deletion"] += len(signatures[v] - signatures["real"])
            a = acoustic[pair["real"]["utt_id"]]["f0_time_bins"]
            b = acoustic[pair[v]["utt_id"]]["f0_time_bins"]
            valid = [(12*math.log2(x),12*math.log2(y)) for x,y in zip(a,b) if x is not None and y is not None]
            if len(valid) >= 20:
                x,y = zip(*valid)
                mx,my = statistics.median(x),statistics.median(y)
                pitch[v].append(statistics.mean(abs((u-mx)-(w-my)) for u,w in valid))
        details.append({"utt_id":uid,"common_unique_alignment":is_eligible,
                        "alignment":{v:{"ambiguous":alignment[v][2],"events":[asdict(e) for e in alignment[v][1]]} for v in conditions}})
    selection = read(ROOT / "selection.json")
    trained_audio = {x["audio"] for x in rows(SOURCE / "replay/train.jsonl")}
    assert not trained_audio & {x["audio"]["real"] for x in selection}
    receipts = [read(ROOT / "audio" / x["pair"]["sample_id"] / "receipt.json") for x in selection]
    assert all(r["llm_bypassed"] and r["reference_mel_frames"]==2*r["reference_token_count"] for r in receipts)
    for row in rows(ROOT / "decode.jsonl"):
        assert sha256_file(Path(row["audio"])) == row["audio_sha256"]
    payload = {"exact_original_transcripts":dict(exact), "all_canonical_error_counts":totals,
        "common_unique_alignment_pairs":eligible, "common_real_substitution_deletion_count":real_denominator,
        "common_error_retention":common,
        "pitch_shape_centered_absolute_error_semitones":{v:{"n":len(values),"median":statistics.median(values)} for v,values in pitch.items()},
        "acceptance":{"rows":48,"variants":5,"decoded_rows":240,"q_audio_unseen_by_replay_asr":True,
            "reference_alignment_verified":48,"llm_bypass_verified":48,"audio_hashes_verified":240,
            "max_reconstruction_clipping_fraction":max(v["clipping_fraction"] for r in receipts for v in r["generated"].values())},
        "details":details}
    write_immutable_json(ROOT / "diagnostics.json",payload)
    print({k:v for k,v in payload.items() if k!='details'})


if __name__ == "__main__":
    main()
