"""Fixed real-data groups and dev-only observations for the Setting E pilot."""

from collections import Counter, defaultdict
import hashlib
import json
from statistics import median_low

from .phonology import align_characters
from .text import text_normalize
from .sampling_phonology import build_phonology_evidence, pronunciation


def content_id(text):
    return "C" + hashlib.sha256(text_normalize(text).encode()).hexdigest()[:12]


def make_sampling_groups(train_rows):
    """One fixed partition: historical recognition difficulty x target length."""
    threshold = median_low(sorted({len(text_normalize(r["clean_gt"])) for r in train_rows}))
    labels = {"easy": "原始模型识别较容易", "medium": "原始模型识别中等困难",
              "hard": "原始模型识别较困难"}
    groups = {}
    for difficulty in labels:
        for length in ("short", "long"):
            members = [r for r in train_rows if r["zero_shot_bucket"] == difficulty and
                       (len(text_normalize(r["clean_gt"])) <= threshold) == (length == "short")]
            if not members:
                continue
            gid = f"{difficulty}_{length}"
            groups[gid] = {
                "group_id": gid,
                "label": labels[difficulty] + ("、较短句子" if length == "short" else "、较长句子"),
                "historical_difficulty": difficulty, "length_class": length,
                "short_max_characters": threshold,
                "train_count": len(members),
                "train_ids": [r["utt_id"] for r in members],
                "patient_count": len({r["speaker_id"] for r in members}),
                "content_count": len({content_id(r["clean_gt"]) for r in members}),
                "target_examples": sorted({r["clean_gt"] for r in members})[:5],
            }
    ids = [uid for group in groups.values() for uid in group["train_ids"]]
    if len(ids) != len(set(ids)) or set(ids) != {r["utt_id"] for r in train_rows}:
        raise ValueError("fixed groups must partition the complete training pool")
    return groups


def _join_predictions(train_rows, predictions):
    canonical = {r["utt_id"]: r for r in train_rows}
    if len(predictions) != len(canonical) or {p["utt_id"] for p in predictions} != set(canonical):
        raise ValueError("MAS diagnostics must contain exactly the dev pool, never train or test rows")
    result = []
    for pred in predictions:
        row = canonical[pred["utt_id"]]
        if pred.get("speaker_id", row["speaker_id"]) != row["speaker_id"]:
            raise ValueError("prediction patient identity differs from its recording")
        reference, hypothesis = text_normalize(row["clean_gt"]), text_normalize(pred["prediction"])
        alignment = align_characters(reference, hypothesis)
        if not reference:
            raise ValueError("empty training target")
        errors = alignment.edit_distance
        if ("errors" in pred and pred["errors"] != errors) or ("chars" in pred and pred["chars"] != len(reference)):
            raise ValueError("stored prediction metrics differ from the canonical CER calculation")
        result.append(dict(row, prediction=pred["prediction"], errors=errors, chars=len(reference),
            substitutions=alignment.substitution_count, deletions=alignment.deletion_count,
            insertions=alignment.insertion_count))
    return result


def summarize_sampling_rows(rows):
    if not rows:
        return {"audio": 0, "CER_percent": None, "patient_equal_CER_percent": None}
    errors, chars = sum(r["errors"] for r in rows), sum(r["chars"] for r in rows)
    patients = defaultdict(lambda: [0, 0])
    for row in rows:
        patients[row["speaker_id"]][0] += row["errors"]
        patients[row["speaker_id"]][1] += row["chars"]
    return {"audio": len(rows), "patients": len(patients), "errors": errors, "chars": chars,
            "CER_percent": 100 * errors / chars,
            "patient_equal_CER_percent": sum(100 * e / n for e, n in patients.values()) / len(patients),
            "substitutions": sum(r.get("substitutions", 0) for r in rows),
            "deletions": sum(r.get("deletions", 0) for r in rows),
            "insertions": sum(r.get("insertions", 0) for r in rows)}


def validate_train_feedback_split(train_rows, heldout_rows, setting):
    """Both protocols hold out recordings; D holds out text, E patients."""
    if setting not in ('D', 'E'):
        raise ValueError('unsupported curriculum split')
    for key in ('utt_id', 'audio'):
        if {r[key] for r in train_rows} & {r[key] for r in heldout_rows}:
            raise ValueError(f'held-out recording overlap: {key}')
    if setting == 'E':
        if {r['speaker_id'] for r in train_rows} & {r['speaker_id'] for r in heldout_rows}:
            raise ValueError('Setting E dev must be recording- and patient-disjoint from train')
    else:
        if {text_normalize(r['clean_gt']) for r in train_rows} & {text_normalize(r['clean_gt']) for r in heldout_rows}:
            raise ValueError('Setting D must hold out complete normalized texts')
        if not {r['speaker_id'] for r in heldout_rows} <= {r['speaker_id'] for r in train_rows}:
            raise ValueError('Setting D held-out speakers must be represented in train')


def build_sampling_evidence(train_rows, feedback_predictions, groups, *, feedback_rows,
                            previous_feedback=None, round_index, checkpoint_step, exposures=None, setting="E"):
    """Dev supplies all measured outcomes; train supplies available practice only."""
    validate_train_feedback_split(train_rows, feedback_rows, setting)
    current = _join_predictions(feedback_rows, feedback_predictions)
    previous = _join_predictions(feedback_rows, previous_feedback) if previous_feedback is not None else None
    before = {r["utt_id"]: r for r in previous or []}
    train_by_id = {r["utt_id"]: r for r in train_rows}
    group_by_id = {uid: gid for gid, group in groups.items() for uid in group["train_ids"]}
    patients = {p: f"DEV{i:03d}" for i, p in enumerate(sorted({r["speaker_id"] for r in feedback_rows}), 1)}
    prefix = f"step{checkpoint_step}"
    observations = [f"{prefix}:overall"]
    result = {
        "round_index": round_index, "checkpoint_step": checkpoint_step,
        "feedback_role": "dev", "setting": setting,
        "data_explanation": "所有识别成绩、转写及前后变化来自独立dev患者，不用于梯度训练。训练池只提供可练习文本、覆盖统计和实际曝光。test不提供。",
        "group_explanation": "训练组固定为原始识别难度与句长的交叉分组，组内等概率抽录音。组的current/previous只统计按同一规则归类的dev录音，不是训练组自身的成绩。难度不是病情。",
        "action_explanation": "唯一操作是下一阶段各训练组的采样概率；有放回抽样。语音成分覆盖用于解释组与问题的关联，不改变组成员。",
        "feedback": {"observation_id": observations[0], "current": summarize_sampling_rows(current),
                     "previous": summarize_sampling_rows(previous) if previous is not None else None},
        "groups": [], "content_observations": [], "patient_observations": [],
        "observation_ids": observations,
    }
    if setting == 'D':
        result['data_explanation'] = (
            '这是UCG：训练和开发集患者重叠，但完整句子不重叠。所有识别成绩、转写及前后变化只来自dev录音；'
            'dev录音和目标句子均不做梯度训练，test不提供。训练池只提供已有录音的文本、读音覆盖及曝光。'
            '开发集完整句子在训练池中没有同句录音。请依据dev错误选择训练池已有覆盖的声母、韵腹、韵尾、声调或完整音节，'
            '不要把dev整句当作可用训练材料，也不创造或搬入dev文本。')
    def previous_summary(members):
        return summarize_sampling_rows([before[r["utt_id"]] for r in members]) if before else None
    def example(row):
        return {"patient": patients[row["speaker_id"]], "target_text": row["clean_gt"],
                "transcription": row["prediction"], "errors": row["errors"], "characters": row["chars"],
                "previous_transcription": before[row["utt_id"]]["prediction"] if before else None}
    for gid, group in groups.items():
        members = [r for r in current if r["zero_shot_bucket"] == group["historical_difficulty"] and
                   (len(text_normalize(r["clean_gt"])) <= group["short_max_characters"]) == (group["length_class"] == "short")]
        oid = f"{prefix}:group:{gid}"
        observations.append(oid)
        item = {k: v for k, v in group.items() if k != "train_ids"}
        item.update(observation_id=oid, measurement_role="dev", current=summarize_sampling_rows(members),
                    previous=previous_summary(members),
                    actual_exposures=sum((exposures or {}).get(uid, 0) for uid in group["train_ids"]),
                    examples=[example(r) for r in sorted(members, key=lambda r: (-r["errors"] / r["chars"], r["utt_id"]))[:2]])
        result["groups"].append(item)
    for target in sorted({r["clean_gt"] for r in current}):
        members = [r for r in current if r["clean_gt"] == target]
        practice = [r for r in train_rows if r["clean_gt"] == target]
        oid = f"{prefix}:{content_id(target)}"
        observations.append(oid)
        mapping = pronunciation(target)
        result["content_observations"].append(dict(observation_id=oid, target_text=target,
            jyutping=[p.jyutping for p in mapping.positions],
            current=summarize_sampling_rows(members), previous=previous_summary(members),
            examples=[example(r) for r in sorted(members, key=lambda r: -r["errors"] / r["chars"])[:1]],
            training_practice=dict(recordings=len(practice), patients=len({r["speaker_id"] for r in practice}),
                group_counts=dict(Counter(group_by_id[r["utt_id"]] for r in practice)),
                actual_exposures=sum((exposures or {}).get(r["utt_id"], 0) for r in practice))))
    for patient, public_id in patients.items():
        members = [r for r in current if r["speaker_id"] == patient]
        oid = f"{prefix}:{public_id}"
        observations.append(oid)
        result["patient_observations"].append(dict(observation_id=oid, patient=public_id,
            current=summarize_sampling_rows(members), previous=previous_summary(members)))
    if before:
        result["feedback"]["recording_transitions"] = dict(Counter(
            "improved" if row["errors"] < before[row["utt_id"]]["errors"] else
            "worsened" if row["errors"] > before[row["utt_id"]]["errors"] else "unchanged"
            for row in current))
    if not set(exposures or {}) <= set(train_by_id):
        raise ValueError("actual training exposures contain a non-training recording")
    result["phonology"] = build_phonology_evidence(train_rows, current, previous, groups, prefix)
    observations.extend(p["observation_id"] for p in result["phonology"]["patterns"])
    serialized = json.dumps(result, ensure_ascii=False)
    if "/data/" in serialized or "vlink" in serialized:
        raise ValueError("local paths or patient identities entered the Agent evidence")
    return result


def build_sampling_outcome(train_rows, before_predictions, after_predictions, groups, *, feedback_rows,
                           decision, round_index, checkpoint_step, exposures=None,
                           before_checkpoint_step=0, setting="E"):
    evidence = build_sampling_evidence(train_rows, after_predictions, groups, feedback_rows=feedback_rows,
        previous_feedback=before_predictions, round_index=round_index,
        checkpoint_step=checkpoint_step, exposures=exposures, setting=setting)
    return {"round_index": round_index, "checkpoint_step": checkpoint_step,
            "feedback_role": "dev", "decision": decision,
            "evidence": evidence, "before_checkpoint_step": before_checkpoint_step,
            "delivery": {"recording_exposures": sum((exposures or {}).values()),
                         "unique_recordings": len([v for v in (exposures or {}).values() if v > 0])},
            "observation_ids": evidence["observation_ids"],
            "interpretation": "只根据dev的识别变化检验分配策略；曝光计数仅描述实际练了什么。test不进入反思。"}
