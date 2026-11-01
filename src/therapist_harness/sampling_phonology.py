"""Dev transcript contrasts and their real-training pronunciation coverage."""

from collections import Counter, defaultdict
from functools import lru_cache

from .phonology import analyze_phonological_errors, align_characters, map_text_to_syllables
from .text import cer_normalization_projection, tts_phonology_text

LEVELS = ("onset", "nucleus", "coda", "tone")


@lru_cache(maxsize=512)
def pronunciation(text):
    return map_text_to_syllables(tts_phonology_text(text))


@lru_cache(maxsize=8192)
def transcript_contrasts(target, prediction):
    ref, hyp = cer_normalization_projection(target), cer_normalization_projection(prediction)
    analysis = analyze_phonological_errors(ref.surface, hyp.surface)
    scored_pairs = {(p.reference_index, p.hypothesis_index) for p in
                    align_characters(ref.normalized, hyp.normalized).forced_substitutions}
    # Keep only surface substitutions that also survive project CER normalization.
    def scored(i, j):
        return (ref.surface_to_normalized[i], hyp.surface_to_normalized[j]) in scored_pairs
    events = tuple(e for e in analysis.component_events if scored(e.reference_index, e.hypothesis_index))
    homophones = sum(
        scored(p.reference_index, p.hypothesis_index) and
        analysis.reference.positions[p.reference_index].resolved and
        analysis.reference.positions[p.reference_index].jyutping == analysis.hypothesis.positions[p.hypothesis_index].jyutping
        for p in analysis.alignment.forced_substitutions)
    return events, homophones, analysis.coverage


def _measure(rows):
    events, denominators, patients, examples = Counter(), Counter(), defaultdict(set), defaultdict(list)
    coverage = Counter()
    for row in rows:
        mapping = pronunciation(row["clean_gt"])
        coverage.update(reference_characters=mapping.coverage.character_count,
                        resolved_reference_characters=mapping.coverage.resolved_count)
        for position in mapping.positions:
            if position.resolved:
                denominators.update((level, getattr(position, level)) for level in LEVELS)
        differences, homophones, detail = transcript_contrasts(row["clean_gt"], row["prediction"])
        coverage.update(homophone_substitutions=homophones,
                        surface_forced_substitutions=detail.forced_substitution_count,
                        unresolved_surface_substitutions=detail.forced_substitution_g2p_unresolved_count,
                        ambiguous_alignment_recordings=int(detail.optimal_alignment_count > 1),
                        scored_component_events=len(differences))
        for e in differences:
            key = (e.level, e.reference_unit, e.hypothesis_unit)
            events[key] += 1
            patients[key].add(row["speaker_id"])
            example = dict(target_text=row["clean_gt"], transcription=row["prediction"],
                           target_character=e.reference_character, recognized_character=e.hypothesis_character,
                           target_jyutping=e.reference_jyutping, recognized_jyutping=e.hypothesis_jyutping,
                           context_sensitive_reading=e.context_sensitive)
            if len(examples[key]) < 2 and example not in examples[key]:
                examples[key].append(example)
    return dict(events=events, denominators=denominators, patients=patients, examples=examples, coverage=dict(coverage))


def build_phonology_evidence(train_rows, current, previous, groups, prefix):
    """No training predictions: only training text coverage links to dev errors."""
    now = _measure(current)
    before = _measure(previous) if previous is not None else None
    grouped = {uid: gid for gid, group in groups.items() for uid in group["train_ids"]}
    targets = sorted({r["clean_gt"] for r in train_rows})
    profiles = {text: pronunciation(text) for text in targets}
    # Rank on both endpoints so a formerly common error remains visible after it disappears.
    keys = sorted(set(now["events"]) | (set(before["events"]) if before else set()),
                  key=lambda k: (-max(now["events"][k], before["events"][k] if before else 0), k))[:12]
    patterns = []
    for index, key in enumerate(keys, 1):
        level, source, destination = key
        carriers = {text for text, mapping in profiles.items()
                    if any(p.resolved and getattr(p, level) == source for p in mapping.positions)}
        records = [r for r in train_rows if r["clean_gt"] in carriers]
        group_counts = Counter(grouped[r["utt_id"]] for r in records)
        def measurement(measured):
            if measured is None:
                return None
            return dict(events=measured["events"][key],
                        target_component_occurrences=measured["denominators"][(level, source)],
                        patients=len(measured["patients"][key]))
        patterns.append(dict(observation_id=f"{prefix}:phonology:{index}", level=level,
            target_unit=source, recognized_unit=destination, current=measurement(now), previous=measurement(before),
            examples=now["examples"][key], previous_examples=before["examples"][key] if before else [],
            training_practice=dict(recordings=len(records), patients=len({r["speaker_id"] for r in records}),
                target_texts=sorted(carriers), group_recordings={gid: group_counts[gid] for gid in groups},
                fraction_within_group={gid: group_counts[gid] / g["train_count"] for gid, g in groups.items()})))
    return dict(
        method="粤拼的声母、韵腹、韵尾、声调差异；只计入对齐明确且项目CER仍计错的替换。不是音频测得的phone错误率。",
        interpretation="字典读音差异是错误关联线索，不证明实际发音原因。漏字、多字、对齐不明和字典未收录单独记录；同音错字不算声母或声调混淆。",
        coverage_current=now["coverage"], coverage_previous=before["coverage"] if before else None,
        patterns=patterns, total_distinct_current_patterns=len(now["events"]), displayed_pattern_limit=12,
        training_link_explanation="练习表只统计训练文本的读音覆盖，没有训练集识别成绩。目标成分在组内占比越高，增加该组才越能增加相关练习；不能直接指定句子或推断因果。",
        sources=["https://jyutping.org/en/jyutping/", "https://docs.pycantonese.org/stable/jyutping.html"])
