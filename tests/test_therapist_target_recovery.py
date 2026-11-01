from itertools import product

from therapist_harness.evidence import _key_descriptor, compare_prediction_snapshots
from therapist_harness.phonology import align_characters


def row(pred, *, gt="我想去見客", index=0):
    return dict(
        utt_id=f"u{index}",
        speaker_id=f"p{index}",
        prompt_id=f"c{index}",
        zero_shot_bucket="hard",
        clean_gt=gt,
        pred=pred,
    )


def compare(before, after, *, monitored=True):
    return compare_prediction_snapshots(
        before,
        after,
        before_prediction_column="pred",
        after_prediction_column="pred",
        critical_cer_threshold=0.5,
        off_target_max_per_level_per_direction=1,
        monitored_clusters=[_key_descriptor(("deletion", "coda", "k", "∅", None, None))]
        if monitored
        else [],
    )


def test_substitution_becoming_whole_character_deletion_is_not_recovery():
    result = compare([row("我想去見卡")], [row("我想去見")])
    change = result.monitored_cluster_changes[0]
    assert (
        result.before_metrics.overall.pooled_cer
        == result.after_metrics.overall.pooled_cer
        == 0.2
    )
    assert (change.before.error_count, change.after.error_count) == (1, 0)
    assert change.response_semantics == "exact_signature_frequency_only"
    positions = change.target_positions
    assert positions.response == "unchanged"
    assert positions.recovered_count == 0
    assert positions.signature_to_deletion_count == 1
    assert positions.before_error_count == positions.after_error_count == 1
    assert positions.by_difficulty["hard"].signature_to_deletion_count == 1


def test_signature_disappearance_into_other_wrong_character_is_not_recovery():
    change = compare(
        [row("我想去見卡")], [row("我想去見格")]
    ).monitored_cluster_changes[0]
    assert change.after.error_count == 0
    assert change.target_positions.signature_to_other_error_count == 1
    assert change.target_positions.response == "unchanged"


def test_substitution_becoming_correct_character_plus_insertion_is_qualified():
    for prediction in ("我想去見客卡", "我想去見卡客"):
        result = compare([row("我想去見卡")], [row(prediction)])
        assert (
            result.before_metrics.overall.pooled_cer
            == result.after_metrics.overall.pooled_cer
            == 0.2
        )
        positions = result.monitored_cluster_changes[0].target_positions
        assert positions.recovered_count == 1  # Partial position recovery is real.
        assert positions.before_cohort_character_insertions == 0
        assert positions.after_cohort_character_insertions == 1
        assert positions.response == "inconclusive"  # Not unqualified ASR gain.


def test_correct_character_is_recovery_but_new_errors_elsewhere_are_counted():
    old = [row("我想去見卡")]
    new = [row("我想去見客")]
    positions = compare(old, new).monitored_cluster_changes[0].target_positions
    assert positions.response == "improved"
    assert positions.signature_recovered_count == 1
    positions = (
        compare(
            [*old, row("我想去見客", index=1)],
            [*new, row("我想去見卡", index=1)],
        )
        .monitored_cluster_changes[0]
        .target_positions
    )
    assert positions.recovered_count == positions.regressed_count == 1
    assert positions.response == "unchanged"


def test_ambiguous_alignment_is_not_treated_as_correction():
    positions = (
        compare([row("卡客", gt="客客")], [row("客", gt="客客")])
        .monitored_cluster_changes[0]
        .target_positions
    )
    assert positions.after_unresolved_count == 2
    assert positions.signature_unresolved_count == 1
    assert positions.recovered_count == 0
    assert positions.response == "inconclusive"


def test_mathematically_tied_patient_macro_cer_does_not_crash():
    gt = "我今日同媽媽去公園玩"
    before = [row(gt[:-edits], gt=gt, index=i) for i, edits in enumerate((1, 2, 3))]
    after = [row(gt[:-edits], gt=gt, index=i) for i, edits in enumerate((3, 2, 1))]
    change = compare(before, after, monitored=False).metric_changes[-1]
    assert change.before_value == change.after_value == 0.2
    assert change.delta == 0
    assert change.response == "unchanged"


def test_all_optimal_gt_outcomes_against_exhaustive_small_alignments():
    # An independent exhaustive oracle; no use of production DP tie-breaking.
    def enumerate_paths(ref, hyp, i=0, j=0):
        if i == len(ref) and j == len(hyp):
            return [(0, ())]
        paths = []
        if i < len(ref) and j < len(hyp):
            paths += [
                (cost + (ref[i] != hyp[j]), (j, *outcomes))
                for cost, outcomes in enumerate_paths(ref, hyp, i + 1, j + 1)
            ]
        if i < len(ref):
            paths += [
                (cost + 1, (None, *outcomes))
                for cost, outcomes in enumerate_paths(ref, hyp, i + 1, j)
            ]
        if j < len(hyp):
            paths += [
                (cost + 1, outcomes)
                for cost, outcomes in enumerate_paths(ref, hyp, i, j + 1)
            ]
        return paths

    strings = [
        "".join(chars) for length in range(4) for chars in product("ab", repeat=length)
    ]
    for ref, hyp in product(strings, repeat=2):
        paths = enumerate_paths(ref, hyp)
        minimum = min(cost for cost, _ in paths)
        best = [outcomes for cost, outcomes in paths if cost == minimum]
        aligned = align_characters(ref, hyp)
        assert aligned.optimal_path_count == len(best)
        assert [set(values) for values in aligned.reference_outcomes] == [
            {path[index] for path in best} for index in range(len(ref))
        ]
