from therapist_harness.text import text_normalize
from therapist_harness.text_scoring_v2 import text_normalize_v2


def test_attested_variants_do_not_count_as_acoustic_errors():
    assert text_normalize('尖銳')!=text_normalize('尖鋭')
    assert text_normalize_v2('尖銳')==text_normalize_v2('尖鋭')
    assert text_normalize_v2('拐杖')==text_normalize_v2('枴杖')==text_normalize_v2('柺杖')


def test_distinct_words_are_not_merged_just_for_sounding_similar():
    assert text_normalize_v2('鑲條裙褂')!=text_normalize_v2('將條裙掛')
    assert text_normalize_v2('發現頭髮')=='發現頭髮'
