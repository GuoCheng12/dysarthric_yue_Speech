import copy
import pytest

from therapist_harness.material_selection_offline import (
    build_index, prepare_candidates, validate_selection, selector_context, digest,
)


def fixtures():
    rows = [dict(utt_id=str(i), audio=f'/{i}.wav', speaker_id=f'p{i % 3}',
                 clean_gt=text, duration=2, split='train')
            for i, text in enumerate(['北', '北', '北', '木', '木', '媽'])]
    index = build_index(rows, [], {})
    brief = dict(producer='offline_fixture', round_index=1, source_evidence_sha256='a'*64,
                 summary='检查韵尾对应材料', focuses=[dict(focus_id='k', purpose='target',
                 level='coda', target_unit='k', reason='测试', evidence_ids=['example'])])
    return rows, index, brief


def test_search_matches_position_and_diversifies_before_repetition():
    _, index, brief = fixtures()
    pack = prepare_candidates(index, brief, limit=2)
    p = pack['focuses'][0]
    assert p['eligible_count'] == 5
    assert len({r['target_text'] for r in p['candidates']}) == 2
    assert len({r['patient_id'] for r in p['candidates']}) == 2
    assert all(pos['coda'] == 'k' for r in p['candidates'] for pos in r['matching_positions'])
    assert all(r['prior_exposures'] is None for r in p['candidates'])


def test_no_match_returns_empty_without_substitution():
    _, index, brief = fixtures()
    brief['focuses'][0]['target_unit'] = 'not-a-coda'
    pack = prepare_candidates(index, brief)
    selection = dict(context_sha256=digest(selector_context(pack)),
                     choices=[dict(focus_id='k', recording_ids=[], reason='无匹配')])
    assert validate_selection(index, pack, selection)['unique_recordings'] == 0


def test_full_syllable_narrows_broad_coda_search():
    _, index, brief = fixtures()
    brief['focuses'][0].update(level='syllable', target_unit='muk6')
    pack = prepare_candidates(index, brief)
    assert pack['focuses'][0]['eligible_count'] == 2
    assert {r['target_text'] for r in pack['focuses'][0]['candidates']} == {'木'}


def test_reject_heldout_identity_patient_and_audio_overlap():
    rows, _, _ = fixtures()
    for key in ['utt_id', 'audio', 'speaker_id']:
        heldout = dict(utt_id='dev', audio='/dev.wav', speaker_id='dev-patient')
        heldout[key] = rows[0][key]
        with pytest.raises(ValueError, match='held-out overlap'):
            build_index(rows, [heldout], {})


def test_selection_provenance_and_multi_focus_deduplication():
    _, index, brief = fixtures()
    brief['focuses'].append(dict(brief['focuses'][0], focus_id='keep', purpose='maintenance'))
    pack = prepare_candidates(index, brief)
    selection = dict(context_sha256=digest(selector_context(pack)), choices=[
        dict(focus_id=f, recording_ids=['0'], reason='包含目标韵尾') for f in ['k', 'keep']])
    result = validate_selection(index, pack, selection)
    assert result['unique_recordings'] == 1
    assert result['recordings'][0]['focus_ids'] == ['k', 'keep']
    bad = copy.deepcopy(selection)
    bad['choices'][0]['recording_ids'] = ['dev-recording']
    with pytest.raises(ValueError, match='displayed train'):
        validate_selection(index, pack, bad)
    bad = copy.deepcopy(selection)
    bad['context_sha256'] = 'b'*64
    with pytest.raises(ValueError, match='different context'):
        validate_selection(index, pack, bad)
    bad_index = copy.deepcopy(index)
    bad_index['recordings'][0]['audio'] = '/changed.wav'
    with pytest.raises(ValueError, match='index hash'):
        validate_selection(bad_index, pack, selection)
