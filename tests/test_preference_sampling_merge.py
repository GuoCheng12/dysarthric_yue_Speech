"""CPU counterfactual shard fixtures; never claim these are actual ASR samples."""

import copy
from pathlib import Path

import pytest

from test_preference_loop_runner import PARENT, registered as registered
from therapist_harness import preference_sampling_merge as merge
from therapist_harness.preference_data import build_preference_pairs, sample_seed
from therapist_harness.store import write_immutable_json, sha256_file


@pytest.fixture
def shards(registered, tmp_path):
    item = registered
    p = item.protocol; p.base_model_dir = str(tmp_path / 'base')
    p.lora = type('LoRA', (), {'seed': 42})()
    p.preference_loop = p.preference_loop.model_copy(update={'review_asr_base_model_sha256': 'b' * 64})
    item.shards = {}
    item.candidates = [{**x, 'seed': sample_seed(42, x['audio_id'], x['sample_index'])} for x in item.candidates]
    for source in ('real', 'synthetic'):
        d = tmp_path / source; d.mkdir(); item.shards[source] = d
        inputs = [x for x in item.inputs if x['source'] == source]
        ids = {x['audio_id'] for x in inputs}
        candidates = [x for x in item.candidates if x['audio_id'] in ids]
        pairs = build_preference_pairs(inputs, candidates, PARENT)[0]
        for name, values in [('validated_inputs.jsonl', inputs), ('candidates.jsonl', candidates), ('preferences.jsonl', pairs)]:
            merge.jsonl(d / name, values)
        config = {**merge.expected_decode(p, PARENT), 'current_adapter': '/fixture/current-parent',
            'source_sha256': 'c'*64, 'preference_code_sha256': 'd'*64}
        write_immutable_json(d / 'sampling_config.json', config)
        write_immutable_json(d / 'sampling_receipt.json', {**config, 'status':'complete', 'parent_files_unchanged':True,
            'candidate_count':len(candidates), 'candidates_sha256':sha256_file(d/'candidates.jsonl'),
            'validated_inputs_sha256':sha256_file(d/'validated_inputs.jsonl')})
    return item


def test_join_is_exactly_the_original_pairs_and_keeps_two_job_receipts(shards):
    item = shards
    inputs, candidates, pairs, audit, provenance = merge.validated_shards(item.shards, item.protocol, PARENT, item.inputs)
    assert inputs == item.inputs
    assert pairs == build_preference_pairs(inputs, candidates, PARENT)[0]
    assert pairs == item.pairs
    assert [p['source'] for p in provenance] == ['real','synthetic']
    assert audit['pair_count'] == 2 and audit['all_correct_audio_count'] == 2
    receipt = {**merge.expected_decode(item.protocol, PARENT), 'status':'complete','candidate_count':len(candidates),
        'schema_version':'current_asr_sampling_merged_v2','source_shards':provenance}
    merge.validate_merged_receipt(receipt, item.protocol, PARENT, inputs, candidates)


@pytest.mark.parametrize('field,value', [('current_asr_sha256','e'*64),('temperature',0.7),('seed',7),
    ('base_model_sha256','e'*64),('samples_per_audio',3)])
def test_even_consistently_rehashed_shards_must_match_frozen_parent_and_decode(shards,field,value):
    item = shards; d = item.shards['synthetic']
    for name in ['sampling_config.json','sampling_receipt.json']:
        p = d/name; data=merge.read(p); data[field]=value; p.unlink(); write_immutable_json(p,data)
    with pytest.raises(ValueError,match='parent/model/decoding'):
        merge.validated_shards(item.shards,item.protocol,PARENT,item.inputs)


def test_duplicate_missing_or_reordered_admissions_are_rejected(shards):
    item=shards
    for expected in [item.inputs[:-1],item.inputs[::-1],item.inputs+[item.inputs[0]]]:
        with pytest.raises(ValueError,match='sample identities/order'):
            merge.validated_shards(item.shards,item.protocol,PARENT,expected)


def test_changed_shard_or_changed_merged_candidates_cannot_keep_receipt(shards):
    item=shards
    inputs,candidates,_,_,provenance=merge.validated_shards(item.shards,item.protocol,PARENT,item.inputs)
    receipt={**merge.expected_decode(item.protocol, PARENT), 'status':'complete','candidate_count':len(candidates),
        'schema_version':'current_asr_sampling_merged_v2','source_shards':provenance}
    changed=copy.deepcopy(candidates);changed[0]['prediction']='invented error'
    with pytest.raises(ValueError,match='merged candidates'):
        merge.validate_merged_receipt(receipt,item.protocol,PARENT,inputs,changed)
    with (item.shards['real']/'candidates.jsonl').open('a') as handle:handle.write('{}\n')
    with pytest.raises(ValueError,match='source shard changed'):
        merge.validate_merged_receipt(receipt,item.protocol,PARENT,inputs,candidates)


def test_merged_metadata_cannot_contradict_its_original_jobs(shards):
    item=shards
    inputs,candidates,_,_,provenance=merge.validated_shards(item.shards,item.protocol,PARENT,item.inputs)
    receipt={**merge.expected_decode(item.protocol, PARENT), 'status':'complete','candidate_count':len(candidates),
        'schema_version':'current_asr_sampling_merged_v2','source_shards':provenance}
    receipt['temperature']=0.1
    with pytest.raises(ValueError,match='merged sampling metadata'):
        merge.validate_merged_receipt(receipt,item.protocol,PARENT,inputs,candidates)
