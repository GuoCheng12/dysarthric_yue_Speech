import copy
import json

from therapist_harness.material_selection_offline import build_selector_request


def test_each_focus_has_its_own_candidate_ids_and_allows_empty_selection():
    leaked_id = "vlink226/1554761758434446517098"
    pack = {"focuses": [
        {"focus_id": "ne1", "candidates": [{"recording_id": leaked_id}], "eligible_ids": ["hidden"]},
        {"focus_id": "l", "candidates": [{"recording_id": "l-only"}], "eligible_ids": ["hidden"]},
        {"focus_id": "empty", "candidates": [], "eligible_ids": []},
    ]}
    original = copy.deepcopy(pack)
    request = build_selector_request(pack)
    variants = request["text"]["format"]["schema"]["properties"]["choices"]["items"]["anyOf"]
    properties = {v["properties"]["focus_id"]["enum"][0]: v["properties"] for v in variants}
    assert set(properties) == {"ne1", "l", "empty"}
    assert properties["ne1"]["recording_ids"]["items"]["enum"] == [leaked_id]
    assert properties["l"]["recording_ids"]["items"]["enum"] == ["l-only"]
    assert properties["empty"]["recording_ids"]["maxItems"] == 0
    for variant in variants:
        assert variant["additionalProperties"] is False
        assert set(variant["required"]) == {"focus_id", "recording_ids", "reason"}
        assert variant["properties"]["recording_ids"].get("minItems", 0) == 0
    assert pack == original
    context = json.loads(request["input"][-1]["content"])["context"]
    assert all("eligible_ids" not in focus for focus in context["focuses"])
