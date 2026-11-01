import copy
import json

from therapist_harness.material_sampling_mas import request
from therapist_harness.material_selection_offline import DiagnosticBrief
from therapist_harness.schema import AgentConfig


def test_diagnosis_preserves_complete_evidence_ids_and_original_context():
    context = {"evidence": {"observation_ids": ["step232:C727dd8e9df7f", "step232:overall"]}, "history": []}
    original = copy.deepcopy(context)
    config = AgentConfig(model="gpt-5.6-sol", base_url="https://api.laicaishare.xyz", provider="OpenAI",
                         reasoning_effort="xhigh", store=False, max_retries=0, timeout_seconds=600,
                         max_output_tokens=8192, api_key_env="OPENAI_API_KEY")
    result = request(config, "diagnose", context, DiagnosticBrief)
    schema = result["text"]["format"]["schema"]
    allowed = schema["$defs"]["PracticeFocus"]["properties"]["evidence_ids"]["items"]["enum"]
    assert set(allowed) == set(context["evidence"]["observation_ids"])
    assert "step232:C727dd8e9df7" not in allowed
    assert context == original == json.loads(result["input"][-1]["content"])
    assert schema["$defs"]["PracticeFocus"]["properties"]["level"]["enum"] == ["onset", "nucleus", "coda", "tone", "syllable", "text"]
