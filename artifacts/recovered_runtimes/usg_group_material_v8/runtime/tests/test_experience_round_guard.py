import json

from therapist_harness.material_sampling_mas import request
from therapist_harness.real_sampling_mas import SamplingExperience
from therapist_harness.schema import AgentConfig


def test_evicted_round_in_nested_memory_is_not_a_valid_current_citation():
    config = AgentConfig(model='gpt-5.6-sol', base_url='https://api.laicaishare.xyz',
        provider='OpenAI', reasoning_effort='xhigh', store=False, max_retries=0,
        timeout_seconds=600, max_output_tokens=8192, api_key_env='OPENAI_API_KEY')
    context = {'completed_rounds': [dict(round_index=n,
        experience={'supporting_rounds': [1, n], 'experience': '此前第1轮有改善'})
        for n in (2, 3, 4, 5)]}
    result = request(config, 'experience', context, SamplingExperience)
    supplied = json.loads(result['input'][-1]['content'])
    assert supplied['available_supporting_rounds'] == [2, 3, 4, 5]
    assert supplied['required_current_round'] == 5
    assert supplied['completed_rounds'] == context['completed_rounds']
    assert result['text']['format']['schema']['properties']['supporting_rounds']['items']['enum'] == [2, 3, 4, 5]
    assert 'available_supporting_rounds' not in context
