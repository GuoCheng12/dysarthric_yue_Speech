"""One dev-only Agent probe: group exposure budgets followed by material choice."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from openai.lib._pydantic import to_strict_json_schema
from pydantic import Field
from therapist_harness.real_sampling_mas import SamplingModel, _call, payload_hash
from therapist_harness.schema import AgentConfig
from therapist_harness.laicai_credentials import load_laicai_key
from therapist_harness.store import write_immutable_json, sha256_file


class GroupAllocation(SamplingModel):
    group_id: str
    probability: float = Field(ge=0, le=1)
    reason: str = Field(min_length=1, max_length=700)


class GroupMaterialDecision(SamplingModel):
    round_index: int = Field(ge=1)
    context_sha256: str
    allocations: list[GroupAllocation]
    selected_probability_within_group: float = Field(ge=0, le=1)
    reason: str = Field(min_length=1, max_length=1200)
    expected_change: str = Field(min_length=1, max_length=1200)


INSTRUCTION = '''你是调度Agent，根据独立dev的识别结果，为已有真实训练录音安排下一阶段练习。
这里只做方案probe，尚未连接训练。不要修改录音、文本、损失、学习率或步数，不知道test结果。
难度由原始ASR识别表现分为easy、medium、hard，不代表患者病情；long/short表示长短句。
现在的练习安排分两层：先决定六组各占多少总曝光，再在该组内决定是否使用已经选中的录音。
allocations必须列出全部六组，probability之和为1。你自行判断各组练习价值，不设固定困难比例。
selected_probability_within_group是所有组共用的一个0到1概率。某组有选中录音时，
以此概率在组内选中录音中等概率抽取，否则在该组全部训练录音中等概率抽取。
组内没有选中录音时，直接在该组全部录音中等概率抽取，不向其他组转移曝光。
两种组内抽取都有放回；同一录音服务多个目标时只计一次，不获得额外权重。
所以组总曝光由你分配的六组概率控制；所选材料用于改变组内具体练习对象。
估算每条选中录音重复量：stage_exposures × 组概率 × 所选概率 ÷ 该组选中录音数。
结合dev前后变化、当前实际曝光及材料覆盖判断，不认为当前错误最多就应无限重复，
也不只保护容易内容而使仍需要学习的部分缺少练习。允许所选概率为0。
用普通中文简述决定与预期可检查的dev变化；不要将文字读音关联说成已证明的实际发音问题。
round_index及context_sha256原样返回。只输出指定JSON。'''


def main(args):
    read = lambda path: json.loads(path.read_text())
    stage = args.stage
    protocol_path = stage.parents[1] / 'protocol.json'
    protocol = read(protocol_path)
    evidence = read(stage / 'evidence.json')
    assert evidence['feedback_role'] == 'dev'
    # Replace the old action description so the probe has one unambiguous contract.
    evidence['action_explanation'] = '先分配六组总曝光，再在组内混合所选与全部录音。仅用于本次独立probe。'
    groups_path = Path(protocol['groups_path'])
    groups = read(groups_path)
    membership = {uid: key for key, group in groups.items() for uid in group['train_ids']}
    selected_path = stage / 'decision/selected_materials.json'
    selected = read(selected_path)
    plan_path = stage / 'sampling_plan.json'
    budget = len(read(plan_path)['utterance_ids'])
    counts = Counter(membership[row['recording_id']] for row in selected['recordings'])
    context = dict(round_index=evidence['round_index'], evidence=evidence,
        diagnosis=read(stage / 'decision/diagnosis/decision.json'),
        stage_exposures=budget, material_counts_by_group={k: counts[k] for k in groups},
        selected_materials=[{k: v for k, v in row.items() if k != 'audio'} for row in selected['recordings']],
        selection_reasons=selected['choices'])
    context_hash = payload_hash(context)
    config = AgentConfig.model_validate(protocol['agent'])
    schema = to_strict_json_schema(GroupMaterialDecision)
    schema['$defs']['GroupAllocation']['properties']['group_id']['enum'] = sorted(groups)
    schema['properties']['allocations'].update(minItems=6, maxItems=6)
    request = dict(model=config.model, reasoning={'effort': config.reasoning_effort}, store=False,
        stream=True, max_output_tokens=config.max_output_tokens,
        input=[dict(role='system', content=INSTRUCTION), dict(role='user',
            content=json.dumps(dict(context=context, context_sha256=context_hash), ensure_ascii=False, sort_keys=True))],
        text={'format': dict(type='json_schema', name='group_material_schedule_probe', strict=True, schema=schema)})

    def validate(result):
        assert result.round_index == context['round_index'] and result.context_sha256 == context_hash
        assert len(result.allocations) == 6 and {row.group_id for row in result.allocations} == set(groups)
        assert math.isclose(sum(row.probability for row in result.allocations), 1, abs_tol=1e-6)

    args.out.mkdir(parents=True, exist_ok=False)
    files = [protocol_path, groups_path, selected_path, plan_path, stage / 'evidence.json',
             stage / 'decision/diagnosis/decision.json', Path(__file__)]
    write_immutable_json(args.out / 'protocol.json', dict(training_connected=False, feedback_role='dev',
        method='Agent group exposure allocation followed by a shared within-group material probability',
        source_stage=str(stage), sources={str(p): sha256_file(p) for p in files}))
    load_laicai_key()
    result, sources, seconds = _call(config, phase='prescribe', context={}, directory=args.out / 'schedule',
        validate=validate, request=request, output_type=GroupMaterialDecision, role='Scheduling Agent')
    allocation = {row.group_id: row.probability for row in result.allocations}
    selected_p = result.selected_probability_within_group
    expected = {key: dict(total=budget * value,
        selected=budget * value * selected_p if counts[key] else 0,
        selected_per_recording=budget * value * selected_p / counts[key] if counts[key] else None)
        for key, value in allocation.items()}
    record = dict(passed=True, actual_model=config.model, training_connected=False, seconds=seconds,
        decision=result.model_dump(), material_counts_by_group=dict(counts), expected_exposures=expected,
        actual_gpu_training_performed=False, source_sha256=sources,
        interpretation='Feasible proposed allocation only; no performance result and no causal claim.')
    write_immutable_json(args.out / 'result.json', record)
    print(json.dumps({k:v for k,v in record.items() if k != 'source_sha256'}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    main(parser.parse_args())
