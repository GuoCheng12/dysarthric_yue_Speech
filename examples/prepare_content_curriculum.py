"""Prepare content lessons or run ONE prescription API smoke. Never train ASR."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from openai.lib._pydantic import to_strict_json_schema
from therapist_harness.agent import execute_therapist_request, serialize_response
from therapist_harness.content_curriculum import (ContentPrescription, ContentReflection,
    compile_lesson_block, delivery_report, enrich_catalog, evidence_pack)
from therapist_harness.schema import AgentConfig
from therapist_harness.store import sha256_file, write_immutable_bytes, write_immutable_json as save

RECORDS = Path("/data/qwen3-asr/records/therapist_mas_setting_d_v1")
SOURCE = RECORDS / "real_curriculum_v2_dynamic_reflection_fix_20260907_v1"
ROOT = RECORDS / "content_curriculum_preparation_20260907_v3"
PROMPTS = REPO / "src/therapist_harness/prompts"


def read(path):
    return json.loads(Path(path).read_text())


def build_request(config, payload, schema, kind):
    def rounded(x):
        if isinstance(x, float):
            return round(x, 6)
        if isinstance(x, dict):
            return {k: rounded(v) for k, v in x.items()}
        if isinstance(x, list):
            return [rounded(v) for v in x]
        return x
    content = json.dumps(rounded(payload), ensure_ascii=False, separators=(",", ":"))
    assert "/data/" not in content and "vlink" not in content.lower()
    return {"model": config["model"], "reasoning": {"effort": config["reasoning_effort"]},
        "store": False, "stream": True, "max_output_tokens": config["max_output_tokens"],
        "input": [{"role": "system", "content": (PROMPTS/f"content_curriculum_{kind}.md").read_text()},
                  {"role": "user", "content": content}],
        "text": {"format": {"type": "json_schema", "name": schema.__name__, "strict": True,
                            "schema": to_strict_json_schema(schema)}}}


def draft_prescription():
    """Researcher-authored content draft, NOT an LLM output or trained intervention."""
    descriptions = [
        ("L001", "请求句后半段内容识别", ["C11", "C13"], "C12", "C16",
         "以两条包含幫手的较长请求句为目标，练习后半段确定文字错误；不预设错误来自某种患者病理。",
         "同为幫手请求句，但表达较短、当前错误较少，作为相关语境复习。"),
        ("L002", "邊個问句中的后续内容识别", ["C39", "C42"], "C34", "C45",
         "两条邊個问句包含不同动作与对象，检验跨内容的后半句识别，不把单句记忆当作泛化。",
         "仍是人物问句，但句子较短且对象不同；这不是无错的简单样本，而是相关表达练习。"),
        ("L003", "陈述句不同实体与动作的识别", ["C20", "C26"], "C25", "C05",
         "将两条涉及不同实体、动作和修饰的陈述句组合，练习现有确定文字错误；不是根据音素相似度分组。",
         "换成较短的另一条陈述内容，检验是否可以在内容变化时保持识别。"),
    ]
    lessons = []
    for lid, title, targets, related, protected, goal, relation in descriptions:
        practice = [{"content_id": cid, "role": "target", "region": "whole" if lid == "L003" else "second_half",
            "selection": "error", "buckets": ["easy", "medium", "hard"], "exposures": 25,
            "reason": "只从所标文本区域有确定替换或删除的录音中跨患者均衡选取。"} for cid in targets]
        practice += [
            {"content_id": related, "role": "related", "region": "whole", "selection": "any",
             "buckets": ["easy", "medium", "hard"], "exposures": 15, "reason": relation},
            {"content_id": protected, "role": "protection", "region": "whole", "selection": "correct",
             "buckets": ["easy", "medium"], "exposures": 15,
             "reason": "保留另一条原本低错且当前整句正确的真实表达，作为保持性复习，不以患者整体难度代替录音状态。"}]
        lessons.append({"lesson_id": lid, "title": title, "recognition_goal": goal,
            "evidence_ids": [f"E_{cid}" for cid in targets],
            "hypothesis": "在相关内容之间分配练习，并使用不同患者的真实发音，可能比只集中重复某位患者全部句子更有利于迁移。待训练检验。",
            "practice": practice,
            "expected_transfer": "分别观察已练目标、同内容其他患者、同患者其他内容，以及真正未见内容；不把前三者混称未见内容泛化。",
            "revision_condition": "若只改善已练内容而held-out恶化，下一轮减少单句重复或更换相关内容；先确认目标曝光和患者覆盖确实落实。"})
    return ContentPrescription(diagnosis="研究者根据真实step20内容证据拟定的三堂课，仅用于训练前审核和编译smoke。",
        lessons=lessons, history_used="没有content-centered课程的已完成干预历史。",
        risk="内容组织关系是待验证假设；普通字符错误不能证明声学病理，训练句收益不代表未见句子收益。")


def materialize(root, proposal, destination, origin):
    cfg = read(root/"protocol.json")
    source, step = Path(cfg["source_root"]), cfg["source_step"]
    catalog = read(source/f"dynamic/catalog_{step}.json")
    entries = enrich_catalog(catalog)
    background = read(source/"coverage.json")[str(step+20)]
    previous = [read(source/f"dynamic/manifest_{end}.json") for end in range(20, step+1, 20)]
    manifest = compile_lesson_block(proposal, entries, background, step+20, previous)
    delivery = delivery_report(proposal, entries, background, manifest)
    mapping = read(source/"local_map.json")
    save(destination/"prescription.json", proposal.model_dump(mode="json"))
    save(destination/"manifest.json", manifest)
    save(destination/"delivery.json", delivery)
    preview = [{**mapping[r["id"]], "curriculum_position": r["position"],
                "curriculum_role": r.get("role", "coverage"), "lesson_id": r["unit_id"]} for r in manifest]
    write_immutable_bytes(destination/"training_preview.jsonl",
        "".join(json.dumps(r, ensure_ascii=False)+"\n" for r in preview).encode())
    course = [r for r in manifest if r["stream"] == "course"]
    receipt = {"origin": origin, "training_started": False, "optimizer_updates": 0,
        "planned_updates_if_launched": 20, "coverage_exposures": 400, "course_exposures": len(course),
        "course_unique_recordings": len({r["id"] for r in course}),
        "course_patients": len({mapping[r["id"]]["patient"] for r in course}),
        "course_contents": len({r["content"] for r in course}),
        "course_role_counts": dict(Counter(r["role"] for r in course)),
        "all_unique_recordings": len({r["id"] for r in manifest}),
        "max_recording_exposures": max(Counter(r["id"] for r in manifest).values()),
        "course_audio_minutes": sum(float(mapping[r["id"]]["duration"]) for r in course)/60,
        "total_audio_minutes": sum(float(mapping[r["id"]]["duration"]) for r in manifest)/60,
        "validation": "all content quotas, selector matches, patient-balanced selection and 20:12 batch composition checked"}
    save(destination/"receipt.json", receipt)
    lines = ["# Content-centered 课程：训练前审核", "", f"来源：{origin}。未启动ASR，optimizer更新数为0。", "",
        f"复用已完成实验的step{step}真实诊断；不是新实验已运行到该step。", "",
        f"课程共{len(course)}次曝光，{receipt['course_unique_recordings']}条唯一录音，"
        f"{receipt['course_patients']}位患者，{receipt['course_contents']}个训练content。", "",
        "加400次固定coverage后共640次曝光；下一训练块的清单预览，不是新增数据或训练结果。", ""]
    for lesson in proposal.lessons:
        lines += [f"## {lesson.lesson_id} {lesson.title}", "", lesson.recognition_goal, "", lesson.hypothesis, "",
                  "|角色|训练content|区域 / 筛选|曝光|实际患者|原因|", "|---|---|---|---:|---:|---|"]
        for dose, actual in zip(lesson.practice, delivery[lesson.lesson_id]):
            lines.append(f"|{dose.role}|{dose.content_id}：{catalog['contents'][dose.content_id]}|"
                f"{dose.region} / {dose.selection}|{dose.exposures}|{len(actual['patient_exposures'])}|{dose.reason}|")
        lines += ["", "预期迁移："+lesson.expected_transfer, "", "修订条件："+lesson.revision_condition, ""]
    write_immutable_bytes(destination/"report.zh.md", "\n".join(lines).encode())
    return receipt


def prepare(root, source, step):
    assert step in (20, 40, 60, 80)
    assert not root.exists(), "use a new preparation directory"
    catalog = read(source/f"dynamic/catalog_{step}.json")
    original = read(source/"protocol.json")
    assert (original["train_rows"], original["train_contents"], original["train_patients"]) == (2117, 48, 66)
    for name, digest in original["data_sha256"].items():
        assert sha256_file(Path(name)) == digest, name
    mapping = read(source/"local_map.json")
    assert len(catalog["recordings"]) == len(mapping) == 2117 and len(catalog["contents"]) == 48
    assert {r["id"] for r in catalog["recordings"]} == set(mapping)
    for r in catalog["recordings"]:
        assert (r["patient"], r["content"]) == (mapping[r["id"]]["patient"], mapping[r["id"]]["content"])
        assert catalog["contents"][r["content"]] == mapping[r["id"]]["clean_gt"]
    background = read(source/"coverage.json")[str(step+20)]
    feedback = read(source/f"dynamic/metrics_{step}.json")["feedback"]
    pack, _ = evidence_pack(catalog, background, step, feedback)
    pack["task"] = "Pre-training prescription smoke on an existing measured checkpoint; no content-lesson training has occurred."
    root.mkdir()
    inputs = [source/f"dynamic/catalog_{step}.json", source/f"dynamic/metrics_{step}.json",
        source/"local_map.json", source/"coverage.json", source/"protocol.json",
        *[source/f"dynamic/manifest_{end}.json" for end in range(20, step+1, 20)]]
    code = [Path(__file__), REPO/"src/therapist_harness/content_curriculum.py",
            *[PROMPTS/f"content_curriculum_{kind}.md" for kind in ("prescribe", "reflect")]]
    save(root/"protocol.json", {"setting": "D", "phase": "pretraining_only", "source_root": str(source),
        "source_step": step, "agent": original["agent"], "training_started": False,
        "data_sha256": original["data_sha256"],
        "input_sha256": {str(p): sha256_file(p) for p in inputs},
        "source_sha256": {str(p): sha256_file(p) for p in code},
        "training_note": "No new ASR run initialized. Old snapshot is for interface smoke; regenerate evidence at the authorized future checkpoint."})
    save(root/"content_evidence_pack.json", pack)
    for name, schema in (("prescription", ContentPrescription), ("reflection", ContentReflection)):
        save(root/f"schemas/{name}.json", to_strict_json_schema(schema))
    request = build_request(original["agent"], pack, ContentPrescription, "prescribe")
    save(root/"smoke/prescribe_request.json", request)
    draft = materialize(root, draft_prescription(), root/"draft", "researcher_authored_draft_not_agent_output")
    result = {"status": "offline_prepared_api_smoke_pending", "training_started": False,
              "optimizer_updates": 0, "agent_calls": 0, "request_characters": len(json.dumps(request, ensure_ascii=False)),
              "draft": draft}
    save(root/"prepared.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def smoke(root):
    """One live prescription; no retries, alternative drafts or ASR updates."""
    cfg = read(root/"protocol.json")
    for group in ("input_sha256", "source_sha256", "data_sha256"):
        for p, digest in cfg[group].items():
            assert sha256_file(Path(p)) == digest, p
    assert not (root/"smoke/attempt.json").exists(), "smoke already attempted; inspect its receipt"
    config = AgentConfig.model_validate(cfg["agent"])
    assert config.max_retries == 0
    assert os.environ.get(config.api_key_env), f"load {config.api_key_env} before API smoke"
    request = read(root/"smoke/prescribe_request.json")
    start = time.monotonic()
    save(root/"smoke/attempt.json", {"started_unix": time.time(), "model": config.model,
        "reasoning_effort": config.reasoning_effort, "request_sha256": sha256_file(root/"smoke/prescribe_request.json")})
    response = execute_therapist_request(SimpleNamespace(agent=config), request)
    save(root/"smoke/response.json", serialize_response(response))
    save(root/"smoke/timing.json", {"seconds": time.monotonic()-start,
        "usage": response.usage.model_dump(mode="json") if response.usage else None})
    assert response.status == "completed", response.status
    proposal = ContentPrescription.model_validate_json(response.output_text)
    result = materialize(root, proposal, root/"agent_smoke", f"{config.model}_{config.reasoning_effort}_live_prescription")
    save(root/"smoke/validated.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "smoke"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--step", type=int, default=20)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.root, args.source, args.step)
    else:
        smoke(args.root)
