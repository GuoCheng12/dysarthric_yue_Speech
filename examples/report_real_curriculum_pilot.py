"""Verify frozen-course execution and summarize the bounded real curriculum pilot."""

import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from therapist_harness.store import sha256_file, sha256_directory, write_immutable_json, write_immutable_bytes  # noqa: E402

ROOT = Path("/data/qwen3-asr/records/therapist_mas_setting_d_v1/real_curriculum_pilot_v1")


def read(path):
    return json.loads(path.read_text())


def main():
    protocol = read(ROOT / "protocol.json")
    for p, digest in {**protocol["source_sha256"], **protocol["data_sha256"]}.items():
        assert sha256_file(Path(p)) == digest, p
    assert sha256_directory(Path(protocol["M0"])) == protocol["M0_sha256"]
    smoke = read(ROOT / "smoke/acceptance.json")
    assert smoke["valid_calls"] == 2
    all_rows = [{"name": "M0", "metrics": protocol["m0_metrics"]}]
    result_files = sorted(ROOT.glob("round_*_results.json"))
    check = ROOT / "doublecheck_results.json"
    if check.exists():
        result_files.append(check)
    for result_file in result_files:
        background_orders = []
        for arm, result in read(result_file).items():
            cfg = result["config"]
            d = Path(cfg["command"][cfg["command"].index("--output_dir")+1]).parent
            index = read(d / "training_index.json")
            assert len(index) == len({r["id"] for r in index}) == 96
            assert {r["source"] for r in index} == {"real"}
            assert sum(r["role"] == "background" for r in index) == 32
            background_orders.append([(r["row"], r["id"]) for r in index if r["role"] == "background"])
            assert sha256_file(d / "train.jsonl") == cfg["train_sha256"]
            assert sha256_directory(Path(cfg["parent"])) == cfg["parent_sha256"]
            if d.name.startswith("r"):
                number = int(d.name[1:3])
                expected_parent = Path(protocol["M0"]) if number == 1 else ROOT / f"r{number-1:02d}_{arm}/training/checkpoint-3"
                assert Path(cfg["parent"]) == expected_parent
            assert len(result["history"]) == 1 and result["history"][0]["step"] == 3
            assert result["history"][0]["epochs"] == 1
            assert not list((d / "training").rglob("optimizer.pt"))
            fit = result["fit_history"]
            all_rows.append({"name": d.name, "metrics": result["history"][-1]["metrics"],
                "fit_before": fit[0]["groups"]["real"]["mean_utterance_loss"],
                "fit_after": fit[-1]["groups"]["real"]["mean_utterance_loss"],
                "statistics": cfg["statistics"], "seed": cfg["seed"]})
        configs = [v["config"] for v in read(result_file).values()]
        assert len({tuple(sorted(c["background_ids"])) for c in configs}) == 1
        assert all(x == background_orders[0] for x in background_orders)
    if check.exists():
        for arm in ("random", "difficulty", "agent"):
            a, b = read(ROOT/f"r01_{arm}/config.json"), read(ROOT/f"check_{arm}/config.json")
            assert set(a["selected_ids"]) == set(b["selected_ids"])
            assert a["parent"] == b["parent"] == str(protocol["M0"])
            assert (a["seed"], b["seed"]) == (42, 43)
    final = read(result_files[-1])
    agent = final["agent"]["history"][-1]["metrics"]
    m0 = protocol["m0_metrics"]
    threshold = min(m0["overall"]["mean_utterance_cer"],
                    *(final[k]["history"][-1]["metrics"]["overall"]["mean_utterance_cer"] for k in ("random", "difficulty")))
    worth = (agent["overall"]["mean_utterance_cer"] <= threshold-.001
        and agent["hard"]["mean_utterance_cer"] <= m0["hard"]["mean_utterance_cer"]
        and all(agent[k]["mean_utterance_cer"] <= m0[k]["mean_utterance_cer"]+.002 for k in ("easy", "medium")))
    course_changes = []
    for number in range(2, 1+len(list(ROOT.glob("round_*_results.json")))):
        request = read(ROOT / f"r{number:02d}_prescribe/request.json")
        packet = json.loads(request["input"][-1]["content"])
        assert packet["round"] == number and len(packet["history"]) == number-1
        assert packet["history"][-1] == read(ROOT / f"round_{number-1:02d}_episode.json")
        previous = read(ROOT / f"r{number-1:02d}_agent/config.json")["selected_ids"]
        current = read(ROOT / f"r{number:02d}_agent/config.json")["selected_ids"]
        course_changes.append({"round": number, "prior_episode_exactly_carried": True,
            "selected_overlap_with_previous": len(set(previous)&set(current)),
            "new_selected_ids": len(set(current)-set(previous))})
    exposure = {}
    for arm in ("random", "difficulty", "agent"):
        ids = [r["id"] for p in sorted(ROOT.glob(f"r??_{arm}/training_index.json")) for r in read(p)]
        exposure[arm] = {"presentations": len(ids), "unique_recordings": len(set(ids)),
                         "repeat_presentations": len(ids)-len(set(ids))}
    summary = {"rows": all_rows, "smoke": smoke, "doublecheck_used": check.exists(),
        "course_changes": course_changes,
        "exposure": exposure,
        "agent_explanation_issue": read(ROOT / "agent_explanation_issue.json"),
        "worth_continuing_under_frozen_rule": worth, "verified": True}
    write_immutable_json(ROOT / "acceptance.json", summary)
    lines = ["# 真实录音 Agent 课程早期实验验收", "",
        f"预先冻结的继续投入条件：{'通过' if worth else '未通过'}。", "",
        "| 分支 | Overall CER | Easy | Medium | Hard | Patient macro | 所选96条 fit loss |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for r in all_rows:
        m = r["metrics"]
        cells = [f"{100*m[k]['mean_utterance_cer']:.4f}%" for k in ("overall", "easy", "medium", "hard")]
        fit = "—" if r["name"] == "M0" else f"{r['fit_before']:.6f} → {r['fit_after']:.6f}"
        lines.append(f"| {r['name']} | " + " | ".join(cells) + f" | {100*m['overall']['patient_macro_cer']:.4f}% | {fit} |")
    lines += ["", "所有实验每轮96条真实录音、0条合成、1遍、3步、LR2e-5；使用既有511条合并语音作探索反馈。",
        f"两次真实 Agent smoke 均通过；64条选择交集 {smoke['intersection']} 条，第一次处方用于训练，未择优。",
        f"一次性复核：{'已使用，同课程、同M0，仅共同训练seed42→43' if check.exists() else '未使用'}。",
        "模型/数据/训练源码哈希、共享背景、唯一录音数、固定末步与无optimizer副本均已核验。", "",
        "## 发现的 Agent 解释错误", "",
        "第二轮把 critical 数量仍为65推断成原来的65条均未解除。一次逐条复核发现：63条保持critical，2条退出，另2条进入。数量相同不等于成员相同。选样ID、训练数量和评测数值没有因此被改写；未围绕该问题新增训练或API试验。", "",
        "## 产物", "", "- protocol.json：执行前冻结的来源、数据与预算。",
        "- smoke/：完整真实请求、回答、用量、时长与处方。",
        "- r*/training_index.json：实际训练顺序与逐条真实录音映射。",
        "- round_*_episode.json：实际干预后的 Therapist Reflection。",
        "- acceptance.json：完整指标、fit loss、覆盖与验收。", ""]
    write_immutable_bytes(ROOT / "report.zh.md", "\n".join(lines).encode())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
