"""Check completed prescription audio and export a self-contained listening page."""

from __future__ import annotations

import argparse
import base64
from collections import Counter
import hashlib
from html import escape
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    root = args.summary.resolve().parent
    summary = json.loads(args.summary.read_text("utf-8"))
    assert summary["status"] == "complete"
    manifest = Path(summary["manifest"]["path"])
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == summary["manifest"]["sha256"]
    rows = [json.loads(line) for line in manifest.read_text("utf-8").splitlines()]
    outputs = {item["sample_id"]: item for item in summary["outputs"]}
    assert len(outputs) == len(rows) == summary["output_count"]
    sidecar = json.loads((root / "materialization_sidecar.json").read_text("utf-8"))
    metadata = {item["sample_id"]: item for item in sidecar["samples"]}
    audio_data = {}
    cards = []
    for row in rows:
        output = outputs[row["sample_id"]]
        for kind in ("reference", "output"):
            path = Path(row[f"{kind}_wav"])
            content = path.read_bytes()
            expected = row["reference_wav_sha256"] if kind == "reference" else output["output_wav_sha256"]
            assert hashlib.sha256(content).hexdigest() == expected
            waveform, sample_rate = sf.read(path)
            assert len(waveform) > 0 and np.isfinite(waveform).all()
            if kind == "output":
                assert len(waveform) == output["frames"]
                assert sample_rate == output["sample_rate"]
            audio_data[str(path)] = "data:audio/wav;base64," + base64.b64encode(content).decode()
        info = metadata[row["sample_id"]]
        cards.append(
            f'<article><h2>{escape(row["target_text"])}</h2>'
            f'<p>{escape(info["role"])} · reference 分组：{escape(info["reference_difficulty_group"])}'
            f' · seed {row["seed"]} · 合成 {output["duration_seconds"]:.2f} 秒</p>'
            f'<p>Reference：{escape(row["reference_text"])}</p>'
            f'<small>{escape(row["reference_wav"])}</small>'
            f'<audio controls preload="none" data-path="{escape(row["reference_wav"], quote=True)}"></audio>'
            f'<p>Speech Generator epoch 8：</p>'
            f'<audio controls preload="none" data-path="{escape(row["output_wav"], quote=True)}"></audio>'
            f'<small>{escape(row["sample_id"])}</small></article>'
        )
    stats = {
        "audio_count": len(rows),
        "unique_texts": len({row["target_text"] for row in rows}),
        "total_seconds": sum(item["duration_seconds"] for item in outputs.values()),
        "by_reference_difficulty": dict(Counter(metadata[row["sample_id"]]["reference_difficulty_group"] for row in rows)),
        "checks": "Manifest and audio SHA-256, complete frame read, sample rate, finite samples passed.",
        "content_audited_by_listening": False,
        "asr_trained": False,
    }
    page = '<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>MAS 首轮处方 · 真实合成试听</title>'
    page += '<style>body{max-width:1000px;margin:40px auto;padding:0 20px;font:17px/1.6 system-ui;background:#f5f5f5;color:#222}article{background:white;padding:24px;margin:24px 0;border-radius:12px}audio{display:block;width:100%;margin:12px 0}small{overflow-wrap:anywhere;color:#555}h2{font-size:22px}</style>'
    page += f'<h1>MAS 处方 → Speech Generator</h1><p>{stats["unique_texts"]} 句，{stats["audio_count"]} 条，总时长 {stats["total_seconds"] / 60:.2f} 分钟。</p>'
    page += '<p>原文、reference、seed 保持处方 smoke 的分配。全部 WAV 原样内嵌，没有剪辑。技术完整性通过不代表没有漏读、错读或噪声；本次尚未训练 ASR。</p>'
    page += ''.join(cards)
    encoded = json.dumps(audio_data).replace('<', '\\u003c')
    page += f'<script>const audioData={encoded};document.querySelectorAll("audio").forEach(a=>{{a.src=audioData[a.dataset.path]}});</script></html>'
    with (root / "index.html").open("x", encoding="utf-8") as handle:
        handle.write(page)
    with (root / "audio_check.json").open("x", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
