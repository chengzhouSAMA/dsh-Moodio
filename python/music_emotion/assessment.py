# -*- coding: utf-8 -*-
"""把音色、旋律和十八轴组织成可追溯的联合解释，不伪造情感概率。"""
from __future__ import annotations

from typing import Any, Dict


def build_music_analysis(report: Dict[str, Any]) -> Dict[str, Any]:
    four = report.get("emotion_4d") or report.get("emotion") or {}
    instruments = report.get("instruments") or {"status": "not_requested", "candidates": []}
    melody = report.get("melody") or {"status": "not_requested"}
    axes = (report.get("axes_v3") or {}).get("axes", {})
    coverage = (melody.get("agent_summary") or {}).get("overview", {}).get("note_coverage", 0.0)
    constraints = []
    parts = []
    emotion = four.get("emotion")
    if emotion:
        parts.append(f"声学先验倾向“{emotion}”。")
    candidates = instruments.get("candidates", [])[:5]
    if candidates:
        parts.append("音色候选包括" + "、".join(c["name"] for c in candidates) + "，需结合各片段和混淆标记判断其作用。")
    else:
        constraints.append("没有可靠的乐器候选，不能按文件名猜乐器。")
    if any(c.get("status") == "ambiguous" for c in candidates):
        constraints.append("存在相似音色歧义，不能强行确定具体乐器。")
    if any(c.get("status") in ("uncorroborated", "mixed_evidence") for c in candidates):
        constraints.append("乐器模型之间存在分歧，未获互证的候选不能作为确定事实。")
    if any(c.get("verification_scope") in ("family_only", "unsupported", "unavailable") for c in candidates):
        constraints.append("部分乐器只有音色家族或零样本证据，具体类别仍需核实。")

    signals = []
    for key, entry in axes.items():
        if entry.get("value") is None or entry.get("status") == "unreliable":
            continue
        signals.append({"axis": key, "label": entry.get("zh", key), "value": entry["value"],
                        **({"raw": entry["raw"], "unit": entry.get("unit")} if "raw" in entry else {}),
                        "evidence_path": f"axes_v3.axes.{key}"})
    if signals:
        selected = [s for s in signals if s["axis"] in ("Energy", "Brightness", "Warmth", "Grandeur", "BuildDirection")]
        parts.append("十八轴证据：" + "、".join(f"{s['label']} {s['value']:.2f}" for s in selected) + "。")
    else:
        constraints.append("十八轴证据缺失或不可用。")

    if melody.get("status") == "candidate":
        parts.append(f"稳定旋律音符覆盖 {coverage:.1%}，重复动机 {len(melody.get('motifs', []))} 组。")
        if coverage < 0.3:
            constraints.append("旋律覆盖较低，只能解释已标记片段，不足以推断整曲旋律情绪。")
    else:
        constraints.append("没有足够的旋律证据，不推断旋律走向或副歌。")
    constraints.append("乐器种类不直接决定情绪；谱亮度不等同于情绪明亮。")

    return {
        "status": "partial" if instruments.get("status") in ("unavailable", "not_requested", "unknown")
                  or len(axes) != 18 or not signals or melody.get("status") != "candidate" or coverage < 0.3 else "evidence_ready",
        "primary_acoustic_emotion": emotion,
        "summary": " ".join(parts),
        "instruments": {"status": instruments.get("status"), "candidates": [
            {k: c[k] for k in ("id", "name", "status", "mean_similarity", "supported_ratio", "corroborated_ratio", "verification_scope") if k in c}
            for c in candidates], "evidence_path": "instruments.segments"},
        "melody": {"status": melody.get("status"), "note_coverage": coverage,
                   "motif_count": len(melody.get("motifs", [])), "evidence_path": "melody.agent_summary"},
        "axes18": {"available_count": len(axes), "usable_count": len(signals), "signals": signals},
        "context": {"status": "not_searched", "note": "独立 CLI 不联网；Host 工具在本地分析后补充背景与歌词来源。"},
        "constraints": constraints,
        "agent_task": "结合乐器音色、可信旋律片段、十八轴和有来源的背景/歌词解释音乐情感；引用时间戳和来源，说明证据冲突与不足。不得按乐器名称套用固定情绪。",
    }
