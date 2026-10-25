# -*- coding: utf-8 -*-
"""把 moodio_batch.jsonl 汇总成可读报告（Markdown + 机器可读 JSON）。

给 agent 用的报告要点：
    - 只写能用数据支撑的结论，不写"听起来像"的猜测。
    - 明确标注模型的能力边界（相对坐标、无人工标注）。
    - 把失败项单独列出，不让它们混进统计。
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from typing import Any, Dict, List

import numpy as np

# 本文件可能被直接运行（python music_emotion/report.py），此时脚本所在目录会被
# 加到 sys.path 最前面，导致 `import types` 解析到本包的 types.py 而非标准库，
# 触发 "cannot import name 'GenericAlias'" 循环导入。
# 所以：先把自己所在目录从 sys.path 移除，再把包根目录加进去。
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from music_emotion import vamodel as vm  # noqa: E402,F401


def load(path: str) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    ok, bad = [], []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            (bad if rec.get("error") else ok).append(rec)
    return ok, bad


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def summarize(rows: List[Dict[str, Any]], errors: List[Dict[str, Any]]) -> Dict[str, Any]:
    V = np.array([r["valence"] for r in rows], dtype=float)
    A = np.array([r["arousal"] for r in rows], dtype=float)
    C = np.array([r["confidence"] for r in rows], dtype=float)

    # 优先用记录里已有的离散情绪词；老记录没有该字段时按 V/A 现算，
    # 保证新旧 JSONL 都能出报告（不需要重跑音频）。
    from music_emotion.emotion_taxonomy import emotion_profile

    def _emo(r: Dict[str, Any]) -> str:
        if r.get("emotion"):
            return r["emotion"]
        return emotion_profile(r["valence"], r["arousal"])["emotion"]

    emotions = Counter(_emo(r) for r in rows)
    labels = Counter(r["label"] for r in rows)
    arcs = Counter(r.get("arc") or "unknown" for r in rows)
    keys_mode = Counter(
        "major" if (r.get("key") or "").endswith("major") else
        "minor" if (r.get("key") or "").endswith("minor") else "unknown"
        for r in rows
    )

    # 象限象限分布：用 4 个非中性象限 + neutral
    quads = ["elated", "peaceful", "tense", "melancholic"]
    quad_present = sum(1 for q in quads if labels.get(q, 0) > 0)

    # neutral 的死区口径（与 vamodel.label_for 一致：|v|<=0.18 且 |a|<=0.18）。
    # 单独算一遍，用来对照"标签分布"与"死区占比"是否自洽；
    # 两者不一致说明 label 字段与 V/A 不同源，是数据问题，报告里要暴露出来。
    neutral_deadzone = int(((np.abs(V) <= 0.18) & (np.abs(A) <= 0.18)).sum())
    label_neutral = labels.get("neutral", 0)
    consistent = neutral_deadzone == label_neutral

    # 相关：速度 vs 唤醒（这是文献里最稳的方向性结论，可作健全性检查）
    tempo = np.array([r.get("tempo_bpm", np.nan) for r in rows], dtype=float)
    m = np.isfinite(tempo)
    r_tempo_a = float(np.corrcoef(tempo[m], A[m])[0, 1]) if m.sum() > 2 else float("nan")

    return {
        "n_ok": len(rows),
        "n_failed": len(errors),
        "valence": {
            "mean": round(float(V.mean()), 4), "sd": round(float(V.std()), 4),
            "min": round(float(V.min()), 4), "max": round(float(V.max()), 4),
            "p5": round(float(np.percentile(V, 5)), 4),
            "p95": round(float(np.percentile(V, 95)), 4),
        },
        "arousal": {
            "mean": round(float(A.mean()), 4), "sd": round(float(A.std()), 4),
            "min": round(float(A.min()), 4), "max": round(float(A.max()), 4),
            "p5": round(float(np.percentile(A, 5)), 4),
            "p95": round(float(np.percentile(A, 95)), 4),
        },
        "confidence": {
            "mean": round(float(C.mean()), 3),
            "low_share": round(float((C < 0.6).mean()), 3),
        },
        "labels": dict(labels),
        "emotions": dict(emotions),
        "quadrants_present": quad_present,
        "neutral_deadzone_count": neutral_deadzone,
        "neutral_label_count": label_neutral,
        "label_va_consistent": consistent,
        "arcs": dict(arcs),
        "key_modes": dict(keys_mode),
        "sanity_tempo_arousal_r": round(r_tempo_a, 3),
        "errors_by_code": dict(Counter(e.get("error", "?") for e in errors)),
    }


def render_markdown(s: Dict[str, Any], calib: Dict[str, Any]) -> str:
    L: List[str] = []
    L.append("# 曲库情绪分析报告")
    L.append("")
    L.append(f"- 成功分析：**{s['n_ok']}** 首")
    L.append(f"- 失败：**{s['n_failed']}** 首")
    L.append(f"- 标定模式：`{calib['mode']}` — {calib['note']}")
    L.append("")

    L.append("## 分布")
    L.append("")
    L.append("| 维度 | 均值 | 标准差 | p5 | p95 | 范围 |")
    L.append("|---|---|---|---|---|---|")
    for dim in ("valence", "arousal"):
        d = s[dim]
        L.append(
            f"| {dim} | {d['mean']:+.3f} | {d['sd']:.3f} | {d['p5']:+.3f} | "
            f"{d['p95']:+.3f} | [{d['min']:+.2f}, {d['max']:+.2f}] |"
        )
    L.append("")

    L.append("## 情绪分布")
    L.append("")
    total = max(s["n_ok"], 1)
    emos = s.get("emotions") or {}
    for k, c in sorted(emos.items(), key=lambda kv: -kv[1]):
        bar = "█" * int(round(c / total * 40))
        L.append(f"- **{k}**: {c} ({pct(c / total)}) {bar}")
    L.append("")

    L.append("## 四象限分布（研究坐标，供程序判断）")
    L.append("")
    zh = {"elated": "激昂/高涨", "peaceful": "平静/舒展", "tense": "紧张/压迫",
          "melancholic": "低沉/忧郁", "neutral": "中性"}
    for k in ("elated", "peaceful", "tense", "melancholic", "neutral"):
        c = s["labels"].get(k, 0)
        bar = "█" * int(round(c / total * 40))
        L.append(f"- `{k}` ({zh[k]}): **{c}** ({pct(c / total)}) {bar}")
    L.append("")
    L.append(f"四个非中性象限中出现了 **{s['quadrants_present']}/4** 个。")
    L.append("")
    if s.get("label_va_consistent"):
        L.append(f"中性判定自洽：死区(|V|≤0.18 且 |A|≤0.18) 计数 {s['neutral_deadzone_count']} "
                 f"= `neutral` 标签数 {s['neutral_label_count']}。")
    else:
        L.append(f"⚠️ **标签与 V/A 不自洽**：死区计数 {s['neutral_deadzone_count']} "
                 f"≠ `neutral` 标签数 {s['neutral_label_count']}。"
                 "说明记录里的 label 字段不是由这两个数算出来的，需回查生成端。")
    L.append("")

    L.append("## 质量与健全性")
    L.append("")
    L.append(f"- 平均 confidence: **{s['confidence']['mean']:.3f}**")
    L.append(f"- 低置信度(<0.6) 占比: **{pct(s['confidence']['low_share'])}**")
    L.append(f"- tempo↔arousal 相关: **r = {s['sanity_tempo_arousal_r']:+.3f}**"
             "（文献方向为正值，用作方向性健全性检查）")
    L.append(f"- 调式分布: {s['key_modes']}")
    L.append("")

    L.append("## 弧型分布")
    L.append("")
    for k, c in sorted(s["arcs"].items(), key=lambda kv: -kv[1]):
        L.append(f"- `{k}`: {c} ({pct(c / total)})")
    L.append("")

    if s["errors_by_code"]:
        L.append("## 失败项")
        L.append("")
        for k, c in sorted(s["errors_by_code"].items(), key=lambda kv: -kv[1]):
            L.append(f"- `{k}`: {c}")
        L.append("")

    L.append("## 必须知道的限制")
    L.append("")
    L.append("1. **没有人工标注**。这是先验驱动的可解释基线，不是拟合人类评分的判别模型；")
    L.append("   绝对数值不可当作\"人类平均评分\"，只能在同一次运行内部比较相对高低。")
    L.append("2. **标定模式决定含义**。`corpus-relative` 表示 0 点被设为本库中位数，")
    L.append("   换一个曲库后同一首歌的符号可能改变；`fixed` 表示跨库可比的绝对坐标。")
    L.append("3. **歌词不参与**。语义与歌词情绪完全没有进入模型。")
    L.append("4. **confidence 衡量输入质量**（时长、响度、调性清晰度），不是预测准确率。")
    L.append("")
    return "\n".join(L)


def main() -> int:
    src = sys.argv[1] if len(sys.argv) > 1 else "moodio_batch.jsonl"
    out_md = sys.argv[2] if len(sys.argv) > 2 else "moodio_report.md"
    out_json = sys.argv[3] if len(sys.argv) > 3 else "moodio_report.json"
    mode = sys.argv[4] if len(sys.argv) > 4 else "fixed"

    rows, errors = load(src)
    if not rows:
        print(f"no successful records in {src}", file=sys.stderr)
        return 1

    s = summarize(rows, errors)
    if mode == "corpus-relative":
        calib = {"mode": "corpus-relative",
                 "note": "0 点 = 本曲库中位数，跨库不可比"}
    else:
        calib = {"mode": "fixed",
                 "note": "0 点 = 中性输入不动点，跨库可比"}

    s["calibration"] = calib
    s["source"] = os.path.abspath(src)

    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(s, fh, ensure_ascii=False, indent=2)
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write(render_markdown(s, calib))

    print(json.dumps({
        "report_md": os.path.abspath(out_md),
        "report_json": os.path.abspath(out_json),
        "n_ok": s["n_ok"], "n_failed": s["n_failed"],
        "quadrants_present": s["quadrants_present"],
        "tempo_arousal_r": s["sanity_tempo_arousal_r"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
