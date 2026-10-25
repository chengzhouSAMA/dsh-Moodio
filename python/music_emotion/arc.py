# -*- coding: utf-8 -*-
"""情感弧线与结构分段。

agent 场景下，"这首歌整体悲伤"这种结论几乎没用；
"第 1:00-1:30 是能量低谷、第 2:10 有一次情绪峰值"才有用。
所以这里显式构造时间序列、弧型、转折点。
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

__all__ = ["build_arc", "detect_turning_points", "classify_arc", "smooth_series"]


def smooth_series(x: np.ndarray, window: int = 3) -> np.ndarray:
    """中值+均值混合平滑，抑制单窗抖动而保留拐点。"""
    if x.size < window or window < 3:
        return x
    pad = window // 2
    padded = np.pad(x, pad, mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")[: x.size]


def detect_turning_points(
    series: np.ndarray,
    times: List[float],
    min_prominence: float = 0.15,
) -> Dict[str, List[Dict[str, float]]]:
    """找峰/谷。无 scipy.signal 依赖的简化实现：局部极值 + 显著度过滤。"""
    peaks: List[Dict[str, float]] = []
    troughs: List[Dict[str, float]] = []

    if series.size < 3:
        return {"peaks": peaks, "troughs": troughs}

    mean = float(np.mean(series))
    for i in range(1, series.size - 1):
        prev, cur, nxt = series[i - 1], series[i], series[i + 1]
        if cur >= prev and cur >= nxt and (cur - mean) >= min_prominence:
            peaks.append({"time_sec": round(times[i], 2), "value": round(float(cur), 3)})
        elif cur <= prev and cur <= nxt and (mean - cur) >= min_prominence:
            troughs.append({"time_sec": round(times[i], 2), "value": round(float(cur), 3)})

    return {"peaks": peaks, "troughs": troughs}


def classify_arc(arousal: np.ndarray, valence: np.ndarray) -> str:
    """把能量/效价序列归类成可命名的弧型。

    采用音乐情绪研究中常用的粗分类，而不是强行套用某一种理论模型：
      build_up   逐步上升（常见于副歌推进）
      wind_down  逐步下降（常见于尾奏收束）
      wave       起伏多次（常见于完整编曲）
      flat       基本平稳（常见于循环型/氛围型）
      drop       先升后降（峰值在中段）
      rise_fall  双峰/复杂
    """
    if arousal.size < 3:
        return "flat"

    a = smooth_series(arousal.astype(np.float64), 3)
    slope = float(np.polyfit(np.arange(a.size), a, 1)[0])
    rng = float(np.max(a) - np.min(a))
    sign_changes = int(np.sum(np.diff(np.sign(np.diff(a))) != 0))

    v_slope = float(np.polyfit(np.arange(valence.size), valence, 1)[0]) if valence.size >= 3 else 0.0

    if rng < 0.12:
        return "flat"
    if sign_changes >= 3:
        return "wave"
    if slope > 0.02 and v_slope >= -0.005:
        return "build_up"
    if slope < -0.02:
        return "wind_down"
    if rng >= 0.30 and sign_changes <= 2:
        return "drop"
    return "wave"


def build_arc(segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """从分段结果构造弧线时间序列。

    Args:
        segments: 每个元素至少含 start_sec / end_sec / arousal / valence。

    Returns:
        {
          "arc": 弧型名,
          "series": [{"t": 秒, "arousal": .., "valence": ..}, ...],
          "turning_points": {"arousal": {...}, "valence": {...}},
          "delta": {"arousal_range": .., "valence_range": .., "arousal_slope": ..},
          "peak": {...}, "trough": {...}
        }
    """
    if not segments:
        return {
            "arc": "flat",
            "series": [],
            "turning_points": {"arousal": {"peaks": [], "troughs": []},
                               "valence": {"peaks": [], "troughs": []}},
            "delta": {"arousal_range": 0.0, "valence_range": 0.0, "arousal_slope": 0.0},
            "peak": None,
            "trough": None,
        }

    times = [float((s["start_sec"] + s["end_sec"]) / 2.0) for s in segments]
    arousal = np.array([float(s["arousal"]) for s in segments], dtype=np.float64)
    valence = np.array([float(s["valence"]) for s in segments], dtype=np.float64)

    series = [
        {"t": round(times[i], 2),
         "arousal": round(float(arousal[i]), 3),
         "valence": round(float(valence[i]), 3)}
        for i in range(len(times))
    ]

    a_peaks = detect_turning_points(smooth_series(arousal, 3), times)
    v_peaks = detect_turning_points(smooth_series(valence, 3), times)

    slope = float(np.polyfit(np.arange(arousal.size), arousal, 1)[0]) if arousal.size >= 2 else 0.0
    peak_i = int(np.argmax(arousal))
    trough_i = int(np.argmin(arousal))

    return {
        "arc": classify_arc(arousal, valence),
        "series": series,
        "turning_points": {"arousal": a_peaks, "valence": v_peaks},
        "delta": {
            "arousal_range": round(float(arousal.max() - arousal.min()), 4),
            "valence_range": round(float(valence.max() - valence.min()), 4),
            "arousal_slope": round(slope, 5),
        },
        "peak": {"time_sec": round(times[peak_i], 2), "arousal": round(float(arousal[peak_i]), 3)},
        "trough": {"time_sec": round(times[trough_i], 2), "arousal": round(float(arousal[trough_i]), 3)},
    }
