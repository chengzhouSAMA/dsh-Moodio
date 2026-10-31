# -*- coding: utf-8 -*-
"""分析流水线：把 IO / 特征 / V-A / 弧线 / 神经侧写串起来。"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

import numpy as np

from . import arc as arc_mod
from .audio_io import ANALYSIS_SR, load_audio
from .features import extract_features
from .neural import NeuralProfiler
from .types import ClipInfo, EmotionReport, MusicEmotionError, SegmentEmotion
from .vamodel import estimate_va, explain

__all__ = ["analyze_file", "analyze_waveform"]


def _analyze_axes_v2(y: np.ndarray, sr: int) -> Dict[str, Any]:
    """对全部已加载音频计算独立十二轴，不做隐式时长截断。"""
    from . import axes_v2 as axes_mod

    audio = y
    if sr != axes_mod.SR:
        import librosa
        audio = librosa.resample(y, orig_sr=sr, target_sr=axes_mod.SR)
    if len(audio) < 3 * axes_mod.SR:
        raise MusicEmotionError(
            "AXES_V2_TOO_SHORT", "12-axis analysis needs at least three seconds of audio",
            "disable --axes-v2 for very short clips",
        )
    features, segments = axes_mod.extract(audio)
    return {
        "axes": axes_mod.score(features, segments, axes_mod.DEFAULT_RANGES),
        "seconds_analyzed": round(len(audio) / axes_mod.SR, 1),
        "sampling": "full_analyzed_audio",
        "scale": "[0,1] acoustic diagnostics; not human ratings",
        "note": "independent 12-axis prior over the entire analyzed audio; not interchangeable with four-axis emotion scores",
    }


def analyze_waveform(
    y: np.ndarray,
    sr: int,
    clip: ClipInfo,
    segment_sec: float = 5.0,
    hop_sec: float = 1.0,
    with_neural: bool = False,
    neural_model_dir: Optional[str] = None,
    with_axes_v2: bool = False,
) -> EmotionReport:
    """对已加载的波形做完整分析。"""
    t0 = time.time()
    warnings = []

    raw = extract_features(y, sr, segment_sec=segment_sec, hop_sec=hop_sec)
    global_feats = dict(raw["global"])
    global_feats["duration_sec"] = clip.duration_sec

    global_va = estimate_va(global_feats)

    # 音色张力（攻击性）与宏大度（规模感）：两个独立维度。
    # V-A 只有两个数，会把"低沉但有力"和"低沉而柔软"压成同一个点；
    # 而"力量"本身有两种（粗糙 vs 宏大），必须分别度量——
    # 《英雄主义》texture=0.10 但 grandeur=0.80，是"悲壮"不是"悲伤"。
    from .four_axis import classify_four_axis
    from .texture import grandeur_score, texture_score

    global_tex = texture_score(global_feats)
    global_gran = grandeur_score(global_feats)
    four = classify_four_axis(
        global_va["valence"], global_va["arousal"],
        global_tex["texture"], global_gran["grandeur"],
    )

    axes_v2 = _analyze_axes_v2(y, sr) if with_axes_v2 else None

    segments = []
    for seg in raw["segments"]:
        result = estimate_va(seg["features"])
        segments.append(
            SegmentEmotion(
                start_sec=float(seg["start_sec"]),
                end_sec=float(seg["end_sec"]),
                valence=float(result["valence"]),
                arousal=float(result["arousal"]),
                intensity=float(result["intensity"]),
                confidence=float(result["confidence"]),
                evidence=result["evidence"]["normalized"],
                label=result["label"],
            )
        )

    arc_info = arc_mod.build_arc([s.to_dict() for s in segments])

    neural = None
    if with_neural:
        profiler = NeuralProfiler(neural_model_dir) if neural_model_dir else NeuralProfiler()
        neural = profiler.profile(y, sr)
        neural["agreement"] = _agreement(neural, arc_info)
        if neural.get("available") and neural["dynamic_range"]["arousal"] < 0.05:
            warnings.append(
                "neural sidecar has very low arousal dynamic range on this track; "
                "treat it as uninformative rather than as a contradicting signal"
            )

    modes = {
        "acoustic": explain(global_va, top_k=3),
        "primary": "acoustic_prior",
        "primary_note": (
            "rule-based prior calibrated on published feature->emotion directions; "
            "relative ranking is reliable, absolute calibration is not"
        ),
    }

    quality = {
        "confidence": round(global_va["confidence"], 3),
        "sampling_rate_hz": sr,
        "analysis_sr_hz": ANALYSIS_SR,
        "segment_sec": segment_sec,
        "hop_sec": hop_sec,
        "n_segments": len(segments),
        "elapsed_sec": round(time.time() - t0, 2),
        "peak_amplitude": round(float(np.abs(y).max()), 4),
        "rms_db": round(float(global_feats["rms_db"]), 2),
        "clipping_suspected": bool(float(np.abs(y).max()) >= 0.995),
    }
    if quality["clipping_suspected"]:
        warnings.append("signal peaks at full scale; dynamic features may be distorted")

    if clip.duration_sec < segment_sec:
        warnings.append(
            f"clip shorter than one segment ({segment_sec:.0f}s); arc analysis is not meaningful"
        )

    return EmotionReport(
        clip=clip,
        global_valence=float(global_va["valence"]),
        global_arousal=float(global_va["arousal"]),
        global_label=global_va["label"],
        segments=segments,
        features={k: _round_value(v) for k, v in global_feats.items()},
        arc=arc_info["arc"],
        delta=arc_info["delta"],
        quality=quality,
        modes=modes,
        neural=neural,
        warnings=warnings,
        texture=global_tex,
        grandeur=global_gran,
        four_axis=four,
        axes_v2=axes_v2,
    )


def analyze_file(
    path: str,
    sr: int = ANALYSIS_SR,
    max_seconds: float = 0.0,
    segment_sec: float = 5.0,
    hop_sec: float = 1.0,
    with_neural: bool = False,
    neural_model_dir: Optional[str] = None,
    with_axes_v2: bool = False,
) -> EmotionReport:
    """从文件路径走完整流程。"""
    y, sr, clip = load_audio(path, sr=sr, max_seconds=max_seconds)
    if max_seconds > 0:
        # 截断后重新计算真实分析时长，避免 confidence 用到原始时长
        clip.duration_sec = float(len(y)) / sr
    return analyze_waveform(
        y,
        sr,
        clip,
        segment_sec=segment_sec,
        hop_sec=hop_sec,
        with_neural=with_neural,
        neural_model_dir=neural_model_dir,
        with_axes_v2=with_axes_v2,
    )


def _agreement(neural: Dict[str, Any], arc_info: Dict[str, Any]) -> Dict[str, Any]:
    """神经侧写与声学基线的方向一致性。

    只比较"相对走势"：两条 arousal 序列的 Pearson 相关。
    这是两者唯一可比的部分——绝对值来自完全不同的标注体系。
    """
    if not neural.get("available") or not arc_info.get("series"):
        return {"available": False}

    n_rel = neural["relative"]["arousal_z"]
    a_ac = [p["arousal"] for p in arc_info["series"]]
    if len(n_rel) < 3 or len(a_ac) < 3:
        return {"available": False, "reason": "too few points"}

    # 重采样到同一长度后比较
    idx = np.linspace(0, len(n_rel) - 1, len(a_ac))
    n_resampled = np.interp(idx, np.arange(len(n_rel)), np.array(n_rel))
    a_arr = np.array(a_ac, dtype=np.float64)

    if a_arr.std() < 1e-6 or n_resampled.std() < 1e-6:
        return {"available": False, "reason": "degenerate series"}

    r = float(np.corrcoef(a_arr, n_resampled)[0, 1])
    return {
        "available": True,
        "method": "pearson r on resampled arousal series",
        "arousal_r": round(r, 3),
        "interpretation": (
            "aligned" if r >= 0.3 else "opposed" if r <= -0.3 else "weak/unrelated"
        ),
    }


def _round_value(v: Any) -> Any:
    if isinstance(v, float):
        return round(v, 2)
    return v
