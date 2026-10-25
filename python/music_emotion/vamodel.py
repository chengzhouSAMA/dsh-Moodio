# -*- coding: utf-8 -*-
"""Valence-Arousal 估计模型。

立场声明（重要，避免误读）：
    这不是在 DEAM/PMEmo 上训练出来的判别模型，而是一个**先验驱动的可解释基线**。
    权重来自音乐心理学文献中被反复复现的方向性结论（见 docs/RESEARCH.md 的引用），
    以及 DEAM 上"节奏规律性/谱通量→唤醒，和声复杂度/调式→效价"的归因结论。
    它给的是**相对情绪坐标**，不是人群平均评分；绝对值有偏，方向性可靠。

设计取舍：
    - 每个特征先归一到 [0,1]，再以中性参考点(neutral)为中心映射到 [-1,1]。
      这样"普通流行歌"大致落在 0 附近，而不是所有歌都挤在 0.7。
    - 保留逐项贡献(evidence)，让 agent 能解释"为什么判成这样"，而不是只给一个黑盒数。
    - 输出带 confidence，显式暴露"这条音轨的证据有多可靠"。
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "estimate_va", "label_for", "explain", "calibrate", "raw_scores",
    "FEATURE_RANGES", "AROUSAL_WEIGHTS", "VALENCE_WEIGHTS",
    "AROUSAL_NEUTRAL", "VALENCE_NEUTRAL", "CALIBRATION_LITERATURE",
]

#: 每个特征的归一区间 (lo, hi)，依据 MIR 常规取值范围设定；超出即饱和。
#:
#: 修正记录：rms_db 早期设为 (-38, -6)，那是"含静音/未母带处理"的通用音频区间。
#: 实测本库 523 首真实音乐响度集中在 -14.6 ~ -5.8 dB，映射后 mean=0.89、p95=1.000，
#: 整条通道贴顶，等于失去区分度。改为围绕实测分布展开，使音乐区间落在中段。
FEATURE_RANGES: Dict[str, Tuple[float, float]] = {
    "tempo_bpm": (60.0, 170.0),
    "onset_rate": (0.3, 6.0),
    # 见上：按音乐实测分布重标定，保留两端余量
    "rms_db": (-26.0, -4.0),
    "spectral_flux": (0.08, 0.45),
    # HPSS 在密集电子编曲上常给出 0.03~0.10 的打击能量占比，
    # 早期设 (0.15,0.65) 导致全部样本落在 0，该通道形同常数。
    "percussive_ratio": (0.02, 0.30),
    "spectral_centroid_hz": (600.0, 3500.0),
    "key_clarity": (0.3, 0.9),
}

#: pulse_clarity 自身已是 [0,1]，直接使用。
CLIP01_FEATURES = {"pulse_clarity"}

#: 唤醒度通道权重。正号=该特征越强唤醒越高。总和为 1。
AROUSAL_WEIGHTS: Dict[str, float] = {
    "tempo_bpm": 0.30,
    "onset_rate": 0.20,
    "rms_db": 0.18,
    "spectral_flux": 0.16,
    "pulse_clarity": 0.10,
    "percussive_ratio": 0.06,
}

#: 效价通道权重。harmonic_complexity 为负号：和声越复杂，效价越低。
VALENCE_WEIGHTS: Dict[str, float] = {
    "mode_score": 0.42,
    "harmonic_complexity": -0.25,
    "spectral_centroid_hz": 0.15,
    "key_clarity": 0.10,
    "pulse_clarity": 0.08,
}

#: 中性参考点：所有归一特征值恰为 0.5 时，V/A 应输出 0。
#:
#: 这组常量是可验算的不动点，不是拟合出来的：
#:   VALENCE 权重和 0.50 -> 0.50 × 0.5 = 0.2500
#:   AROUSAL 权重和 1.00 -> 1.00 × 0.5 = 0.5000
#: 由 tests/validate_calibration.py 的"中性不变量"用例强制保证。
#:
#: 修正记录（重要）：早期版本设为 (0.47, 0.46)，依据是"中等速度、中等亮度"的
#: 直觉假设。实测 523 首真实曲目后证伪——经 FEATURE_RANGES 映射后各通道均值并
#: 不等于 0.5（harmonic_complexity ≈0.17、pulse_clarity ≈0.22、rms_db ≈0.89），
#: 常量中性点因此把 valence 系统性拉负：全库 mean=-0.222，264/523 判"低沉/忧郁"，
#: 4 个象限只有 2 个能出现，等于把一个二维模型退化成一条线。
#:
#: 需要"相对本库居中"时用 calibrate() 按语料统计，**两种模式不要混用**。
AROUSAL_NEUTRAL = 0.5000
VALENCE_NEUTRAL = 0.2500

#: 固定中性点的出处，输出里要带上，避免被当成"绝对人类评分"。
CALIBRATION_LITERATURE = "fixed:neutral-input-invariant"

#: 调式分数单独处理：它有正负号，不能按 [0,1] 归一。
MODE_SCORE_RANGE = (-0.30, 0.30)

#: 四象限标签（英文 id + 中文释义）。
_LABELS = {
    "elated": ("elated", "激昂/高涨"),
    "peaceful": ("peaceful", "平静/舒展"),
    "tense": ("tense", "紧张/压迫"),
    "melancholic": ("melancholic", "低沉/忧郁"),
    "neutral": ("neutral", "中性"),
}


def _norm(value: float, lo: float, hi: float) -> float:
    if hi <= lo or not np.isfinite(value):
        return 0.0
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))


def _f(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _feature_norm(name: str, feats: Dict[str, Any]) -> float:
    """把单个特征映射到 [0,1]，缺失或非法值退化为中性 0.5。"""
    if name == "mode_score":
        lo, hi = MODE_SCORE_RANGE
        v = _f(feats.get("mode_score"), 0.0)
        return float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))

    if name == "harmonic_complexity":
        return _f(feats.get("harmonic_complexity"), 0.5)

    if name in CLIP01_FEATURES:
        return float(np.clip(_f(feats.get(name), 0.5), 0.0, 1.0))

    if name in FEATURE_RANGES:
        lo, hi = FEATURE_RANGES[name]
        return _norm(_f(feats.get(name), (lo + hi) / 2.0), lo, hi)

    raw = feats.get(name)
    if raw is None:
        return 0.5
    return float(np.clip(_f(raw, 0.5), 0.0, 1.0))


def estimate_va(
    features: Dict[str, Any],
    neutral: Optional[Tuple[float, float]] = None,
) -> Dict[str, Any]:
    """由特征字典估计 V-A。

    Args:
        features: 特征字典。
        neutral: 可选的 (valence_neutral, arousal_neutral)。默认用模块级常量，
            即"中性输入→0"的不动点。传入 calibrate() 的结果即切换为按库标定模式。

    Returns:
        {
          "valence": float in [-1,1],
          "arousal": float in [-1,1],
          "intensity": float in [0,1],
          "evidence": {"arousal": {feat: contribution}, "valence": {...},
                       "normalized": {feat: 0..1}},
          "confidence": float in [0,1],
          "label": str, "label_zh": str
        }
    """
    v_neutral, a_neutral = neutral if neutral is not None else (VALENCE_NEUTRAL, AROUSAL_NEUTRAL)

    a_scores: Dict[str, float] = {}
    v_scores: Dict[str, float] = {}

    for name, w in AROUSAL_WEIGHTS.items():
        a_scores[name] = w * _feature_norm(name, features)
    for name, w in VALENCE_WEIGHTS.items():
        v_scores[name] = w * _feature_norm(name, features)

    a_raw = sum(a_scores.values())
    v_raw = sum(v_scores.values())

    # 以中性点为中心映射到 [-1,1]
    arousal = float(np.clip(2.0 * (a_raw - a_neutral), -1.0, 1.0))
    valence = float(np.clip(2.0 * (v_raw - v_neutral), -1.0, 1.0))

    intensity = float(min(1.0, math.hypot(valence, arousal) / math.sqrt(2.0)))

    label, label_zh = label_for(valence, arousal)

    evidence = {
        "arousal": {k: round(v, 4) for k, v in a_scores.items()},
        "valence": {k: round(v, 4) for k, v in v_scores.items()},
        "normalized": {
            k: round(_feature_norm(k, features), 4)
            for k in set(AROUSAL_WEIGHTS) | set(VALENCE_WEIGHTS)
        },
    }

    confidence = _confidence(features, valence, arousal)

    # 离散情绪词：把连续坐标翻译成日常说法（悲伤/欢快/平静/紧张…）。
    # 这是**附加**输出，不替代 label —— label 保留四象限 id 供程序判断，
    # emotion 给人看。两者都从同一组 V/A 派生，不会矛盾。
    from .emotion_taxonomy import emotion_profile
    emo = emotion_profile(valence, arousal)

    return {
        "valence": valence,
        "arousal": arousal,
        "intensity": intensity,
        "evidence": evidence,
        "confidence": confidence,
        "label": label,
        "label_zh": label_zh,
        "emotion": emo["emotion"],
        "emotion_id": emo["emotion_id"],
        "emotion_detail": emo["detail"],
        "emotion_intensity": emo["intensity"],
        "emotion_desc": emo["desc"],
        "emotion_magnitude": emo["magnitude"],
    }


def label_for(valence: float, arousal: float, deadzone: float = 0.18) -> Tuple[str, str]:
    """把连续 V-A 映射到四象限标签。

    注意：这个函数的 `neutral` 与 `emotion_taxonomy` 的"平淡"口径**不同**，
    是有意保留的两套语义：

      - `label_for`（本函数）：研究用四象限，deadzone=0.18，
        `neutral` 表示"两个维度都不够显著"。适合程序做粗分组。
      - `emotion_taxonomy.emotion_profile`：日常情绪词，只在模长 <0.06 时给"平淡"。
        像《诀别书》V=-0.163/A=-0.129 这种"安静地悲伤"会被它判为"悲伤"，
        而本函数因 |A|<0.18 仍归 neutral —— 两者都对，只是回答的问题不同。

    面向用户的输出请用 `emotion`；`label` 留给需要稳定四分类的程序。

    修正记录（重要）：早期版本只检查 valence 是否在死区内就判 neutral，
    **arousal 完全不参与**。后果是"效价不明确但唤醒度很确定"的曲子被吞掉：
    本库 794 首里有 45 首（如 162BPM、arousal=+0.47 的 Runaway Baby）被判成"中性"，
    等于把二维模型的一整个维度丢掉。现在两个维度都必须落在死区内才算中性。
    """
    v_high = valence > deadzone
    v_low = valence < -deadzone
    a_high = arousal > deadzone
    a_low = arousal < -deadzone

    if not v_high and not v_low and not a_high and not a_low:
        key = "neutral"
    elif v_high:
        key = "elated" if a_high else "peaceful"
    elif v_low:
        key = "tense" if a_high else "melancholic"
    else:
        # valence 在死区内，但 arousal 明确：此时不应报 neutral，
        # 用 arousal 决定"高能但情绪不明确"的一侧。
        key = "elated" if a_high else "melancholic"

    return _LABELS[key]


def _confidence(features: Dict[str, Any], valence: float, arousal: float) -> float:
    """置信度 = "这条音轨的特征有多可信"，不是"预测有多准"。

    修正记录：早期版本由 4 个几乎恒为 1 的因子几何平均，输出恒等于 1.0，
    等于没有置信度。现在改为惩罚项叠加：从 1.0 出发，按可诊断的问题扣分，
    使低质量输入（短、静、削波、无调性、无节拍）确实得到低分。
    """
    score = 1.0
    reasons: List[str] = []

    duration = _f(features.get("duration_sec"), 0.0)
    # 窗口级调用会传入很短的真实时长，那是设计使然而不是缺陷；
    # 只有"整条音轨"级别的判断才应惩罚短时长。用 n_segments 是否给定来区分。
    if duration > 0 and "is_window" not in features:
        if duration < 10.0:
            score *= 0.45
            reasons.append("very_short")
        elif duration < 30.0:
            score *= 0.80
            reasons.append("short")

    rms_db = _f(features.get("rms_db"), -25.0)
    if rms_db < -34.0:
        score *= 0.55
        reasons.append("very_quiet")
    elif rms_db < -28.0:
        score *= 0.80
        reasons.append("quiet")

    peak = _f(features.get("peak"), 0.9)
    if peak >= 0.995:
        score *= 0.75
        reasons.append("clipping")

    key_clarity = _f(features.get("key_clarity"), 0.0)
    if key_clarity < 0.35:
        score *= 0.60
        reasons.append("no_clear_key")

    if _f(features.get("pulse_clarity"), 0.0) < 0.15:
        score *= 0.70
        reasons.append("no_clear_pulse")

    return float(np.clip(score, 0.0, 1.0))


def explain(result: Dict[str, Any], top_k: int = 3) -> Dict[str, List[Dict[str, float]]]:
    """把 evidence 转成"最主要的推动因素"，供报告/agent 引用。"""
    out: Dict[str, List[Dict[str, float]]] = {}
    for dim in ("arousal", "valence"):
        scored = sorted(result["evidence"][dim].items(), key=lambda kv: abs(kv[1]), reverse=True)
        out[dim] = [{"feature": k, "contribution": round(v, 4)} for k, v in scored[:top_k]]
    return out


def raw_scores(features: Dict[str, Any]) -> Tuple[float, float]:
    """返回未居中、未截断的 (v_raw, a_raw)。标定与诊断都用它。"""
    v_raw = sum(w * _feature_norm(k, features) for k, w in VALENCE_WEIGHTS.items())
    a_raw = sum(w * _feature_norm(k, features) for k, w in AROUSAL_WEIGHTS.items())
    return float(v_raw), float(a_raw)


def calibrate(
    feature_dicts: List[Dict[str, Any]],
    method: str = "median",
) -> Dict[str, Any]:
    """按语料统计量给出中性点，用于"相对本库居中"模式。

    为什么需要它：固定中性点保证"没有证据→中性"，跨库可比；
    但一个整体偏明亮的曲库里，几乎所有曲子都会落在 valence>0 一侧，
    此时用中位数居中能让四象限都用到。**两种模式不能混用**，
    否则同一首歌在两个不同曲库里会得到不同符号，结论无法比较。

    Args:
        feature_dicts: 若干条特征字典（通常是整库）。
        method: "median" 或 "mean"。

    Returns:
        {"neutral": (v, a), "method": ..., "n": ..., "note": ...}
    """
    if not feature_dicts:
        raise ValueError("calibrate() needs at least one feature dict")
    if method not in ("median", "mean"):
        raise ValueError(f"unknown method: {method}")

    pairs = [raw_scores(f) for f in feature_dicts]
    vs = np.array([p[0] for p in pairs], dtype=float)
    as_ = np.array([p[1] for p in pairs], dtype=float)
    fn = np.median if method == "median" else np.mean

    return {
        "neutral": (float(fn(vs)), float(fn(as_))),
        "method": method,
        "n": len(feature_dicts),
        "note": (
            "corpus-relative: values are centered on this collection, "
            "so they are NOT comparable across different collections"
        ),
    }
