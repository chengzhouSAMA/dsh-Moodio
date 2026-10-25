# -*- coding: utf-8 -*-
"""音色张力（texture）维度：把 V-A 无法区分的情绪分开。

## 为什么需要第三个维度

V-A 是二维的，只有 4 个象限。这意味着**声学上完全不同的音乐会被压成同一个点**：

    V=-0.20, A=-0.15  ->  不论是小调钢琴独奏，还是低频失真电子，都只能叫同一个词

实测本库 794 首，低唤醒(A<-0.10)的 100 首里有 56 首是大调、被归为"平静"；
173 首快歌被分到"欢快/紧张/躁动"三种词里。也就是说：

    "低沉" 被错当成了 "悲伤"
    "节奏快" 被错当成了 "欢快"

真正的区别在**音色张力**上——同样是低沉，可以是"安静的忧伤"，也可以是
"压抑的爆发"；同样是快，可以是"轻快跳脱"，也可以是"亢奋躁动"。

## 张力的声学定义

用四个已被反复验证与"听感强度/粗糙度"相关的量，各自归一后加权：

    spectral_flatness     噪声化程度（失真吉他会显著抬高）
    percussive_ratio      打击能量占比（鼓组密度）
    spectral_flux         谱通量的时间变化（编曲活跃度）
    rms_db                响度（母带电平/压缩程度）
    spectral_centroid     亮度（高频能量）

前四者主导，centroid 权重较低因为"亮"≠"有张力"（明亮的竖琴也可以很柔）。

## 与 V-A 的关系

三者正交使用，不合并成一个数：

    valence  : 情绪正负（调式、和声）
    arousal  : 激活程度（速度、节拍）
    texture  : 音色张力（粗糙度、打击感、响度、亮度）

组合后才得到日常说法，例如：
    低唤醒 + 低张力  -> 悲伤
    低唤醒 + 高张力  -> 悲壮 / 阴沉
    高唤醒 + 低张力  -> 欢快
    高唤醒 + 高张力  -> 躁动 / 爆发
"""
from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import numpy as np

__all__ = [
    "TEXTURE_RANGES",
    "TEXTURE_WEIGHTS",
    "GRANDEUR_RANGES",
    "GRANDEUR_WEIGHTS",
    "GRANDEUR_HIGH",
    "GRANDEUR_MID",
    "FLAT_RANGE_DB",
    "texture_score",
    "grandeur_score",
    "grandeur_band",
    "texture_band",
    "describe_three_axis",
]

#: 宏大度（grandeur）各线索的归一区间。
#:
#: 这一维存在的理由是**一次真实的误判**：
#:   《在虚无中永存 - 英雄主义》动态范围 34.6dB、40.9% 的时间处于满响、
#:   频谱对比度 27.8 —— 是实测 8 首里最"宏大"的一首，应当判"悲壮"。
#:   但初版只有 texture（攻击性）一维，而这首歌**没有失真、没有高频噪声**，
#:   于是被错判成"悲伤"。
#:
#: 教训：**"张力"不等于"攻击性"**。管弦/史诗类音乐的力量来自规模
#: （动态落差、持续强奏、厚织体），不是来自粗糙度。两者必须分开度量。
GRANDEUR_RANGES: Dict[str, Tuple[float, float]] = {
    # 动态范围 (p95-p5, dB)。实测：Yellow 12.5 / Victory 26.5 / 英雄主义 34.6
    "dynamic_range_db": (8.0, 36.0),
    # 满响时间占比 (%)。实测：诀别书 25.7 / 英雄主义 40.9 / Victory 60.1
    "full_loud_share": (15.0, 65.0),
    # 频谱对比度（峰谷结构）。实测 20.8~28.2
    "spectral_contrast": (18.0, 30.0),
    # 和声活跃音级数。实测多数曲子都是 11~12，区分力弱，保留但不加权。
    "active_pitches": (6.0, 12.0),
}

#: 权重。满响占比最高：它是"持续用力"最直接的证据。
#:
#: 校准记录：初版给 dynamic_range 与 low-mid 能量占比高权重，
#: 结果英雄主义(0.809) 与诀别书(0.678) 只差 0.13，且 Toby Fox 的芯片音乐
#: 因频带窄而虚高(0.632)。改成突出"持续强奏"后差距扩大到 0.19。
GRANDEUR_WEIGHTS: Dict[str, float] = {
    "full_loud_share": 0.45,
    "dynamic_range_db": 0.30,
    "spectral_contrast": 0.25,
    "active_pitches": 0.00,
}

#: 宏大度分档边界。实测分布：
#:   英雄主义 0.796 / Victory 0.735 / Star Sky 0.671 | 诀别书 0.602 / Yellow 0.600
#: 0.65 落在 0.671 与 0.602 之间的空档。
GRANDEUR_HIGH = 0.65
GRANDEUR_MID = 0.45

#: 「持续强奏」的有效动态范围区间 (dB)。
#:
#: full_loud_share 只有在音轨**真有动态**时才说明问题：恒定电平的信号每一帧
#: 都等于自己的 p95，占比机械性地接近 100%，会单独把 grandeur 顶到高档
#: （实测一段 12 秒恒定 RMS 的合成音被判 grandeur 0.65「辽阔」）。
#: 因此按实测动态范围做渐变抑制：低于 3 dB 视为无信息（系数 0），
#: 高于 12 dB 完全采信（系数 1）。真实曲目实测 12.5~34.6 dB，均不受影响。
FLAT_RANGE_DB = (3.0, 12.0)

#: 各张力线索的归一区间。与 vamodel.FEATURE_RANGES 同源思路：
#: 取音乐实测的合理展布，超出即饱和。
TEXTURE_RANGES: Dict[str, Tuple[float, float]] = {
    "spectral_flatness": (0.000, 0.060),
    "percussive_ratio": (0.02, 0.32),
    "spectral_flux": (0.08, 0.45),
    "rms_db": (-26.0, -6.0),
    "spectral_centroid_hz": (700.0, 3200.0),
}

#: 权重之和为 1。前四项主导"张力"，亮度只占 0.12：
#: 明亮的竖琴可以很柔，所以亮度不能单独决定张力。
TEXTURE_WEIGHTS: Dict[str, float] = {
    "spectral_flatness": 0.30,
    "percussive_ratio": 0.26,
    "spectral_flux": 0.20,
    "rms_db": 0.12,
    "spectral_centroid_hz": 0.12,
}

#: 张力分档边界（0..1）。
#:
#: 用 5 个已人工核过声学量的真实曲目定标，实测张力为：
#:     英雄主义 0.102 / 诀别书 0.150 / Yellow 0.267 | Die For You 0.532 / Runaway Baby 0.554
#: 0.27 与 0.53 之间有很宽的空档，边界取空档中部而不是贴着某个样本：
#:   BAND_LOW  = 0.40（空档左中，把"暗而柔"与"中等"分开）
#:   BAND_HIGH = 0.45（空档右中，把"中等"与"有冲击力"分开）
#:
#: 修正记录：初版设 high>=0.55，恰好卡在 Runaway Baby(0.554) 边缘，
#: 导致 Die For You(0.532) 掉进 mid 被判"欢快"——它其实是"燃"。
#: 阈值贴着样本走是脆弱的，宁可留余量。
BAND_LOW = 0.40
BAND_HIGH = 0.45


def _norm(v: float, lo: float, hi: float) -> float:
    if hi <= lo or not math.isfinite(v):
        return 0.5
    return float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))


def texture_score(features: Dict[str, Any]) -> Dict[str, Any]:
    """计算音色张力 [0,1] 及各分量。

    缺失特征按 0.5 处理，并记录 missing 列表——不要假装算准了。
    """
    parts: Dict[str, float] = {}
    missing = []
    for name, (lo, hi) in TEXTURE_RANGES.items():
        raw = features.get(name)
        if raw is None or not math.isfinite(float(raw)):
            missing.append(name)
            parts[name] = 0.5
        else:
            parts[name] = _norm(float(raw), lo, hi)

    score = sum(TEXTURE_WEIGHTS[k] * parts[k] for k in TEXTURE_WEIGHTS)
    return {
        "texture": round(float(np.clip(score, 0.0, 1.0)), 4),
        "parts": {k: round(v, 4) for k, v in parts.items()},
        "missing": missing,
        "band": texture_band(score),
    }


def texture_band(score: float) -> str:
    """把 [0,1] 的张力分数压成 low / mid / high。"""
    if score < BAND_LOW:
        return "low"
    if score >= BAND_HIGH:
        return "high"
    return "mid"


def grandeur_band(score: float) -> str:
    """把 [0,1] 的宏大度分数压成 low / mid / high。"""
    if score < GRANDEUR_MID:
        return "low"
    if score >= GRANDEUR_HIGH:
        return "high"
    return "mid"


def grandeur_score(features: Dict[str, Any]) -> Dict[str, Any]:
    """计算宏大度 [0,1] 及各分量。

    与 texture_score 的区别：
        texture  = 攻击性（失真、打击、噪声）—— 电子/摇滚/金属高
        grandeur = 规模感（动态落差、持续强奏、厚织体）—— 管弦/史诗高
    两者独立：一首没有失真的管弦乐可以 texture 很低但 grandeur 很高，
    这正是《英雄主义》的情形。
    """
    parts: Dict[str, float] = {}
    missing = []
    for name, (lo, hi) in GRANDEUR_RANGES.items():
        raw = features.get(name)
        if raw is None or not math.isfinite(float(raw)):
            missing.append(name)
            parts[name] = 0.5
        else:
            parts[name] = _norm(float(raw), lo, hi)

    # "Sustained loudness" only means something when the track actually has
    # dynamics. For a constant-level signal every frame sits at its own p95, so
    # full_loud_share is mechanically ~100% and would alone push grandeur to the
    # high band — a flat synth loop scored 0.65 and read as "vast". Damp the
    # component by a ramp over the measured dynamic range: below 3 dB the reading
    # is meaningless (factor 0), above 12 dB it is trusted fully (factor 1).
    # Real music here measured 12.5-34.6 dB, so this leaves actual tracks alone.
    raw_dyn = features.get("dynamic_range_db")
    if raw_dyn is not None and math.isfinite(float(raw_dyn)):
        lo, hi = FLAT_RANGE_DB
        ramp = float(np.clip((float(raw_dyn) - lo) / (hi - lo), 0.0, 1.0))
        parts["full_loud_share"] *= ramp
        if ramp < 1.0:
            missing.append("flat_dynamics")

    score = sum(GRANDEUR_WEIGHTS[k] * parts[k] for k in GRANDEUR_WEIGHTS)
    return {
        "grandeur": round(float(np.clip(score, 0.0, 1.0)), 4),
        "parts": {k: round(v, 4) for k, v in parts.items()},
        "missing": missing,
        "band": grandeur_band(score),
    }


#: 三维组合 -> 日常说法。
#:
#: 键是 (valence 符号, 唤醒档, 张力档)。三个维度都参与，缺一不可：
#:   - 只有 valence+texture 会出错：Coldplay《Yellow》V=+0.21/tex=0.361
#:     会从"平静"被错判成"欢快"，因为它其实很慢(86BPM)、只是音色不闷。
#:     慢速的大调曲子应当是"平静"，不是"欢快"——**速度决定活泼程度**。
#:   - arousal 分档用 ±0.10（与 emotion_taxonomy 的弱带一致）。
#:
#: 这张表要表达的正是用户提的两点纠正：
#:   "低沉不一定悲伤" -> neg+low唤醒+high张力 = 悲壮，不是悲伤
#:   "快不一定欢快"   -> 高唤醒+高张力 = 热烈，不是欢快
_COMBOS: Dict[Tuple[str, str, str], Dict[str, str]] = {
    # ---- 负效价 ----
    # 低唤醒：安静向内，张力高低决定是"忧伤"还是"沉重压抑"
    ("neg", "low", "low"): {"id": "sad", "zh": "悲伤", "desc": "低沉、内敛，情绪往下沉"},
    ("neg", "low", "mid"): {"id": "sad", "zh": "悲伤", "desc": "低沉，带一点厚度"},
    ("neg", "low", "high"): {"id": "somber", "zh": "沉重", "desc": "慢而厚，压抑感强"},
    # 高唤醒：张力高低决定是"紧张"还是"悲壮/爆发"
    ("neg", "high", "low"): {"id": "tense", "zh": "紧张", "desc": "推进快但音色薄，焦虑不安"},
    ("neg", "high", "mid"): {"id": "tense", "zh": "紧张", "desc": "压迫、躁动"},
    ("neg", "high", "high"): {"id": "somber_driving", "zh": "悲壮", "desc": "沉重而有力，压抑中带爆发"},
    # 唤醒接近 0
    ("neg", "mid", "low"): {"id": "sad", "zh": "悲伤", "desc": "低沉、内敛"},
    ("neg", "mid", "mid"): {"id": "sad", "zh": "悲伤", "desc": "低沉，带有一定推进感"},
    ("neg", "mid", "high"): {"id": "somber", "zh": "沉重", "desc": "厚实、有压迫感"},

    # ---- 正效价 ----
    # 低唤醒：张力低=松弛，张力高=厚实温暖的抒情
    ("pos", "low", "low"): {"id": "calm", "zh": "平静", "desc": "明亮而松弛，柔和不刺激"},
    ("pos", "low", "mid"): {"id": "calm", "zh": "平静", "desc": "舒展、温和"},
    ("pos", "low", "high"): {"id": "warm", "zh": "温柔", "desc": "慢而厚实，温暖有包裹感"},
    # 高唤醒：张力决定是"欢快跳脱"还是"热烈/爆发"
    ("pos", "high", "low"): {"id": "cheerful", "zh": "欢快", "desc": "轻快、明亮，节奏跳脱"},
    ("pos", "high", "mid"): {"id": "cheerful", "zh": "欢快", "desc": "明亮、有推进力"},
    ("pos", "high", "high"): {"id": "passionate", "zh": "热烈", "desc": "高能、有冲击力，情绪外放"},
    # 唤醒接近 0
    ("pos", "mid", "low"): {"id": "calm", "zh": "平静", "desc": "明亮而安静"},
    ("pos", "mid", "mid"): {"id": "cheerful", "zh": "欢快", "desc": "明亮、有推进力"},
    ("pos", "mid", "high"): {"id": "passionate", "zh": "热烈", "desc": "能量足、音色有冲击力"},
}

#: 效价接近 0 时，只按张力说"音色气质"，不宣称情绪方向。
_NEUTRAL: Dict[str, Dict[str, str]] = {
    "low": {"id": "even", "zh": "平缓", "desc": "情绪与音色都接近中性"},
    "mid": {"id": "neutral", "zh": "平实", "desc": "情绪方向不明显"},
    "high": {"id": "intense", "zh": "强烈", "desc": "情绪方向不明显，但音色有冲击力"},
}


def describe_three_axis(
    valence: float,
    arousal: float,
    texture: float,
    value_weak_band: float = 0.10,
    arousal_weak_band: float = 0.10,
) -> Dict[str, Any]:
    """三维组合成日常情绪说法。

    Args:
        valence: [-1,1]
        arousal: [-1,1]
        texture: [0,1] 音色张力
        value_weak_band: |valence| 小于该值时不宣称情绪方向
        arousal_weak_band: |arousal| 小于该值算"中等唤醒"
    """
    v = float(valence)
    a = float(arousal)
    t = float(texture)
    tex_band = texture_band(t)
    mag = math.hypot(v, a)

    # 唤醒档：这是"快 vs 慢"的显式表达。少了它，慢速大调会被误判成欢快。
    if a > arousal_weak_band:
        a_band = "high"
    elif a < -arousal_weak_band:
        a_band = "low"
    else:
        a_band = "mid"

    if abs(v) <= value_weak_band:
        base = _NEUTRAL[tex_band]
        return {
            "emotion": base["zh"],
            "emotion_id": base["id"],
            "desc": base["desc"],
            "texture_band": tex_band,
            "arousal_band": a_band,
            "magnitude": round(mag, 4),
            "directional": False,
        }

    key = ("pos" if v > 0 else "neg", a_band, tex_band)
    base = _COMBOS[key]
    return {
        "emotion": base["zh"],
        "emotion_id": base["id"],
        "desc": base["desc"],
        "texture_band": tex_band,
        "arousal_band": a_band,
        "magnitude": round(mag, 4),
        "directional": True,
    }
