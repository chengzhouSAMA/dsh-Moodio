# -*- coding: utf-8 -*-
"""四轴情绪判定：valence / arousal / texture / grandeur。

## 为什么需要四轴（两次真实误判的教训）

**第一次**：只有 valence+arousal。结果全库只有 7 种可能，
小调钢琴独奏与低频失真电子只要 V/A 相近就得到同一个词。
用户指出："低沉不一定是悲伤的，节奏快也不一定是欢快的。"

**第二次**：加了 texture（攻击性 = 噪声+打击+响度+亮度）。
结果《在虚无中永存 - 英雄主义》被判"悲伤"——实际它是**悲壮**的。
原因：这首歌动态 34.6dB、40.9% 时间满响、对比度 27.8，是全库最宏大的；
但它**没有失真、没有高频噪声**，所以 texture 只有 0.10（很低）。
于是"管弦式的壮阔"被误读成了"安静"。

教训：**"力量"有两种，必须分开度量**

    texture  = 攻击性：失真、打击、噪声      -> 摇滚/电子/金属高
    grandeur = 规模感：动态落差、持续强奏、厚织体 -> 管弦/史诗高

两者可任意组合：
    英雄主义      texture 0.10 + grandeur 0.80 = 悲壮（宏大但不粗糙）
    Bury the Light texture 高   + grandeur 中   = 激昂（有冲击但不算宏大）
    诀别书        texture 0.15 + grandeur 0.60 = 悲伤（内敛、规模不大）
"""
from __future__ import annotations

import math
from typing import Any, Dict, Tuple

__all__ = ["classify_four_axis", "AXIS_LABELS"]

#: 效价弱带：|V| 小于此值时不宣称情绪方向。
VALUE_WEAK_BAND = 0.10
#: 唤醒弱带：|A| 小于此值时不区分"快/慢"。
AROUSAL_WEAK_BAND = 0.10

#: 四轴组合表。键 = (valence 符号, grandeur 档, texture 档)
#:
#: 决策顺序有意如此：**先看宏大度再看攻击性**。
#: 因为"宏大"对情绪词的改变比"粗糙"更根本：
#:   同样是小调低沉，grandeur 高 -> 悲壮（史诗感）；grandeur 低 -> 悲伤（私人感）
#:   而 texture 高低只决定"是否带躁动/失真"。
AXIS_LABELS: Dict[Tuple[str, str, str], Dict[str, str]] = {
    # ================= 负效价 =================
    # 宏大：庄严、史诗，压抑中有分量
    ("neg", "high", "high"): {"id": "somber_majestic", "zh": "悲壮",
                              "desc": "沉重而有规模感，压抑中带力量"},
    ("neg", "high", "mid"): {"id": "somber_majestic", "zh": "悲壮",
                             "desc": "厚重、庄严，情绪往下但有分量"},
    ("neg", "high", "low"): {"id": "somber_majestic", "zh": "悲壮",
                             "desc": "宏大而不粗糙，庄严的沉郁"},
    # 中等规模 + 高攻击性：压抑的爆发
    ("neg", "mid", "high"): {"id": "anguished", "zh": "愤懑",
                             "desc": "压抑而有冲击力，接近爆发"},
    ("neg", "mid", "mid"): {"id": "sad", "zh": "悲伤", "desc": "低沉，带有一定推进感"},
    ("neg", "mid", "low"): {"id": "sad", "zh": "悲伤", "desc": "低沉、内敛"},
    # 规模小：私人的、安静的
    ("neg", "low", "high"): {"id": "heavy", "zh": "沉重",
                             "desc": "音色有攻击性，但整体规模不大"},
    ("neg", "low", "mid"): {"id": "sad", "zh": "悲伤", "desc": "低沉，带一点厚度"},
    ("neg", "low", "low"): {"id": "sad", "zh": "悲伤", "desc": "低沉、内敛，情绪往下沉"},

    # ================= 正效价 =================
    ("pos", "high", "high"): {"id": "triumphant", "zh": "昂扬",
                              "desc": "宏大且有力，情绪向上、外放"},
    ("pos", "high", "mid"): {"id": "triumphant", "zh": "昂扬",
                             "desc": "有规模感，明亮而开阔"},
    ("pos", "high", "low"): {"id": "majestic", "zh": "壮丽",
                             "desc": "宏大但柔和，辽阔、舒展"},
    ("pos", "mid", "high"): {"id": "passionate", "zh": "热烈",
                             "desc": "高能、有冲击力，情绪外放"},
    ("pos", "mid", "mid"): {"id": "cheerful", "zh": "欢快", "desc": "明亮、有推进力"},
    ("pos", "mid", "low"): {"id": "calm", "zh": "平静", "desc": "明亮而松弛"},
    ("pos", "low", "high"): {"id": "bright", "zh": "明快",
                             "desc": "音色明亮有弹性，规模不大"},
    ("pos", "low", "mid"): {"id": "cheerful", "zh": "欢快", "desc": "轻快、明亮"},
    ("pos", "low", "low"): {"id": "calm", "zh": "平静",
                            "desc": "明亮而松弛，柔和不刺激"},
}

#: 效价接近 0 时只描述气质，不宣称情绪方向。
_NEUTRAL: Dict[Tuple[str, str], Dict[str, str]] = {
    ("high", "high"): {"id": "intense", "zh": "强烈", "desc": "情绪方向不明显，但宏大且有冲击力"},
    ("high", "mid"): {"id": "grand", "zh": "宏伟", "desc": "情绪方向不明显，但有规模感"},
    ("high", "low"): {"id": "vast", "zh": "辽阔", "desc": "情绪方向不明显，但空间感很大"},
    ("mid", "high"): {"id": "intense", "zh": "强烈", "desc": "情绪方向不明显，但音色有冲击力"},
    ("mid", "mid"): {"id": "neutral", "zh": "平实", "desc": "情绪方向不明显"},
    ("mid", "low"): {"id": "even", "zh": "平缓", "desc": "情绪与音色都接近中性"},
    ("low", "high"): {"id": "edgy", "zh": "尖利", "desc": "情绪方向不明显，音色有攻击性"},
    ("low", "mid"): {"id": "neutral", "zh": "平实", "desc": "情绪方向不明显"},
    ("low", "low"): {"id": "plain", "zh": "素淡", "desc": "情绪与规模都不明显"},
}


def classify_four_axis(
    valence: float,
    arousal: float,
    texture: float,
    grandeur: float,
) -> Dict[str, Any]:
    """四轴组合成日常情绪说法。

    Args:
        valence: [-1,1] 情绪正负
        arousal: [-1,1] 激活程度
        texture: [0,1] 音色攻击性
        grandeur: [0,1] 规模感

    Returns:
        {"emotion": 中文, "emotion_id": 英文, "desc": 说明,
         "arousal_band"/"texture_band"/"grandeur_band": 各轴档位,
         "directional": 是否给出了情绪方向判断}
    """
    from .texture import grandeur_band, texture_band

    v = float(valence)
    a = float(arousal)
    g_band = grandeur_band(float(grandeur))
    t_band = texture_band(float(texture))

    if a > AROUSAL_WEAK_BAND:
        a_band = "high"
    elif a < -AROUSAL_WEAK_BAND:
        a_band = "low"
    else:
        a_band = "mid"

    mag = math.hypot(v, a)
    common = {
        "arousal_band": a_band,
        "texture_band": t_band,
        "grandeur_band": g_band,
        "magnitude": round(mag, 4),
        "texture": round(float(texture), 4),
        "grandeur": round(float(grandeur), 4),
    }

    if abs(v) <= VALUE_WEAK_BAND:
        base = _NEUTRAL[(g_band, t_band)]
        return {"emotion": base["zh"], "emotion_id": base["id"],
                "desc": base["desc"], "directional": False, **common}

    key = ("pos" if v > 0 else "neg", g_band, t_band)
    base = AXIS_LABELS[key]
    return {"emotion": base["zh"], "emotion_id": base["id"],
            "desc": base["desc"], "directional": True, **common}
