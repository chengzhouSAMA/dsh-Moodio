# -*- coding: utf-8 -*-
"""离散情绪词表：把连续的 V-A 坐标翻译成日常能说的情绪词。

为什么需要这一层：
    Russell 的 valence/arousal 是**研究用坐标**，适合做数值比较，
    但对普通使用者没有意义——没人会说"这首歌 melancholic 象限里 arousal=-0.13"。
    日常描述是离散的："悲伤""欢快""平静""紧张"。这一层就是做这个翻译。

词表选择：
    用四象限作为**主词**，因为它们是人最常用的分类，且恰好覆盖
    用户最常说的那几类（悲伤 / 欢快 / 平静 / 紧张）。
    再附一个 finer 的 `detail` 词用于"强度够明显时给更具体的说法"，
    避免把 0.2 的轻微悲伤和 0.6 的深切悲痛说成同一个词。

重要：这不是新增一个模型，而是**对已有 V-A 的重新命名**。
       V-A 的数值、标定、权重全都不变，所以已有的分析结果只需重贴标签即可。
"""
from __future__ import annotations

import math
from typing import Any, Dict, Tuple

__all__ = [
    "EMOTIONS",
    "FLAT_RADIUS",
    "VALUE_WEAK_BAND",
    "WEAK_LABELS",
    "DETAIL",
    "emotion_for",
    "emotion_profile",
]

#: 判定"没有明显情绪倾向"的半径。取 0.06 而不是原来的 0.18：
#: 旧的四象限死区 0.18 会把 27% 的曲目吞成中性（本库实测 214/794），
#: 其中包括"效价不高但唤醒极低"的深沉曲目（如《诀别书》V=-0.163/A=-0.129），
#: 那显然不是"没有情绪"，而是"安静地悲伤"。
#: 0.06 只留真正贴近原点的平坦音轨。
FLAT_RADIUS = 0.06

#: 效价的"弱带"。|V| 落在这里时，不宣称"悲伤/欢快"这种强判断，
#: 因为本库实测 V<0 占 49.7%，其中 31% 的 |V|<0.10 —— 那只是"略偏暗"，
#: 直接叫"紧张"或"悲伤"是过度解读。此时改用由唤醒度主导的中性词。
#:
#: 修正记录：第一版没有这条弱带，结果 258 首被判"紧张"（超过"平静"的 115 首），
#: 对一个流行/电音为主的曲库明显不合理——根因是"略负"被当成了"紧张"。
VALUE_WEAK_BAND = 0.10

#: 弱带内，按唤醒度给的中性词（不含效价方向判断）。
WEAK_LABELS = {
    "high": {"id": "restless", "zh": "躁动", "en": "restless",
             "desc": "情绪偏中性，但能量偏高、有压迫感"},
    "mid": {"id": "even", "zh": "平缓", "en": "even",
            "desc": "情绪与能量都接近中性"},
    "low": {"id": "low", "zh": "低回", "en": "low-key",
            "desc": "情绪偏中性，但能量很低、内敛安静"},
}

#: 主词表：每个象限一个日常用词。
#: 键是 (valence 符号, arousal 符号) -> (英文 id, 中文词, 一句话解释)
EMOTIONS: Dict[Tuple[int, int], Dict[str, str]] = {
    (1, 1): {
        "id": "cheerful",
        "zh": "欢快",
        "en": "cheerful",
        "desc": "明亮、有推进力，让人想跟着动",
    },
    (1, -1): {
        "id": "calm",
        "zh": "平静",
        "en": "calm",
        "desc": "明亮但不激动，松弛、舒展",
    },
    (-1, 1): {
        "id": "tense",
        "zh": "紧张",
        "en": "tense",
        "desc": "压迫、躁动、不安",
    },
    (-1, -1): {
        "id": "sad",
        "zh": "悲伤",
        "en": "sad",
        "desc": "低沉、内敛、带着失落感",
    },
}

#: 平坦音轨（|V|、|A| 都极小）用的词。
FLAT = {"id": "flat", "zh": "平淡", "en": "flat", "desc": "情绪倾向不明显，接近中性"}

#: 强度够明显时给的更具体说法（比主词更贴近听感）。
#: 阈值按 |V-A| 模长划分，0.35 以上才算"强烈"。
DETAIL = {
    (1, 1): {"mild": "轻快", "strong": "激昂"},
    (1, -1): {"mild": "安详", "strong": "舒展"},
    (-1, 1): {"mild": "疑虑", "strong": "焦躁"},
    (-1, -1): {"mild": "怅然", "strong": "哀伤"},
}

#: 模长分档：< mild_below 算轻微，>= strong_above 算强烈。
MILD_BELOW = 0.18
STRONG_ABOVE = 0.35


def _sign(x: float) -> int:
    """把数值压成 ±1；真零归到正侧（避免 0 落在哪边不确定）。"""
    return 1 if x >= 0 else -1


def emotion_profile(valence: float, arousal: float) -> Dict[str, Any]:
    """返回完整的离散情绪描述。

    Args:
        valence: [-1, 1]
        arousal: [-1, 1]

    Returns:
        {
          "emotion": 中文主词, "emotion_id": 英文 id,
          "detail": 更具体的词（轻微/强烈时才与主词不同）,
          "intensity": "flat" | "mild" | "moderate" | "strong",
          "magnitude": 0..1.41 的模长,
          "quadrant_zh": 象限说明,
        }
    """
    v = float(valence)
    a = float(arousal)
    mag = math.hypot(v, a)

    if mag < FLAT_RADIUS:
        return {
            "emotion": FLAT["zh"],
            "emotion_id": FLAT["id"],
            "detail": FLAT["zh"],
            "intensity": "flat",
            "magnitude": round(mag, 4),
            "desc": FLAT["desc"],
        }

    # 效价落在弱带内：不宣称方向性情绪，改用唤醒度主导的中性词。
    # 这样 V=-0.05/A=+0.40 不会被称为"紧张"（它只是能量高、情绪不明显）。
    # 边界取闭区间：|V| 恰好等于阈值时保守地算"弱"，避免 0.1000001 这种
    # 浮点误差让同一首歌在两次运行里换词。
    if abs(v) <= VALUE_WEAK_BAND:
        band_key = "high" if a >= MILD_BELOW else ("low" if a <= -MILD_BELOW else "mid")
        weak = WEAK_LABELS[band_key]
        return {
            "emotion": weak["zh"],
            "emotion_id": weak["id"],
            "detail": weak["zh"],
            "intensity": "moderate",
            "magnitude": round(mag, 4),
            "desc": weak["desc"],
        }

    key = (_sign(v), _sign(a))
    base = EMOTIONS[key]

    if mag < MILD_BELOW:
        band = "mild"
        intensity = "mild"
    elif mag >= STRONG_ABOVE:
        band = "strong"
        intensity = "strong"
    else:
        band = "mild"
        intensity = "moderate"

    detail = DETAIL[key].get(band, base["zh"]) if intensity != "moderate" else base["zh"]

    return {
        "emotion": base["zh"],
        "emotion_id": base["id"],
        "detail": detail,
        "intensity": intensity,
        "magnitude": round(mag, 4),
        "desc": base["desc"],
    }


def emotion_for(valence: float, arousal: float) -> Tuple[str, str, str]:
    """便捷入口：返回 (中文主词, 英文 id, 具体词)。"""
    p = emotion_profile(valence, arousal)
    return p["emotion"], p["emotion_id"], p["detail"]
