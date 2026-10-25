# -*- coding: utf-8 -*-
"""核心数据结构与错误类型。

设计约束：
- 所有对外结构必须可 JSON 序列化（CLI 默认输出 JSON）。
- 数值统一 round 到适度精度，避免 JSON 噪声。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = [
    "MusicEmotionError",
    "AudioDecodeError",
    "EMOTION_SCHEMA_VERSION",
    "ClipInfo",
    "SegmentEmotion",
    "EmotionReport",
]

#: 输出 schema 版本。字段含义变更时必须升次版本号。
EMOTION_SCHEMA_VERSION = "1.0"


class MusicEmotionError(Exception):
    """带机器可读 code 的领域错误，CLI 会转成结构化错误 JSON。"""

    def __init__(self, code: str, message: str, suggestion: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.suggestion = suggestion


class AudioDecodeError(MusicEmotionError):
    def __init__(self, message: str, suggestion: str = "") -> None:
        super().__init__("AUDIO_DECODE_FAILED", message, suggestion)


@dataclass
class ClipInfo:
    """音频片段的客观元信息。"""

    path: Optional[str]
    duration_sec: float
    sample_rate: int
    channels: int
    source_format: str
    subtype: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "duration_sec": round(self.duration_sec, 2),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "source_format": self.source_format,
            "subtype": self.subtype,
        }


@dataclass
class SegmentEmotion:
    """一个时间窗的 V-A 估计及其证据。

    valence / arousal 取值 [-1, 1]，0 表示中性。
    """

    start_sec: float
    end_sec: float
    valence: float
    arousal: float
    intensity: float
    confidence: float
    evidence: Dict[str, float] = field(default_factory=dict)
    label: str = ""

    def to_dict(self, include_evidence: bool = True) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "start_sec": round(self.start_sec, 2),
            "end_sec": round(self.end_sec, 2),
            "valence": round(self.valence, 3),
            "arousal": round(self.arousal, 3),
            "intensity": round(self.intensity, 3),
            "confidence": round(self.confidence, 3),
            "label": self.label,
        }
        if include_evidence:
            out["evidence"] = {k: round(v, 4) for k, v in self.evidence.items()}
        return out


@dataclass
class EmotionReport:
    """完整分析结果。"""

    clip: ClipInfo
    global_valence: float
    global_arousal: float
    global_label: str
    segments: List[SegmentEmotion]
    features: Dict[str, Any]
    arc: str
    delta: Dict[str, float]
    quality: Dict[str, Any]
    modes: Dict[str, Any] = field(default_factory=dict)
    neural: Optional[Dict[str, Any]] = None
    warnings: List[str] = field(default_factory=list)
    #: 音色攻击性（texture）：{"texture": 0..1, "band": low/mid/high, "parts": {...}}
    texture: Optional[Dict[str, Any]] = None
    #: 规模感（grandeur）：{"grandeur": 0..1, "band": ..., "parts": {...}}
    grandeur: Optional[Dict[str, Any]] = None
    #: 四轴组合后的日常说法：{"emotion": 悲壮/悲伤/昂扬..., ...}
    four_axis: Optional[Dict[str, Any]] = None

    def to_dict(self, include_segments: bool = True) -> Dict[str, Any]:
        # 离散情绪词由同一组 global V/A 派生，保证与 emotion.valence/arousal 永不矛盾。
        from .emotion_taxonomy import emotion_profile

        emo = emotion_profile(self.global_valence, self.global_arousal)
        payload: Dict[str, Any] = {
            "schema_version": EMOTION_SCHEMA_VERSION,
            "clip": self.clip.to_dict(),
            "emotion": {
                "emotion": emo["emotion"],
                "emotion_id": emo["emotion_id"],
                "detail": emo["detail"],
                "intensity": emo["intensity"],
                "desc": emo["desc"],
                "magnitude": emo["magnitude"],
                "valence": round(self.global_valence, 3),
                "arousal": round(self.global_arousal, 3),
                "label": self.global_label,
                "scale": "[-1,1]; 0 is neutral",
            },
            "deltas": {k: round(v, 4) for k, v in self.delta.items()},
            "arc": self.arc,
            "modes": self.modes,
            "features": self.features,
            "quality": self.quality,
        }
        # 四轴情绪：把"悲壮"与"悲伤"、"昂扬"与"欢快"分开的那一层。
        # 字段名刻意叫 emotion_4d，与旧的二维 emotion 并列，便于对比。
        if self.four_axis is not None:
            payload["emotion_4d"] = {
                **self.four_axis,
                "texture_parts": (self.texture or {}).get("parts"),
                "grandeur_parts": (self.grandeur or {}).get("parts"),
            }
        if self.texture is not None:
            payload["texture"] = self.texture
        if self.grandeur is not None:
            payload["grandeur"] = self.grandeur
        if include_segments:
            payload["segments"] = [s.to_dict() for s in self.segments]
        else:
            payload["segments"] = {
                "count": len(self.segments),
                "omitted": True,
                "hint": "rerun without --no-segments",
            }
        if self.neural is not None:
            payload["neural"] = self.neural
        if self.warnings:
            payload["warnings"] = list(self.warnings)
        return payload
