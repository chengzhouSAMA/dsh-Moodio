# -*- coding: utf-8 -*-
"""music_emotion: 音频音乐情感(Valence-Arousal)分析。

设计目标：给 agent 用的、可解释、可交叉验证的音乐情感感知层。
不训练模型，不调外部 API，纯本地确定性计算。
"""
from .types import EMOTION_SCHEMA_VERSION, EmotionReport, MusicEmotionError, SegmentEmotion
from .vamodel import estimate_va, explain, label_for

__version__ = "1.0.0"

__all__ = [
    "EMOTION_SCHEMA_VERSION",
    "EmotionReport",
    "SegmentEmotion",
    "MusicEmotionError",
    "analyze_file",
    "analyze_waveform",
    "estimate_va",
    "explain",
    "label_for",
    "__version__",
]


def analyze_file(*args, **kwargs):
    """惰性转发，避免 import 本包时就拉起 librosa/torch。"""
    from .pipeline import analyze_file as _impl

    return _impl(*args, **kwargs)


def analyze_waveform(*args, **kwargs):
    from .pipeline import analyze_waveform as _impl

    return _impl(*args, **kwargs)
