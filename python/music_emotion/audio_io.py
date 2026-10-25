# -*- coding: utf-8 -*-
"""音频读取。

关键决策：用 soundfile 直接读 FLAC/WAV，不依赖 ffmpeg。
本机没有 ffmpeg，而目标素材正是 24bit/48k FLAC，soundfile 的 libsndfile 1.2.2 可直接解码。
遇 mp3/m4a 等 libsndfile 不支持的格式时给出明确报错与替代建议，而不是静默降级。
"""
from __future__ import annotations

import os
from typing import Tuple

import numpy as np

from .types import AudioDecodeError, ClipInfo, MusicEmotionError

#: 分析采样率。22050Hz 足够覆盖音乐情感相关的音色/节奏特征，且比 48k 快 ~5 倍。
ANALYSIS_SR = 22050

_SUPPORTED_EXT = {".flac", ".wav", ".ogg", ".aiff", ".aif", ".au", ".w64", ".caf", ".mp3"}


def _ensure_readable(path: str) -> None:
    if not os.path.exists(path):
        raise MusicEmotionError(
            "FILE_NOT_FOUND",
            f"audio file not found: {path}",
            "check the path; quote it if it contains spaces or CJK characters",
        )
    if not os.path.isfile(path):
        raise MusicEmotionError(
            "NOT_A_FILE", f"path is not a regular file: {path}", "pass an audio file path"
        )
    size = os.path.getsize(path)
    if size == 0:
        raise MusicEmotionError(
            "EMPTY_FILE", f"audio file is empty: {path}", "re-download or re-export the file"
        )
    ext = os.path.splitext(path)[1].lower()
    if ext and ext not in _SUPPORTED_EXT:
        raise MusicEmotionError(
            "UNSUPPORTED_FORMAT",
            f"extension {ext} is not in the known-supported set",
            "supported: flac, wav, ogg, aiff, au, w64, caf (mp3 only if libsndfile is built with mpg123)",
        )


def load_audio(
    path: str,
    sr: int = ANALYSIS_SR,
    mono: bool = True,
    max_seconds: float = 0.0,
) -> Tuple[np.ndarray, int, ClipInfo]:
    """读取音频为单声道 float32 波形。

    Args:
        path: 音频文件路径（支持中文/空格）。
        sr: 目标采样率。
        mono: 是否下混为单声道。
        max_seconds: >0 时只读取前 N 秒（用于快速预览）。

    Returns:
        (y, sr, clip_info)，y 归一化到 [-1, 1]。
    """
    _ensure_readable(path)

    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover
        raise MusicEmotionError(
            "DEPENDENCY_MISSING",
            f"soundfile is required: {exc}",
            "pip install soundfile",
        ) from exc

    try:
        info = sf.info(path)
    except Exception as exc:  # noqa: BLE001 - libsndfile 抛的异常类型不稳定
        raise AudioDecodeError(
            f"failed to read audio header: {exc}",
            "the file may be corrupted or use a codec libsndfile cannot handle; "
            "convert it first (e.g. ffmpeg -i in.m4a -c:a flac out.flac)",
        ) from exc

    if info.frames == 0:
        raise AudioDecodeError("audio contains zero frames", "re-export the file")

    try:
        y, file_sr = sf.read(
            path,
            frames=int(max_seconds * info.samplerate) if max_seconds > 0 else -1,
            dtype="float32",
            always_2d=False,
        )
    except Exception as exc:  # noqa: BLE001
        raise AudioDecodeError(
            f"failed to decode audio: {exc}",
            "convert the file to 16-bit or 24-bit PCM FLAC/WAV and retry",
        ) from exc

    if y.ndim > 1:
        if mono:
            y = y.mean(axis=1)
        else:
            y = y[:, 0]

    y = np.ascontiguousarray(y, dtype=np.float32)

    if file_sr != sr:
        import librosa

        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr, res_type="soxr_hq")

    peak = float(np.abs(y).max()) if y.size else 0.0
    if peak == 0.0:
        raise AudioDecodeError("audio is pure silence", "check the source file")

    dur = float(len(y)) / sr
    clip = ClipInfo(
        path=os.path.abspath(path),
        duration_sec=dur,
        sample_rate=file_sr,
        channels=info.channels,
        source_format=info.format,
        subtype=info.subtype,
    )
    return y, sr, clip
