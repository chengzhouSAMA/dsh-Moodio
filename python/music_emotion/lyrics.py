# -*- coding: utf-8 -*-
"""使用本地 faster-whisper 模型转写歌曲中的可辨认人声。

默认分析混音音轨；无网络请求、无自动模型下载、无歌词情绪推断。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from .types import MusicEmotionError

__all__ = ["transcribe_lyrics"]


def transcribe_lyrics(
    path: str,
    model_dir: Optional[str] = None,
    language: Optional[str] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """返回文本与时间戳；文本是 ASR 猜测，不是核对过的官方歌词。"""
    audio = Path(path).expanduser().resolve()
    if not audio.is_file():
        raise MusicEmotionError("FILE_NOT_FOUND", f"audio file not found: {path}", "pass an existing audio file")

    local_dir = model_dir or os.environ.get("MOODIO_LYRICS_MODEL_DIR")
    if not local_dir:
        raise MusicEmotionError(
            "LYRICS_MODEL_REQUIRED",
            "a local faster-whisper model directory is required",
            "pass --model-dir or set MOODIO_LYRICS_MODEL_DIR; no model is downloaded automatically",
        )
    model_path = Path(local_dir).expanduser().resolve()
    if not model_path.is_dir() or not (model_path / "model.bin").is_file():
        raise MusicEmotionError(
            "LYRICS_MODEL_NOT_FOUND",
            f"no local faster-whisper model.bin found in {model_path}",
            "provide the directory containing a converted faster-whisper model.bin",
        )
    if device not in ("cpu", "cuda"):
        raise MusicEmotionError("INVALID_ARGUMENT", f"unsupported device: {device}", "use cpu or cuda")

    # 本接口只接受已存在的本地模型；阻止后端依赖尝试隐式联网。
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise MusicEmotionError(
            "LYRICS_DEPENDENCY_MISSING",
            "faster-whisper is not installed",
            "install the optional lyrics dependency: python -m pip install faster-whisper",
        ) from exc

    try:
        model = WhisperModel(
            str(model_path), device=device, compute_type="int8" if device == "cpu" else "float16",
        )
        # 歌声并非普通讲话；默认不启用 VAD，以免切掉持续演唱。
        raw_segments, info = model.transcribe(
            str(audio), language=language, beam_size=5, vad_filter=False,
            condition_on_previous_text=False,
        )
        segments = []
        for item in raw_segments:
            text = item.text.strip()
            if text:
                segments.append({
                    "start_sec": round(float(item.start), 2),
                    "end_sec": round(float(item.end), 2),
                    "text": text,
                })
    except Exception as exc:  # noqa: BLE001 - 后端/音频解码异常转为结构化错误
        raise MusicEmotionError(
            "LYRICS_TRANSCRIPTION_FAILED",
            f"faster-whisper failed: {type(exc).__name__}: {exc}",
            "verify the model files and audio format, or try --device cpu",
        ) from exc

    probability = getattr(info, "language_probability", None)
    warnings = ["ASR on a full mix can mishear sung words or invent text during instrumentals; verify against audio."]
    if not segments:
        warnings.append("No vocal text was recognized; this does not prove the track has no lyrics.")
    return {
        "engine": "faster-whisper",
        "model_dir": str(model_path),
        "source": "mixed_audio",
        "language": getattr(info, "language", None),
        "language_probability": round(float(probability), 3) if probability is not None else None,
        "text": "\n".join(segment["text"] for segment in segments),
        "segments": segments,
        "warnings": warnings,
    }
