# -*- coding: utf-8 -*-
"""歌词转写的离线契约：不下载模型，输出真实转写片段。"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from music_emotion.types import MusicEmotionError  # noqa: E402


class LyricsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.audio = Path(self.temp.name, "song.wav")
        self.audio.write_bytes(b"fake audio; ASR is replaced in this test")
        self.model = Path(self.temp.name, "local-model")
        self.model.mkdir()
        (self.model / "model.bin").write_bytes(b"test model")

    def test_missing_model_stops_before_import_or_download(self) -> None:
        from music_emotion.lyrics import transcribe_lyrics

        with self.assertRaises(MusicEmotionError) as caught:
            transcribe_lyrics(str(self.audio), str(self.model / "missing"))
        self.assertEqual(caught.exception.code, "LYRICS_MODEL_NOT_FOUND")

    def test_offline_model_produces_timestamped_text(self) -> None:
        from music_emotion.lyrics import transcribe_lyrics

        observed = {}

        class FakeWhisperModel:
            def __init__(self, model_dir, **options):
                observed["model_dir"] = model_dir
                observed["options"] = options
                observed["offline"] = os.environ.get("HF_HUB_OFFLINE")

            def transcribe(self, source, **options):
                observed["source"] = source
                observed["transcribe_options"] = options
                segments = iter([
                    types.SimpleNamespace(start=0.2, end=2.4, text=" Hello world "),
                    types.SimpleNamespace(start=3.0, end=4.6, text=" Never give up "),
                ])
                info = types.SimpleNamespace(language="en", language_probability=0.96)
                return segments, info

        fake_lib = types.SimpleNamespace(WhisperModel=FakeWhisperModel)
        with patch.dict(os.environ, {"HF_HUB_OFFLINE": "0"}):
            with patch.dict(sys.modules, {"faster_whisper": fake_lib}):
                output = transcribe_lyrics(str(self.audio), str(self.model), language="en")

        self.assertEqual(observed["offline"], "1")

        self.assertEqual(observed["model_dir"], str(self.model.resolve()))
        self.assertEqual(observed["options"]["device"], "cpu")
        self.assertEqual(observed["transcribe_options"]["language"], "en")
        self.assertFalse(observed["transcribe_options"]["vad_filter"])
        self.assertEqual(output["text"], "Hello world\nNever give up")
        self.assertEqual(output["segments"][0], {"start_sec": 0.2, "end_sec": 2.4, "text": "Hello world"})
        self.assertEqual(output["source"], "mixed_audio")

    def test_missing_optional_dependency_is_actionable(self) -> None:
        from music_emotion.lyrics import transcribe_lyrics

        with patch.dict(sys.modules, {"faster_whisper": None}):
            with self.assertRaises(MusicEmotionError) as caught:
                transcribe_lyrics(str(self.audio), str(self.model))
        self.assertEqual(caught.exception.code, "LYRICS_DEPENDENCY_MISSING")

    def test_cli_exposes_lyrics_as_json(self) -> None:
        from moodio import main
        from music_emotion import lyrics

        payload = {"text": "Hello world", "segments": [], "language": "en"}
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(lyrics, "transcribe_lyrics", return_value=payload) as transcribe:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                exit_code = main(["lyrics", "--path", str(self.audio),
                                  "--model-dir", str(self.model), "--language", "en"])
        self.assertEqual(exit_code, 0, stderr.getvalue())
        self.assertEqual(json.loads(stdout.getvalue())["text"], "Hello world")
        self.assertEqual(transcribe.call_args.kwargs["model_dir"], str(self.model))


if __name__ == "__main__":
    unittest.main()
