# -*- coding: utf-8 -*-
"""十二轴扩展契约：补充声学证据，不替换现有四轴分类。"""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TwelveAxesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sr = 22050
        t = np.arange(sr * 12, dtype=np.float32) / sr
        pulse = 0.6 + 0.4 * (np.sin(2 * np.pi * 2 * t) > 0)
        cls.audio = (0.05 * pulse * (np.sin(2 * np.pi * 220 * t) +
                                    0.5 * np.sin(2 * np.pi * 330 * t))).astype(np.float32)
        cls.sr = sr

    def test_twelve_axes_are_bounded_and_keep_raw_units(self) -> None:
        from music_emotion.axes_v2 import AXES, DEFAULT_RANGES, extract, score

        features, segments = extract(self.audio)
        axes = score(features, segments, DEFAULT_RANGES)
        self.assertEqual(set(axes), set(AXES))
        self.assertEqual(len(axes), 12)
        self.assertTrue(all(0.0 <= entry["value"] <= 1.0 for entry in axes.values()))
        self.assertEqual(axes["Tempo"]["unit"], "BPM")
        self.assertEqual(axes["Density"]["unit"], "onsets/s")

    def test_opt_in_does_not_change_existing_four_axis_result(self) -> None:
        from music_emotion.pipeline import analyze_waveform
        from music_emotion.types import ClipInfo

        clip = ClipInfo("synthetic.wav", 12.0, self.sr, 1, "WAV", "PCM_16")
        baseline = analyze_waveform(self.audio, self.sr, clip, segment_sec=6, hop_sec=6)
        expanded = analyze_waveform(self.audio, self.sr, clip, segment_sec=6, hop_sec=6, with_axes_v2=True)
        plain = baseline.to_dict(include_segments=False)
        extra = expanded.to_dict(include_segments=False)
        self.assertNotIn("axes_v2", plain)
        self.assertEqual(plain["emotion"]["valence"], extra["emotion"]["valence"])
        self.assertEqual(plain["emotion"]["arousal"], extra["emotion"]["arousal"])
        self.assertEqual(plain["emotion_4d"]["emotion_id"], extra["emotion_4d"]["emotion_id"])
        self.assertEqual(len(extra["axes_v2"]["axes"]), 12)
        self.assertEqual(extra["axes_v2"]["seconds_analyzed"], 12.0)
        self.assertEqual(extra["axes_v2"]["sampling"], "full_analyzed_audio")


    def test_full_audio_over_sixty_seconds_is_not_truncated(self) -> None:
        from unittest.mock import patch
        from music_emotion.pipeline import _analyze_axes_v2

        audio = np.ones(self.sr * 65, dtype=np.float32)
        with patch("music_emotion.axes_v2.extract", return_value=({}, {})) as extractor:
            with patch("music_emotion.axes_v2.score", return_value={"Energy": {"value": 0.5}}):
                result = _analyze_axes_v2(audio, self.sr)
        self.assertEqual(extractor.call_args.args[0].size, audio.size)
        self.assertEqual(result["seconds_analyzed"], 65.0)
        self.assertEqual(result["sampling"], "full_analyzed_audio")

    def test_standalone_reader_defaults_to_entire_audio(self) -> None:
        import tempfile
        from pathlib import Path
        from scipy.io import wavfile
        from unittest.mock import patch
        from music_emotion.axes_v2 import SR, analyze, load_audio

        with tempfile.TemporaryDirectory() as td:
            audio_file = Path(td, "whole.wav")
            wavfile.write(str(audio_file), SR, np.ones(SR * 4, dtype=np.int16))
            self.assertEqual(len(load_audio(str(audio_file), seconds=0.0)), SR * 4)
            self.assertEqual(len(load_audio(str(audio_file), seconds=2.0)), SR * 2)
        with patch("music_emotion.axes_v2.load_audio", return_value=self.audio) as reader:
            with patch("music_emotion.axes_v2.extract", return_value=({}, {})):
                with patch("music_emotion.axes_v2.score", return_value={}):
                    analyze("fake.wav")
        self.assertEqual(reader.call_args.args[2], 0.0)

    def test_ffmpeg_full_read_has_no_zero_duration_flag(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import patch
        from music_emotion.axes_v2 import _load_ffmpeg

        fake = SimpleNamespace(stdout=np.ones(64, dtype=np.float32).tobytes())
        with patch("music_emotion.axes_v2.subprocess.run", return_value=fake) as process:
            audio = _load_ffmpeg("sample.m4a", self.sr, seconds=0.0)
        self.assertEqual(len(audio), 64)
        self.assertNotIn("-t", process.call_args.args[0])
        self.assertNotIn("-ss", process.call_args.args[0])

    def test_cli_opt_in_selects_axes_v2_without_replacing_emotion(self) -> None:
        import io
        import json
        from contextlib import redirect_stderr, redirect_stdout
        from types import SimpleNamespace
        from unittest.mock import patch
        from moodio import main

        payload = {"emotion": {"valence": 0.1, "arousal": 0.2},
                   "axes_v2": {"axes": {"Energy": {"value": 0.7}}}}
        fake = SimpleNamespace(to_dict=lambda **_kwargs: payload)
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch("music_emotion.pipeline.analyze_file", return_value=fake) as analyzer:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = main(["analyze", "--path", __file__, "--axes-v2",
                             "--fields", "emotion,axes_v2", "--no-segments"])
        self.assertEqual(code, 0, stderr.getvalue())
        self.assertTrue(analyzer.call_args.kwargs["with_axes_v2"])
        self.assertIn("axes_v2", json.loads(stdout.getvalue()))


    def test_batch_record_adds_axes_only_when_requested(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import patch
        from music_emotion.batch import analyze_one

        payload = {
            "emotion": {"emotion": "强烈", "valence": 0.1, "arousal": 0.2},
            "features": {}, "quality": {"confidence": 0.7, "n_segments": 2},
            "clip": {"duration_sec": 12.0}, "arc": "wave",
            "axes_v2": {"axes": {"Energy": {"value": 0.7}}},
        }
        report = SimpleNamespace(to_dict=lambda **_kwargs: payload)
        with patch("music_emotion.pipeline.analyze_file", return_value=report) as analyzer:
            result = analyze_one(__file__, os.path.dirname(__file__), with_axes_v2=True)
        self.assertIsNone(result["error"])
        self.assertEqual(result["axes_v2"]["axes"]["Energy"]["value"], 0.7)
        self.assertTrue(analyzer.call_args.kwargs["with_axes_v2"])


if __name__ == "__main__":
    unittest.main()
