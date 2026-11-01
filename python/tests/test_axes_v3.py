# -*- coding: utf-8 -*-
"""十八轴契约：新增声学证据、不可靠读数与旧版兼容。"""
from __future__ import annotations

import os
import sys
import unittest
import io
import json
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class EighteenAxesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sr = 22050
        cls.sr = sr
        t = np.arange(sr * 12, dtype=np.float32) / sr
        cls.tone = (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        cls.clicks = np.zeros(len(t), dtype=np.float32)
        rng = np.random.default_rng(4)
        for i in range(0, len(t) - 700, sr // 2):
            cls.clicks[i:i + 700] += 0.5 * rng.standard_normal(700) * np.hanning(700)

    def test_eighteen_axes_keep_legacy_twelve_and_four_axis_emotion(self) -> None:
        from music_emotion.axes_v2 import AXES, AXES_V3
        from music_emotion.pipeline import analyze_waveform
        from music_emotion.types import ClipInfo
        from unittest.mock import patch

        clip = ClipInfo("pulse.wav", 12.0, self.sr, 1, "WAV", "PCM_16")
        original = analyze_waveform(self.clicks, self.sr, clip, segment_sec=6, hop_sec=6,
                                    with_axes_v2=True)
        from music_emotion.axes_v2 import extract
        with patch("music_emotion.axes_v2.extract", wraps=extract) as extractor:
            expanded = analyze_waveform(self.clicks, self.sr, clip, segment_sec=6, hop_sec=6,
                                        with_axes_v2=True, with_axes_v3=True)
        payload = expanded.to_dict(include_segments=False)
        self.assertEqual(extractor.call_count, 1)
        self.assertEqual(original.to_dict(include_segments=False)["emotion_4d"], payload["emotion_4d"])
        self.assertEqual(original.to_dict(include_segments=False)["axes_v2"]["axes"], payload["axes_v2"]["axes"])
        self.assertEqual(set(payload["axes_v2"]["axes"]), set(AXES))
        self.assertEqual(set(payload["axes_v3"]["axes"]), set(AXES_V3))
        self.assertEqual(payload["axes_v3"]["axis_count"], 18)

    def test_pulse_and_no_pulse_are_not_confused_with_low_scores(self) -> None:
        from music_emotion.axes_v2 import DEFAULT_RANGES, extract, score

        rhythmic = score(*extract(self.clicks, extended=True), DEFAULT_RANGES, extended=True)
        self.assertEqual(rhythmic["Tempo"]["status"], "ok")
        self.assertEqual(rhythmic["TempoStability"]["status"], "ok")
        self.assertGreater(rhythmic["TempoStability"]["value"], 0.7)
        self.assertEqual(rhythmic["Tempo"]["unit"], "BPM")

        unpulsed = score(*extract(self.tone, extended=True), DEFAULT_RANGES, extended=True)
        for axis in ("Tempo", "RhythmicRegularity", "TempoStability"):
            self.assertIsNone(unpulsed[axis]["value"])
            self.assertEqual(unpulsed[axis]["reason"], "weak_or_missing_pulse")
        self.assertNotIn("raw", unpulsed["Tempo"])

    def test_new_axes_react_to_distinct_acoustic_cues(self) -> None:
        from music_emotion.axes_v2 import DEFAULT_RANGES, extract, score

        t = np.arange(len(self.tone)) / self.sr
        bright = 0.1 * np.sin(2 * np.pi * 1100 * t)
        bass = 0.1 * np.sin(2 * np.pi * 110 * t)
        dark = score(*extract(bass, extended=True), DEFAULT_RANGES, extended=True)
        high = score(*extract(bright, extended=True), DEFAULT_RANGES, extended=True)
        self.assertGreater(dark["BassPresence"]["value"], high["BassPresence"]["value"])
        noise = np.random.default_rng(5).normal(0, 0.1, len(self.tone)).astype(np.float32)
        noisy = score(*extract(noise, extended=True), DEFAULT_RANGES, extended=True)
        self.assertGreater(noisy["Noisiness"]["value"], dark["Noisiness"]["value"])
        self.assertGreater(noisy["TransientSharpness"]["value"], dark["TransientSharpness"]["value"])
        self.assertIsNone(noisy["TonalStability"]["value"])
        self.assertEqual(noisy["TonalStability"]["reason"], "weak_or_missing_tonality")
        self.assertEqual(dark["HarmonicComplexity"]["measure"], "pitch_class_entropy; not chord complexity")

    def test_build_direction_reports_rising_and_falling(self) -> None:
        from music_emotion.axes_v2 import DEFAULT_RANGES, extract, score

        ramp = np.linspace(0.03, 0.5, len(self.tone), dtype=np.float32)
        rising = score(*extract(self.tone * ramp, extended=True), DEFAULT_RANGES, extended=True)
        falling = score(*extract((self.tone * ramp)[::-1], extended=True), DEFAULT_RANGES, extended=True)
        self.assertGreater(rising["BuildDirection"]["value"], 0.5)
        self.assertLess(falling["BuildDirection"]["value"], 0.5)

    def test_low_tonality_is_reported_as_unreliable(self) -> None:
        from music_emotion.axes_v2 import DEFAULT_RANGES, extract, score

        features, segments = extract(self.tone, extended=True)
        features["key_clarity"] = 0.0
        result = score(features, segments, DEFAULT_RANGES, extended=True)
        self.assertIsNone(result["TonalStability"]["value"])
        self.assertEqual(result["TonalStability"]["reason"], "weak_or_missing_tonality")

    def test_short_excerpt_has_no_structural_direction(self) -> None:
        from music_emotion.axes_v2 import DEFAULT_RANGES, extract, score

        sample = self.tone[:self.sr * 4]
        result = score(*extract(sample, extended=True), DEFAULT_RANGES, extended=True)
        self.assertIsNone(result["BuildDirection"]["value"])
        self.assertEqual(result["BuildDirection"]["reason"], "insufficient_segments")

    def test_cli_and_batch_expose_eighteen_axes_on_request(self) -> None:
        from moodio import main
        from music_emotion.batch import analyze_one

        axes = {"axis_count": 18, "axes": {"Tempo": {"value": None, "status": "unreliable"}}}
        payload = {"emotion": {"valence": 0.1}, "quality": {"confidence": 0.7, "n_segments": 2},
                   "clip": {"duration_sec": 12}, "features": {}, "axes_v3": axes}
        report = SimpleNamespace(to_dict=lambda **_kwargs: payload)
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch("music_emotion.pipeline.analyze_file", return_value=report) as analyzer:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = main(["analyze", "--path", __file__, "--axes-v3",
                             "--fields", "emotion,axes_v3", "--no-segments"])
            self.assertEqual(code, 0, stderr.getvalue())
            self.assertTrue(analyzer.call_args.kwargs["with_axes_v3"])
            self.assertFalse(analyzer.call_args.kwargs["with_axes_v2"])
            self.assertEqual(json.loads(stdout.getvalue())["axes_v3"], axes)

            record = analyze_one(__file__, os.path.dirname(__file__), with_axes_v3=True)
            self.assertIsNone(record["error"])
            self.assertEqual(record["axes_v3"], axes)
            self.assertTrue(analyzer.call_args.kwargs["with_axes_v3"])


if __name__ == "__main__":
    unittest.main()
