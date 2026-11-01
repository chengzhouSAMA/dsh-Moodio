# -*- coding: utf-8 -*-
"""乐器目录、未知类拒判、时间覆盖与联合解释契约；不把模拟分数当真实准确率。"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class InstrumentTests(unittest.TestCase):
    def test_catalog_covers_all_requested_instruments(self):
        from music_emotion.instrument_catalog import catalog

        names = {item["name"] for item in catalog()}
        self.assertEqual(names, set("古筝 二胡 琵琶 笛子 葫芦丝 唢呐 扬琴 笙 陶笛 古琴 钢琴 吉他 小提琴 萨克斯 架子鼓 大提琴 长笛 贝斯 尤克里里 口琴 手风琴".split()))
        self.assertEqual(len({item["id"] for item in catalog()}), 21)

    def test_each_label_has_correct_score_alignment(self):
        from music_emotion.instrument_catalog import INSTRUMENTS, OTHER_SOUNDS
        from music_emotion.instruments import classify_similarities

        for index, (identifier, _, _, _) in enumerate(INSTRUMENTS):
            scores = np.full(len(INSTRUMENTS) + len(OTHER_SOUNDS), 0.1)
            scores[index] = 0.6
            result = classify_similarities(scores)
            self.assertEqual([c["id"] for c in result["candidates"]], [identifier])

    def test_unknown_and_similar_timbres_are_not_forced_to_one_class(self):
        from music_emotion.instrument_catalog import INSTRUMENTS, OTHER_SOUNDS
        from music_emotion.instruments import classify_similarities

        scores = np.full(len(INSTRUMENTS) + len(OTHER_SOUNDS), 0.1)
        scores[-1] = 0.65
        self.assertEqual(classify_similarities(scores)["status"], "unknown")
        scores[-1] = 0.1
        scores[0], scores[9] = 0.41, 0.40
        ambiguous = classify_similarities(scores)
        self.assertEqual(ambiguous["status"], "ambiguous")
        self.assertEqual(ambiguous["candidates"][0]["confusable_with"], ["guqin"])
        with self.assertRaises(ValueError):
            classify_similarities(np.array([np.nan]))

    def test_secondary_instrument_requires_direct_classifier_support(self):
        from music_emotion.instrument_catalog import INSTRUMENTS, OTHER_SOUNDS
        from music_emotion.instruments import fuse_instrument_scores

        scores = np.full(len(INSTRUMENTS) + len(OTHER_SOUNDS), 0.1)
        scores[10], scores[11] = 0.55, 0.45
        solo = fuse_instrument_scores(scores, {"Piano": 0.9, "Guitar": 0.02})
        self.assertEqual([c["id"] for c in solo["candidates"]], ["piano"])
        together = fuse_instrument_scores(scores, {"Piano": 0.9, "Guitar": 0.8})
        self.assertEqual([c["id"] for c in together["candidates"]], ["piano", "guitar"])
        self.assertEqual(together["candidates"][1]["verification"]["scope"], "direct_label")

    def test_family_support_does_not_invent_two_specific_instruments(self):
        from music_emotion.instrument_catalog import INSTRUMENTS, OTHER_SOUNDS
        from music_emotion.instruments import fuse_instrument_scores

        scores = np.full(len(INSTRUMENTS) + len(OTHER_SOUNDS), 0.1)
        scores[3], scores[16] = 0.55, 0.45
        result = fuse_instrument_scores(scores, {"Flute": 0.9})
        self.assertEqual([c["id"] for c in result["candidates"]], ["dizi"])
        self.assertEqual(result["candidates"][0]["verification"]["scope"], "family_only")
        scores[:] = 0.1
        scores[6] = 0.55
        unsupported = fuse_instrument_scores(scores, {"Piano": 0.01})
        self.assertEqual(unsupported["candidates"][0]["verification"]["scope"], "unsupported")
        self.assertIsNone(unsupported["candidates"][0]["verification"]["tag_score"])
        with self.assertRaises(ValueError):
            fuse_instrument_scores(scores, {"Flute": float("nan")})

    def test_missing_model_is_explicit_and_never_downloads(self):
        from music_emotion.instruments import analyze_instruments

        with tempfile.TemporaryDirectory() as tmp:
            with patch("music_emotion.instruments._recognizer") as loader:
                result = analyze_instruments(np.ones(1000, dtype=np.float32), 1000, tmp)
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["reason"], "model_missing")
        loader.assert_not_called()

    def test_whole_song_and_tail_are_covered_without_sampling_gaps(self):
        from music_emotion.instrument_catalog import INSTRUMENTS, OTHER_SOUNDS
        from music_emotion.instruments import analyze_instruments, classify_similarities

        scores = np.full(len(INSTRUMENTS) + len(OTHER_SOUNDS), 0.1)
        scores[10] = 0.6
        engine = SimpleNamespace(device="cpu", recognize=lambda _: classify_similarities(scores))
        with patch("music_emotion.instruments.model_status", return_value={"available": True, "model_dir": "local"}):
            with patch("music_emotion.instruments._recognizer", return_value=engine):
                result = analyze_instruments(np.full(48000 * 23, 0.1, dtype=np.float32), 48000)
        self.assertEqual([(s["start_sec"], s["end_sec"]) for s in result["segments"]], [(0, 10), (10, 20), (20, 23)])
        self.assertEqual(result["seconds_analyzed"], 23)
        self.assertEqual(result["candidates"][0]["supported_ratio"], 1.0)
        self.assertEqual(result["score_type"], "cosine_similarity_not_probability")

    def test_fusion_limits_low_melody_coverage_and_unavailable_models(self):
        from music_emotion.assessment import build_music_analysis

        result = build_music_analysis({
            "emotion_4d": {"emotion": "平静"},
            "instruments": {"status": "unavailable", "candidates": []},
            "melody": {"status": "candidate", "agent_summary": {"overview": {"note_coverage": 0.02}}, "motifs": []},
            "axes_v3": {"axes": {"Brightness": {"value": 0.04, "status": "ok", "zh": "明亮"}}},
        })
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["primary_acoustic_emotion"], "平静")
        self.assertTrue(any("覆盖较低" in item for item in result["constraints"]))
        self.assertEqual(result["context"]["status"], "not_searched")

    def test_disagreement_is_reported_per_window_without_erasing_support(self):
        from music_emotion.instrument_catalog import INSTRUMENTS, OTHER_SOUNDS
        from music_emotion.instruments import analyze_instruments, fuse_instrument_scores

        scores = np.full(len(INSTRUMENTS) + len(OTHER_SOUNDS), 0.1)
        scores[10] = 0.6
        decisions = iter([fuse_instrument_scores(scores, {"Piano": 0.9}),
                          fuse_instrument_scores(scores, {"Piano": 0.05})])
        engine = SimpleNamespace(device="cpu", recognize=lambda _: next(decisions))
        with patch("music_emotion.instruments.model_status", return_value={"available": True, "model_dir": "local"}):
            with patch("music_emotion.instruments._recognizer", return_value=engine):
                result = analyze_instruments(np.full(48000 * 20, 0.1, dtype=np.float32), 48000)
        candidate = result["candidates"][0]
        self.assertEqual(candidate["status"], "mixed_evidence")
        self.assertEqual(candidate["corroborated_ratio"], 0.5)
        self.assertEqual(candidate["uncorroborated_seconds"], 10)

    def test_pipeline_and_batch_preserve_instrument_evidence(self):
        from music_emotion.batch import analyze_one
        from music_emotion.pipeline import analyze_waveform
        from music_emotion.types import ClipInfo

        sr = 22050
        y = (0.1 * np.sin(2 * np.pi * 440 * np.arange(sr * 4) / sr)).astype(np.float32)
        clip = ClipInfo("tone.wav", 4.0, sr, 1, "WAV", "PCM_16")
        instrument_result = {"status": "candidate", "candidates": [{"id": "piano", "name": "钢琴", "status": "candidate"}]}
        with patch("music_emotion.instruments.analyze_instruments", return_value=instrument_result):
            report = analyze_waveform(y, sr, clip, with_instruments=True, with_axes_v3=True)
        self.assertEqual(report.to_dict()["instruments"], instrument_result)
        self.assertIn("music_analysis", report.to_dict())
        with patch("music_emotion.pipeline.analyze_file", return_value=report) as analyzer:
            rec = analyze_one(__file__, os.path.dirname(__file__), with_instruments=True)
        self.assertIsNone(rec["error"])
        self.assertTrue(analyzer.call_args.kwargs["with_instruments"])
        self.assertEqual(rec["music_analysis"]["instruments"]["candidates"][0]["name"], "钢琴")


if __name__ == "__main__":
    unittest.main()
