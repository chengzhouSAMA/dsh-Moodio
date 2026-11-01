# -*- coding: utf-8 -*-
"""旋律三层的时间证据、重复识别与兼容性测试。"""
from __future__ import annotations

import os
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MelodyTests(unittest.TestCase):
    @staticmethod
    def notes(pitches, start=0.0):
        return [{"start_sec": round(start + i * 0.25, 3),
                 "end_sec": round(start + (i + 1) * 0.25, 3),
                 "pitch_midi": p, "voicing_probability": 0.9}
                for i, p in enumerate(pitches)]

    def test_transposed_repetition_has_timestamped_evidence(self):
        from music_emotion.melody import describe_notes

        notes = self.notes([60, 62, 64, 67]) + self.notes([72, 74, 76, 79], 1.6)
        result = describe_notes(notes)
        self.assertEqual(len(result["phrases"]), 2)
        self.assertEqual(result["phrases"][0]["contour"], "ascending")
        self.assertEqual(result["phrases"][1]["motif_id"], "motif_1")
        self.assertEqual(len(result["motifs"][0]["occurrences"]), 2)
        self.assertIn("1.6", result["agent_summary"]["text"])
        self.assertEqual(result["agent_summary"]["evidence"][-1]["type"], "repetition")

    def test_unrelated_melodies_do_not_repeat(self):
        from music_emotion.melody import describe_notes

        notes = self.notes([60, 62, 64, 67]) + self.notes([72, 71, 69, 72], 1.6)
        result = describe_notes(notes)
        self.assertEqual(result["motifs"], [])
        self.assertTrue(all(phrase["motif_id"] is None for phrase in result["phrases"]))

    def test_semitones_and_multiple_turns_have_correct_contours(self):
        from music_emotion.melody import describe_notes

        for pitches, expected in [([60, 61, 62, 63], "ascending"),
                                  ([63, 62, 61, 60], "descending"),
                                  ([60, 62, 64, 62, 60], "arch"),
                                  ([60, 62, 60, 62, 60], "mixed")]:
            with self.subTest(pitches=pitches):
                result = describe_notes(self.notes(pitches))
                self.assertEqual(result["phrases"][0]["contour"], expected)

    def test_unvoiced_frame_keeps_two_notes_separate(self):
        from music_emotion.melody import _notes_from_frames

        f0 = np.full(17, 440.0)
        voiced = np.ones(17, dtype=bool)
        probability = np.full(17, 0.9)
        f0[8], voiced[8], probability[8] = np.nan, False, 0.0
        notes = _notes_from_frames(f0, voiced, probability, 1.0, 11025, 256)
        self.assertEqual(len(notes), 2)
        self.assertLess(notes[0]["end_sec"], notes[1]["start_sec"])

    def test_near_matches_do_not_form_a_false_transitive_motif(self):
        from music_emotion.melody import describe_notes

        notes = (self.notes([60, 62, 64, 65, 67]) +
                 self.notes([60, 62, 65, 66, 68], 3.0) +
                 self.notes([60, 62, 66, 67, 69], 6.0))
        result = describe_notes(notes)
        self.assertEqual(len(result["motifs"]), 1)
        self.assertEqual(len(result["motifs"][0]["occurrences"]), 2)
        self.assertIsNone(result["phrases"][2]["motif_id"])

    def test_boundary_artifacts_are_rejected_and_counted(self):
        from music_emotion.melody import analyze_melody

        pitches = np.repeat([48, 60, 62, 64, 67, 48, 69], 12)
        f0 = 440.0 * 2 ** ((pitches - 69) / 12)
        expanded = f0.copy()
        expanded[pitches == 48] = 440.0 * 2 ** ((45 - 69) / 12)
        voiced = np.ones(len(f0), dtype=bool)
        probability = np.full(len(f0), 0.95)
        audio = np.zeros((len(f0) - 1) * 256, dtype=np.float32)
        with patch("music_emotion.melody._track_pitch", side_effect=[
            (f0, voiced, probability), (expanded, voiced, probability),
        ]) as tracker:
            result = analyze_melody(audio, 11025)
        self.assertEqual(tracker.call_count, 2)
        self.assertEqual(tracker.call_args.args[1:], (36, 96))
        quality = result["quality"]["boundary_validation"]
        self.assertEqual(quality["initial_note_count"], 7)
        self.assertEqual(quality["initial_lower_bound_note_count"], 2)
        self.assertAlmostEqual(quality["initial_lower_bound_note_fraction"], 2 / 7, places=4)
        self.assertEqual(quality["rejected_boundary_frames"], 24)
        self.assertEqual(quality["retained_lower_bound_note_count"], 0)
        self.assertNotIn(48, [n["pitch_midi"] for n in result["notes"]])
        self.assertEqual(result["status"], "candidate")

    def test_true_boundary_pitch_survives_wider_range_verification(self):
        from music_emotion.melody import analyze_melody

        sr = 11025
        t = np.arange(round(sr * 0.4)) / sr
        audio = np.concatenate([0.18 * np.sin(2 * np.pi * 440 * 2 ** ((p - 69) / 12) * t)
                                for p in [48, 52, 55, 60]]).astype(np.float32)
        result = analyze_melody(audio, sr)
        self.assertEqual(result["status"], "candidate")
        self.assertIn(48, [n["pitch_midi"] for n in result["notes"]])
        self.assertGreater(result["quality"]["boundary_validation"]["confirmed_boundary_frames"], 0)

    def test_second_pass_boundary_or_unvoiced_pitch_is_not_confirmation(self):
        from music_emotion.melody import _validate_boundaries

        original = np.full(30, 440 * 2 ** ((48 - 69) / 12))
        voiced = np.ones(30, dtype=bool)
        probability = np.full(30, 0.9)
        for pitch, is_voiced in [(36, True), (48, False), (96, True)]:
            with self.subTest(pitch=pitch, voiced=is_voiced):
                check = np.full(30, 440 * 2 ** ((pitch - 69) / 12))
                with patch("music_emotion.melody._track_pitch", return_value=(check, np.full(30, is_voiced), probability)):
                    f0, flags, _, quality = _validate_boundaries(np.zeros(29 * 256), original, voiced, probability)
                self.assertEqual(quality["rejected_boundary_frames"], 30)
                self.assertTrue(np.all(np.isnan(f0)))
                self.assertFalse(np.any(flags))

    def test_upper_boundary_is_verified_against_an_expanded_ceiling(self):
        from music_emotion.melody import _validate_boundaries

        original = np.full(30, 440 * 2 ** ((96 - 69) / 12))
        voiced = np.ones(30, dtype=bool)
        probability = np.full(30, 0.9)
        with patch("music_emotion.melody._track_pitch", return_value=(original.copy(), voiced, probability)) as tracker:
            _, flags, _, quality = _validate_boundaries(np.zeros(29 * 256), original, voiced, probability)
        self.assertEqual(tracker.call_args.args[1:], (48, 108))
        self.assertEqual(quality["confirmed_boundary_frames"], 30)
        self.assertTrue(np.all(flags))

    def test_summary_covers_full_track_and_late_evidence(self):
        from music_emotion.melody import describe_notes

        notes = (self.notes([60, 62, 64, 67], 0.7) + self.notes([67, 65, 64, 62], 70) +
                 self.notes([60, 64, 67, 69], 160))
        summary = describe_notes(notes, duration=169)["agent_summary"]
        self.assertEqual(summary["overview"]["duration_sec"], 169)
        self.assertEqual(summary["overview"]["detected_note_seconds"], 3.0)
        self.assertAlmostEqual(summary["overview"]["note_coverage"], 3 / 169, places=4)
        self.assertEqual(len(summary["sections"]), 6)
        self.assertEqual(summary["sections"][-1]["end_sec"], 169)
        self.assertTrue(any(e["type"] == "contour" and e["start_sec"] >= 160 for e in summary["evidence"]))
        self.assertIn("169.0", summary["text"])
        self.assertIn("实际覆盖 3.0 秒", summary["text"])

    def test_summary_does_not_extrapolate_intro_to_empty_sections(self):
        from music_emotion.melody import describe_notes

        result = describe_notes(self.notes([60, 62, 64, 67], 0.7), duration=169)
        self.assertTrue(all(section["note_count"] == 0 and section["contour"] is None
                            for section in result["agent_summary"]["sections"][1:]))
        self.assertEqual(result["motif_search"]["status"], "insufficient_notes")

    def test_motif_inside_unequal_phrases_can_cross_phrase_boundary(self):
        from music_emotion.melody import describe_notes

        notes = (self.notes([50, 53, 55, 57, 59, 60, 62, 64, 67, 65]) +
                 self.notes([81, 72, 74, 76, 79, 83], 80))
        result = describe_notes(notes, duration=169)
        match = next(m for m in result["motifs"]
                     if {1.25, 80.25}.issubset({o["start_sec"] for o in m["occurrences"]}))
        self.assertEqual(match["note_count"], 4)
        self.assertEqual(len(match["occurrences"][0]["phrase_ids"]), 2)
        self.assertEqual(result["motif_search"]["status"], "detected")

    def test_same_intervals_with_incompatible_rhythm_are_not_a_motif(self):
        from music_emotion.melody import describe_notes

        other = [{"start_sec": t, "end_sec": t + 0.08, "pitch_midi": p}
                 for t, p in zip([10, 10.1, 10.8, 10.9], [72, 74, 76, 79])]
        result = describe_notes(self.notes([60, 62, 64, 67]) + other)
        self.assertEqual(result["motifs"], [])
        self.assertEqual(result["motif_search"]["status"], "not_detected")
        self.assertEqual(result["motif_search"]["reason"], "relative_rhythm_mismatch")

    def test_continuous_repetition_does_not_duplicate_shifted_windows(self):
        from music_emotion.melody import describe_notes

        result = describe_notes(self.notes([60, 62, 65, 64, 67, 65, 62, 60] * 4))
        self.assertEqual(len(result["motifs"]), 1)
        self.assertEqual(len(result["motifs"][0]["occurrences"]), 4)

    def test_tone_sequence_is_timestamped_and_short_clips_abstain(self):
        from music_emotion.melody import analyze_melody

        sr = 11025
        parts = []
        for pitch in [60, 62, 64, 67]:
            t = np.arange(round(sr * 0.28)) / sr
            parts.append((0.18 * np.sin(2 * np.pi * 440 * 2 ** ((pitch - 69) / 12) * t)).astype(np.float32))
        parts.append(np.zeros(round(sr * 0.55), dtype=np.float32))
        for pitch in [72, 74, 76, 79]:
            t = np.arange(round(sr * 0.28)) / sr
            parts.append((0.18 * np.sin(2 * np.pi * 440 * 2 ** ((pitch - 69) / 12) * t)).astype(np.float32))
        audio = np.concatenate(parts)
        result = analyze_melody(audio, sr)
        self.assertEqual(result["status"], "candidate")
        self.assertEqual([n["pitch_midi"] for n in result["notes"]], [60, 62, 64, 67, 72, 74, 76, 79])
        self.assertEqual(len(result["phrases"]), 2)
        self.assertEqual(len(result["motifs"]), 1)
        self.assertGreaterEqual(result["notes"][0]["voicing_probability"], 0.5)
        self.assertTrue(all(0 <= n["start_sec"] < n["end_sec"] <= len(audio) / sr
                            for n in result["notes"]))
        self.assertEqual(result["source_assumption"], "monophonic_or_dominant_lead")
        short = analyze_melody(audio[:sr // 4], sr)
        self.assertEqual(short["status"], "insufficient_evidence")
        self.assertEqual(short["agent_summary"]["evidence"], [])

    def test_python_api_melody_option_does_not_change_emotion(self):
        from music_emotion.pipeline import analyze_waveform
        from music_emotion.types import ClipInfo

        sr = 22050
        t = np.arange(sr * 4, dtype=np.float32) / sr
        y = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        clip = ClipInfo("tone.wav", 4.0, sr, 1, "WAV", "PCM_16")
        baseline = analyze_waveform(y, sr, clip, segment_sec=4, hop_sec=4).to_dict()
        with patch("music_emotion.melody.analyze_melody", return_value={
            "status": "insufficient_evidence", "reason": "too_few_stable_notes",
            "notes": [], "phrases": [], "motifs": [],
            "agent_summary": {"text": "证据不足", "evidence": []},
        }):
            expanded = analyze_waveform(y, sr, clip, segment_sec=4, hop_sec=4,
                                        with_melody=True).to_dict()
        self.assertNotIn("melody", baseline)
        self.assertEqual(baseline["emotion_4d"], expanded["emotion_4d"])
        self.assertEqual(expanded["melody"]["status"], "insufficient_evidence")
        self.assertTrue(any("melody" in warning for warning in expanded["warnings"]))

    def test_cli_fields_return_melody_without_external_models(self):
        import soundfile as sf

        sr = 11025
        t = np.arange(sr * 2, dtype=np.float32) / sr
        audio = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tone.wav")
            sf.write(path, audio, sr)
            script = os.path.join(os.path.dirname(os.path.dirname(__file__)), "moodio.py")
            run = subprocess.run(
                [sys.executable, "-X", "utf8", script, "analyze", "--path", path,
                 "--fields", "melody", "--no-segments", "--no-instruments", "--no-axes-v3"],
                capture_output=True, text=True, check=False,
            )
        self.assertEqual(run.returncode, 0, run.stderr)
        payload = json.loads(run.stdout)
        self.assertEqual(payload["melody"]["status"], "insufficient_evidence")
        self.assertEqual(payload["melody"]["agent_summary"]["evidence"], [])

    def test_cli_defaults_to_full_song_melody_and_allows_explicit_skip(self):
        from moodio import build_parser

        for command in ("analyze", "batch"):
            with self.subTest(command=command):
                default = build_parser().parse_args([command, "--path", "music"])
                self.assertTrue(default.melody)
                self.assertTrue(default.instruments)
                self.assertTrue(default.axes_v3)
                self.assertEqual(default.max_seconds, 0.0)
                skipped = build_parser().parse_args([command, "--path", "music", "--no-melody"])
                self.assertFalse(skipped.melody)


if __name__ == "__main__":
    unittest.main()
