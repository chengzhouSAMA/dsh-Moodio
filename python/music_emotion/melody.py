# -*- coding: utf-8 -*-
"""从单声部或突出声部提取候选旋律，保留可核对的时间证据。"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


MELODY_SR = 11025
HOP_LENGTH = 256
FRAME_LENGTH = 2048
PITCH_MIN_MIDI = 48
PITCH_MAX_MIDI = 96
VERIFY_MIN_MIDI = 36
VERIFY_MAX_MIDI = 108
BOUNDARY_MARGIN = 0.5
MIN_NOTE_SEC = 0.10
MIN_GAP_SEC = 0.35
MAX_PHRASE_NOTES = 8
MAX_PHRASE_SEC = 8.0


def _track_pitch(audio: np.ndarray, low: int, high: int):
    import librosa

    return librosa.pyin(
        audio, sr=MELODY_SR, fmin=librosa.midi_to_hz(low),
        fmax=librosa.midi_to_hz(high), frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
    )


def _validate_boundaries(audio: np.ndarray, f0: np.ndarray, voiced: np.ndarray,
                         probability: np.ndarray):
    """移开搜索边界做局部复核；一致性仅验证音高，不证明它属于主旋律。"""
    valid = voiced & np.isfinite(f0) & (f0 > 0) & (probability >= 0.5)
    midi = np.full(len(f0), np.nan)
    midi[valid] = 69 + 12 * np.log2(f0[valid] / 440.0)
    lower = valid & (midi <= PITCH_MIN_MIDI + BOUNDARY_MARGIN)
    upper = valid & (midi >= PITCH_MAX_MIDI - BOUNDARY_MARGIN)
    suspect = lower | upper
    verify_low = VERIFY_MIN_MIDI if np.any(lower) else PITCH_MIN_MIDI
    verify_high = VERIFY_MAX_MIDI if np.any(upper) else PITCH_MAX_MIDI
    confirmed = np.zeros(len(f0), dtype=bool)
    reviewed = 0
    # 只复核命中边界的十秒块；两侧上下文保持帧对齐，避免整首重复计算。
    block = max(1, int(10 * MELODY_SR / HOP_LENGTH))
    padding = int(np.ceil(0.4 * MELODY_SR / HOP_LENGTH))
    for lo in range(0, len(f0), block):
        hi = min(len(f0), lo + block)
        indices = np.flatnonzero(suspect[lo:hi]) + lo
        if not len(indices):
            continue
        start = max(0, lo - padding)
        stop = min(len(audio), (hi + padding) * HOP_LENGTH)
        check_f0, check_voiced, check_probability = _track_pitch(
            audio[start * HOP_LENGTH:stop], verify_low, verify_high,
        )
        reviewed += 1
        local = indices - start
        in_range = local < len(check_f0)
        indices, local = indices[in_range], local[in_range]
        check_valid = (check_voiced[local] & np.isfinite(check_f0[local]) &
                       (check_f0[local] > 0) & (check_probability[local] >= 0.5))
        candidate_midi = np.full(len(local), np.nan)
        candidate_midi[check_valid] = 69 + 12 * np.log2(check_f0[local[check_valid]] / 440.0)
        stable = (check_valid & (np.abs(candidate_midi - midi[indices]) <= BOUNDARY_MARGIN) &
                  (candidate_midi > verify_low + BOUNDARY_MARGIN) &
                  (candidate_midi < verify_high - BOUNDARY_MARGIN))
        confirmed[indices[stable]] = True

    rejected = suspect & ~confirmed
    cleaned_f0, cleaned_voiced, cleaned_probability = f0.copy(), voiced.copy(), probability.copy()
    cleaned_f0[rejected] = np.nan
    cleaned_voiced[rejected] = False
    cleaned_probability[rejected] = 0.0
    diagnostics = {
        "initial_pitch_bounds_midi": [PITCH_MIN_MIDI, PITCH_MAX_MIDI],
        "verification_pitch_bounds_midi": [verify_low, verify_high] if np.any(suspect) else None,
        "initial_voiced_frames": int(np.count_nonzero(valid)),
        "initial_lower_bound_frames": int(np.count_nonzero(lower)),
        "initial_upper_bound_frames": int(np.count_nonzero(upper)),
        "confirmed_boundary_frames": int(np.count_nonzero(confirmed)),
        "rejected_boundary_frames": int(np.count_nonzero(rejected)),
        "verification_blocks": reviewed,
        "status": "not_needed" if not np.any(suspect) else
                  ("rejected_inconsistent" if np.any(rejected) else "confirmed_across_ranges"),
    }
    return cleaned_f0, cleaned_voiced, cleaned_probability, diagnostics


def _contour(pitches: List[int]) -> str:
    steps = np.diff(pitches)
    directions = [int(np.sign(step)) for step in steps if step != 0]
    if not directions:
        return "level"
    if all(d > 0 for d in directions):
        return "ascending"
    if all(d < 0 for d in directions):
        return "descending"
    turns = [directions[0]]
    for direction in directions[1:]:
        if direction != turns[-1]:
            turns.append(direction)
    if turns == [1, -1]:
        return "arch"
    if turns == [-1, 1]:
        return "valley"
    return "mixed"


def _matches(left: List[int], right: List[int]) -> bool:
    """音程匹配忽略整体移调，但不把任意相似走向当成重复。"""
    if len(left) != len(right) or len(left) < 4:
        return False
    a, b = np.diff(left), np.diff(right)
    errors = np.abs(a - b)
    return bool(np.all(errors == 0) or (len(left) >= 5 and np.count_nonzero(errors) == 1
                                         and np.max(errors) <= 1))


def _notes_from_frames(f0: np.ndarray, voiced: np.ndarray, probability: np.ndarray,
                       duration: float, sr: int, hop_length: int) -> List[Dict[str, Any]]:
    midi = np.full(len(f0), -1, dtype=np.int32)
    usable = voiced & np.isfinite(f0) & (f0 > 0) & (probability >= 0.5)
    if np.any(usable):
        import librosa
        midi[usable] = np.rint(librosa.hz_to_midi(f0[usable])).astype(np.int32)

    # 仅修复夹在相同音符之间的单帧误检，不跨越真实休止符。
    for i in range(1, len(midi) - 1):
        if midi[i] >= 0 and midi[i - 1] >= 0 and midi[i - 1] == midi[i + 1] and midi[i] != midi[i - 1]:
            midi[i] = midi[i - 1]

    notes = []
    i = 0
    while i < len(midi):
        if midi[i] < 0:
            i += 1
            continue
        j = i + 1
        while j < len(midi) and midi[j] == midi[i]:
            j += 1
        start = max(0.0, (i - 0.5) * hop_length / sr)
        end = min(duration, (j - 0.5) * hop_length / sr)
        if end - start >= MIN_NOTE_SEC:
            notes.append({
                "start_sec": round(start, 3), "end_sec": round(end, 3),
                "pitch_midi": int(midi[i]),
                "voicing_probability": round(float(np.median(probability[i:j])), 3),
            })
        i = j
    return notes


def _rhythm_matches(left: List[Dict[str, Any]], right: List[Dict[str, Any]]) -> bool:
    """比较相对起音间隔，允许整体变速，避免只凭音高形状误报重复。"""
    a = np.diff([n["start_sec"] for n in left])
    b = np.diff([n["start_sec"] for n in right])
    if np.any(a <= 0) or np.any(b <= 0):
        return False
    error = np.abs(np.log2((a / np.median(a)) / (b / np.median(b))))
    return bool(np.mean(error) <= 0.25 and np.max(error) <= 0.55)


def _find_motifs(notes: List[Dict[str, Any]], phrases: List[Dict[str, Any]],
                 spans: List[tuple]) -> Dict[str, Any]:
    """按音程序列索引滑窗；匹配不依赖分句位置，重复实例必须互不重叠。"""
    candidates = []
    windows_examined = 0
    interval_candidates = 0
    rhythm_rejections = 0
    for size in range(min(12, len(notes) // 2), 3, -1):
        clusters: List[List[int]] = []
        by_signature: Dict[tuple, List[int]] = {}
        for start in range(len(notes) - size + 1):
            window = notes[start:start + size]
            if (window[-1]["end_sec"] - window[0]["start_sec"] > 20 or
                    any(b["start_sec"] - a["end_sec"] > 1.2 for a, b in zip(window, window[1:]))):
                continue
            pitches = [n["pitch_midi"] for n in window]
            intervals = tuple(int(v) for v in np.diff(pitches))
            if np.count_nonzero(intervals) < 2:
                continue
            windows_examined += 1
            keys = {intervals}
            if size >= 5:
                for i in range(len(intervals)):
                    for delta in (-1, 1):
                        key = list(intervals)
                        key[i] += delta
                        keys.add(tuple(key))
            possible = sorted({index for key in keys for index in by_signature.get(key, [])})
            for index in possible:
                cluster = clusters[index]
                if start < cluster[-1] + size:
                    continue
                if not all(_matches(pitches, [n["pitch_midi"] for n in notes[other:other + size]]) for other in cluster):
                    continue
                interval_candidates += 1
                if all(_rhythm_matches(window, notes[other:other + size]) for other in cluster):
                    cluster.append(start)
                    break
                rhythm_rejections += 1
            else:
                by_signature.setdefault(intervals, []).append(len(clusters))
                clusters.append([start])
        candidates.extend((size, cluster) for cluster in clusters if len(cluster) >= 2)

    candidates.sort(key=lambda item: (-item[0] * (len(item[1]) - 1), -item[0], item[1][0]))
    selected = []
    for size, starts in candidates:
        # 长模式已经解释的短窗口不再重复报告；避免同一次重现产生十几个动机。
        if any(all(sum(max(0, min(start + size, other + old_size) - max(start, other))
                       for other in old_starts) / size >= 0.75 for start in starts)
               for old_size, old_starts in selected):
            continue
        selected.append((size, starts))
        if len(selected) >= 12:
            break

    motifs = []
    for size, starts in selected:
        motif_id = f"motif_{len(motifs) + 1}"
        occurrences = []
        for start in starts:
            phrase_ids = [phrases[i]["id"] for i, (lo, hi) in enumerate(spans)
                          if lo < start + size and hi > start]
            for phrase in phrases:
                if phrase["id"] in phrase_ids and phrase["motif_id"] is None:
                    phrase["motif_id"] = motif_id
            occurrences.append({
                "phrase_id": phrase_ids[0] if len(phrase_ids) == 1 else None,
                "phrase_ids": phrase_ids,
                "start_sec": notes[start]["start_sec"], "end_sec": notes[start + size - 1]["end_sec"],
                "note_start_index": start, "note_end_index_exclusive": start + size,
                "transposition_semitones": notes[start]["pitch_midi"] - notes[starts[0]]["pitch_midi"],
            })
        motifs.append({"id": motif_id, "note_count": size,
                       "intervals_semitones": [int(v) for v in np.diff([n["pitch_midi"] for n in notes[starts[0]:starts[0] + size]])],
                       "occurrences": occurrences})
    return {"motifs": motifs, "motif_search": {
        "status": "detected" if motifs else ("insufficient_notes" if len(notes) < 8 else "not_detected"),
        "method": "sliding_interval_and_relative_rhythm",
        "windows_examined": windows_examined,
        "interval_candidates": interval_candidates,
        "rhythm_rejections": rhythm_rejections,
        "reason": None if motifs else (
            "too_few_notes_for_two_occurrences" if len(notes) < 8 else
            "no_contiguous_variable_pitch_windows" if not windows_examined else
            "no_nonoverlapping_interval_matches" if not interval_candidates else "relative_rhythm_mismatch"
        ),
    }}


def _summarize_track(notes: List[Dict[str, Any]], phrases: List[Dict[str, Any]],
                      motifs: List[Dict[str, Any]], duration: float) -> Dict[str, Any]:
    """扫描时长与音符证据覆盖率分开报告，按时间分段取证而非只读开头。"""
    seconds = sum(max(0.0, min(duration, n["end_sec"]) - max(0.0, n["start_sec"])) for n in notes)
    overview = {
        "duration_sec": round(duration, 3), "note_count": len(notes), "phrase_count": len(phrases),
        "detected_note_seconds": round(seconds, 3),
        "note_coverage": round(min(1.0, seconds / duration), 4) if duration > 0 else 0.0,
        "pitch_range_midi": [min(n["pitch_midi"] for n in notes), max(n["pitch_midi"] for n in notes)] if notes else None,
        "motif_count": len(motifs),
    }
    sections = []
    evidence = [{"type": "track_coverage", "start_sec": 0.0, "end_sec": round(duration, 3),
                 "detected_note_seconds": overview["detected_note_seconds"], "note_coverage": overview["note_coverage"]}]
    contours = {"level": "平稳", "ascending": "上行", "descending": "下行",
                "arch": "先升后降", "valley": "先降后升", "mixed": "起伏"}
    count = min(6, max(1, int(np.ceil(duration / 30))))
    for i in range(count):
        lo, hi = duration * i / count, duration * (i + 1) / count
        local = [n for n in notes if n["start_sec"] < hi and n["end_sec"] > lo]
        covered = sum(min(hi, n["end_sec"]) - max(lo, n["start_sec"]) for n in local)
        # 仅选完整落在本区间且至少有三个音符的乐句，边界外的音符不拿来补证据。
        eligible = [p for p in phrases if p["start_sec"] >= lo and p["end_sec"] <= hi and p["note_count"] >= 3]
        representative = max(eligible, key=lambda p: (p["note_count"], p["end_sec"] - p["start_sec"]), default=None)
        section = {
            "start_sec": round(lo, 3), "end_sec": round(hi, 3),
            "note_count": len(local), "detected_note_seconds": round(covered, 3),
            "note_coverage": round(covered / (hi - lo), 4) if hi > lo else 0.0,
            "representative_phrase_id": representative["id"] if representative else None,
            "representative_start_sec": representative["start_sec"] if representative else None,
            "representative_end_sec": representative["end_sec"] if representative else None,
            "contour": representative["contour"] if representative else None,
            "status": "candidate" if representative else "insufficient_phrase_evidence",
        }
        sections.append(section)
        if representative:
            evidence.append({"type": "contour", "phrase_id": representative["id"],
                             "start_sec": representative["start_sec"], "end_sec": representative["end_sec"],
                             "contour": representative["contour"]})
    for motif in motifs[:3]:
        evidence.append({"type": "repetition", "motif_id": motif["id"], "occurrences": motif["occurrences"]})

    text = (f"已扫描 {duration:.1f} 秒，保留 {len(notes)} 个候选音符、{len(phrases)} 个乐句；"
            f"稳定音符实际覆盖 {seconds:.1f} 秒（{overview['note_coverage']:.1%}）。")
    if sections and notes:
        parts = []
        for section in sections:
            label = (f"代表乐句呈{contours[section['contour']]}" if section["contour"] else "缺少完整乐句证据")
            parts.append(f"{section['start_sec']:.1f}–{section['end_sec']:.1f} 秒：{label}，音符覆盖 {section['note_coverage']:.0%}")
        text += " 分段：" + "；".join(parts) + "。"
    if motifs:
        a, b = motifs[0]["occurrences"][:2]
        text += (f" 检出 {len(motifs)} 组候选重复；例如 {a['start_sec']:.1f}–{a['end_sec']:.1f} 秒与"
                 f"{b['start_sec']:.1f}–{b['end_sec']:.1f} 秒的音程及相对节奏相似。")
    else:
        text += " 未检出满足音程、相对节奏及非重叠条件的重复；这不等于原曲没有重复。"
    return {"text": text, "overview": overview, "sections": sections, "evidence": evidence}


def describe_notes(notes: List[Dict[str, Any]], duration: Optional[float] = None) -> Dict[str, Any]:
    """分句用于呈现；重复检测另行滑窗，摘要覆盖整个分析时长。"""
    duration = float(duration if duration is not None else max((n["end_sec"] for n in notes), default=0.0))
    groups: List[List[Dict[str, Any]]] = []
    for note in notes:
        if not groups or note["start_sec"] - groups[-1][-1]["end_sec"] >= MIN_GAP_SEC \
                or len(groups[-1]) >= MAX_PHRASE_NOTES \
                or note["end_sec"] - groups[-1][0]["start_sec"] > MAX_PHRASE_SEC:
            groups.append([])
        groups[-1].append(note)

    phrases = []
    spans = []
    offset = 0
    for group in groups:
        spans.append((offset, offset + len(group)))
        offset += len(group)
        pitches = [n["pitch_midi"] for n in group]
        phrases.append({
            "id": f"phrase_{len(phrases) + 1}",
            "start_sec": group[0]["start_sec"], "end_sec": group[-1]["end_sec"],
            "note_count": len(group), "contour": _contour(pitches),
            "pitch_range_midi": [min(pitches), max(pitches)], "motif_id": None,
        })

    detected = _find_motifs(notes, phrases, spans)
    return {"phrases": phrases, **detected,
            "agent_summary": _summarize_track(notes, phrases, detected["motifs"], duration)}


def analyze_melody(y: np.ndarray, sr: int) -> Dict[str, Any]:
    """pYIN 是单声部音高估计；混音结果只能作为候选，不声明主旋律已确认。"""
    import librosa

    duration = len(y) / sr
    base = {
        "status": "insufficient_evidence", "method": "librosa.pyin",
        "source_assumption": "monophonic_or_dominant_lead",
        "limitation": "未分离音源；复杂混音可能将伴奏或贝斯误识别为旋律。",
        "seconds_analyzed": round(duration, 2), "notes": [], "phrases": [], "motifs": [],
        "agent_summary": _summarize_track([], [], [], duration),
    }
    if duration < 0.6:
        base["reason"] = "too_short"
        base["agent_summary"]["evidence"] = []
        return base
    audio = librosa.resample(y, orig_sr=sr, target_sr=MELODY_SR) if sr != MELODY_SR else y
    f0, voiced, probability = _track_pitch(audio, PITCH_MIN_MIDI, PITCH_MAX_MIDI)
    original_notes = _notes_from_frames(f0, voiced, probability, duration, MELODY_SR, HOP_LENGTH)
    f0, voiced, probability, boundary_quality = _validate_boundaries(audio, f0, voiced, probability)
    lower_notes = sum(n["pitch_midi"] == PITCH_MIN_MIDI for n in original_notes)
    boundary_quality.update({
        "initial_note_count": len(original_notes),
        "initial_lower_bound_note_count": lower_notes,
        "initial_lower_bound_note_fraction": round(lower_notes / max(1, len(original_notes)), 4),
    })
    notes = _notes_from_frames(f0, voiced, probability, duration, MELODY_SR, HOP_LENGTH)
    boundary_quality["retained_lower_bound_note_count"] = sum(n["pitch_midi"] == PITCH_MIN_MIDI for n in notes)
    boundary_quality["retained_note_count"] = len(notes)
    boundary_quality["retained_lower_bound_note_fraction"] = round(
        boundary_quality["retained_lower_bound_note_count"] / max(1, len(notes)), 4,
    )
    base["quality"] = {"boundary_validation": boundary_quality}
    if len(notes) < 4 or sum(n["end_sec"] - n["start_sec"] for n in notes) < 0.8:
        base["reason"] = "too_few_stable_notes"
        base["notes"] = notes
        base["agent_summary"] = _summarize_track(notes, [], [], duration)
        base["agent_summary"]["text"] += " 稳定音符不足，无法推断乐句。"
        base["agent_summary"]["evidence"] = []
        base["motif_search"] = {"status": "insufficient_notes", "reason": "too_few_stable_notes"}
        return base
    base.update({"status": "candidate", "notes": notes, **describe_notes(notes, duration)})
    if boundary_quality["rejected_boundary_frames"]:
        base["agent_summary"]["text"] += " 未通过跨音域复核的边界音高已剔除，覆盖率按剔除后计算。"
    return base
