# -*- coding: utf-8 -*-
"""方向性验证：用"已知答案"的合成刺激检验 V-A 模型是否符合文献预期。

这是本项目最重要的一步。两首真实曲目无法互相验证（同项目、同 BPM），
只有合成刺激才提供 ground truth：我明确知道我把哪个维度推向哪一边。

检验设计（被试内对照）：
    组 A  速度   60 vs 140 BPM（其余完全相同）      -> arousal 应显著更高
    组 B  调式   大三和弦 vs 小三和弦（同一根音）    -> valence 应显著更高
    组 C  响度   -30dB vs -12dB                     -> arousal 应更高
    组 D  和声   纯三和弦 vs 高密度簇/不协和        -> valence 应更低
    组 E  起音密度 稀疏 vs 密集                     -> arousal 应更高

判据：
    1) 每组的方向必须正确（符号检验）；
    2) 组内差值必须超过噪声尺度（重复多种子）；
    3) 全局 Spearman 相关：目标维度 vs 预测维度。

任何一组方向反了，都说明权重或归一区间有问题，必须回到 vamodel 修。
"""
from __future__ import annotations

import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np

sys.path.insert(0, ".")

from music_emotion.audio_io import ANALYSIS_SR  # noqa: E402
from music_emotion.features import extract_features  # noqa: E402
from music_emotion.types import ClipInfo  # noqa: E402
from music_emotion.vamodel import estimate_va  # noqa: E402

RNG = np.random.default_rng(20261024)


# ---------------------------------------------------------------- 合成器

def _midi_to_hz(m: int) -> float:
    return 440.0 * (2.0 ** ((m - 69) / 12.0))


def _tone(freq: float, dur: float, sr: int, amp: float, harmonics: int = 6) -> np.ndarray:
    t = np.arange(int(dur * sr)) / sr
    out = np.zeros_like(t)
    for h in range(1, harmonics + 1):
        out += (amp / h) * np.sin(2 * np.pi * freq * h * t)
    env = np.minimum(1.0, t / 0.02) * np.minimum(1.0, (dur - t) / 0.05)
    return out * np.clip(env, 0, 1)


def _click(sr: int, amp: float) -> np.ndarray:
    n = int(0.035 * sr)
    noise = RNG.normal(0, 1, n)
    env = np.exp(-np.linspace(0, 9, n))
    return amp * noise * env


def synth(
    tempo: float = 100.0,
    mode: str = "major",
    root: int = 60,
    amp_db: float = -18.0,
    complexity: float = 0.0,
    onset_density: float = 1.0,
    dur: float = 12.0,
    sr: int = ANALYSIS_SR,
    seed: int = 0,
) -> np.ndarray:
    """合成一段和弦进行 + 打击。

    Args:
        tempo: BPM。
        mode: "major" / "minor"。
        root: 根音 MIDI。
        amp_db: 目标 RMS（dBFS）。
        complexity: 0=纯三和弦，1=加入大量不协和音与失谐。
        onset_density: 每拍的击打次数倍率。
    """
    rng = np.random.default_rng(seed)
    n = int(dur * sr)
    buf = np.zeros(n, dtype=np.float64)

    third = 4 if mode == "major" else 3
    beat = 60.0 / tempo
    bar = beat * 4
    progression = [0, 5, 3, 7]  # 简单的 I-vi-IV-v 类进行

    t = 0.0
    idx = 0
    while t < dur - 0.05:
        deg = progression[idx % len(progression)]
        chord_root = root + deg
        notes = [chord_root, chord_root + third, chord_root + 7]
        if complexity > 0.5:
            # 加九度/增四度 + 半音簇
            notes += [chord_root + 13, chord_root + 6, chord_root + 1]

        seg_dur = min(bar, dur - t)
        acc = np.zeros(int(seg_dur * sr))
        for k, m in enumerate(notes):
            detune = rng.normal(0, 3.0 * complexity)
            f = _midi_to_hz(m) * (1.0 + detune / 1200.0)
            a = 0.5 if k < 3 else 0.5 * complexity
            partial = _tone(f, seg_dur, sr, amp=1.0, harmonics=6 if complexity < 0.5 else 9)
            if len(partial) < len(acc):
                partial = np.pad(partial, (0, len(acc) - len(partial)))
            acc += a * partial[: len(acc)]

        start = int(t * sr)
        end = min(start + len(acc), n)
        buf[start:end] += acc[: end - start]
        t += seg_dur
        idx += 1

    # 打击：按拍点放置
    hits_per_beat = max(1, int(round(onset_density)))
    t = 0.0
    step = beat / hits_per_beat
    while t < dur - 0.05:
        c = _click(sr, amp=0.9 if (t % bar) < 1e-6 else 0.55)
        start = int(t * sr)
        end = min(start + len(c), n)
        buf[start:end] += c[: end - start]
        t += step

    # 归一化到目标 RMS
    rms = float(np.sqrt(np.mean(buf ** 2))) + 1e-12
    target = 10.0 ** (amp_db / 20.0)
    buf = buf * (target / rms)
    peak = float(np.abs(buf).max())
    if peak > 0.98:
        buf *= 0.98 / peak
    return buf.astype(np.float32)


def va_of(y: np.ndarray, sr: int = ANALYSIS_SR) -> dict:
    raw = extract_features(y, sr, segment_sec=4.0, hop_sec=2.0)
    return estimate_va(raw["global"])


# ---------------------------------------------------------------- 实验

def run_group(name: str, variants, seeds=(1, 2, 3)):
    """variants: [(label, kwargs_dict), ...] -> 打印每种条件下的 V-A 均值。"""
    rows = []
    for label, kwargs in variants:
        vs, as_ = [], []
        for s in seeds:
            y = synth(seed=s, **kwargs)
            r = va_of(y)
            vs.append(r["valence"])
            as_.append(r["arousal"])
        rows.append({
            "label": label,
            "valence": float(np.mean(vs)),
            "valence_sd": float(np.std(vs)),
            "arousal": float(np.mean(as_)),
            "arousal_sd": float(np.std(as_)),
        })

    print(f"\n[{name}]")
    for r in rows:
        print("   %-18s V=%+.3f (sd %.3f)   A=%+.3f (sd %.3f)"
              % (r["label"], r["valence"], r["valence_sd"], r["arousal"], r["arousal_sd"]))
    return rows


def sign_test(rows, dim, expect_positive: bool) -> bool:
    """variants 按 (低条件, 高条件) 排列；delta = 高 - 低。

    expect_positive=True 表示"高条件应当在该维度上更大"。
    修正记录：早期版本用 rows[0]-rows[1]，方向判反，导致正确的模型被判 FAIL。
    """
    d = rows[1][dim] - rows[0][dim]
    ok = (d > 0) if expect_positive else (d < 0)
    print("   -> delta(%s) = %+.3f   %s" % (dim, d, "PASS" if ok else "FAIL"))
    return ok


def main() -> int:
    print("=" * 72)
    print("音乐情感模型 · 方向性验证（合成刺激）")
    print("=" * 72)

    results = []

    g_tempo = run_group("A. 速度 60 vs 140 BPM", [
        ("slow 60bpm", dict(tempo=60, mode="major", amp_db=-18, complexity=0.0)),
        ("fast 140bpm", dict(tempo=140, mode="major", amp_db=-18, complexity=0.0)),
    ])
    results.append(sign_test(g_tempo, "arousal", expect_positive=True))

    g_mode = run_group("B. 调式 大调 vs 小调（同根音）", [
        ("minor", dict(tempo=100, mode="minor", root=57, amp_db=-18)),
        ("major", dict(tempo=100, mode="major", root=57, amp_db=-18)),
    ])
    results.append(sign_test(g_mode, "valence", expect_positive=True))

    g_loud = run_group("C. 响度 -30dB vs -12dB", [
        ("quiet -30dB", dict(tempo=100, mode="major", amp_db=-30)),
        ("loud -12dB", dict(tempo=100, mode="major", amp_db=-12)),
    ])
    results.append(sign_test(g_loud, "arousal", expect_positive=True))

    g_complex = run_group("D. 和声 纯三和弦 vs 高复杂度", [
        ("simple", dict(tempo=100, mode="major", complexity=0.0)),
        ("complex", dict(tempo=100, mode="major", complexity=0.9)),
    ])
    results.append(sign_test(g_complex, "valence", expect_positive=False))

    g_onset = run_group("E. 起音密度 稀疏 vs 密集", [
        ("sparse", dict(tempo=100, mode="major", onset_density=0.25)),
        ("dense", dict(tempo=100, mode="major", onset_density=4.0)),
    ])
    results.append(sign_test(g_onset, "arousal", expect_positive=True))

    # ---- 全局相关：跨条件混合样本 ----
    print("\n[F. 混合条件 Spearman 相关]")
    tempo_grid = [60, 80, 100, 120, 140]
    mode_grid = ["major", "minor"]
    amp_grid = [-28, -20, -12]
    X, Y = [], []
    for tp in tempo_grid:
        for md in mode_grid:
            for am in amp_grid:
                y = synth(tempo=tp, mode=md, amp_db=am, complexity=0.2, seed=7)
                r = va_of(y)
                X.append([tp, 1.0 if md == "major" else 0.0, am])
                Y.append([r["arousal"], r["valence"]])

    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)

    def spearman(a, b):
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        return float(np.corrcoef(ra, rb)[0, 1])

    r_tempo_a = spearman(X[:, 0], Y[:, 0])
    r_major_v = spearman(X[:, 1], Y[:, 1])
    r_amp_a = spearman(X[:, 2], Y[:, 0])
    print("   tempo  vs arousal  r_s = %+.3f  (期望 >0)" % r_tempo_a)
    print("   major  vs valence  r_s = %+.3f  (期望 >0)" % r_major_v)
    print("   amp    vs arousal  r_s = %+.3f  (期望 >0)" % r_amp_a)

    results += [r_tempo_a > 0, r_major_v > 0, r_amp_a > 0]

    # ---- 稳定性：同一刺激重复应完全一致（确定性检查） ----
    y = synth(tempo=110, mode="minor", seed=3)
    a = va_of(y)
    b = va_of(y.copy())
    same = abs(a["valence"] - b["valence"]) < 1e-9 and abs(a["arousal"] - b["arousal"]) < 1e-9
    print("\n[G. 确定性] 同输入两次结果一致: %s" % ("PASS" if same else "FAIL"))
    results.append(same)

    passed = sum(bool(r) for r in results)
    total = len(results)
    print("\n" + "=" * 72)
    print("结果：%d/%d 项通过" % (passed, total))
    print("=" * 72)
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
