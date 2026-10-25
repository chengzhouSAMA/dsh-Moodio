# -*- coding: utf-8 -*-
"""V-A 模型标定性验证。

这里不测"预测准不准"（没有人工标注，无法测），而是测**模型的不变量**：
在已知输入下，输出必须满足哪些约束。这些约束一旦被破坏，历史上有过真实后果：

    2026-10 实测：VALENCE_NEUTRAL 误设为 0.47（而非可验算的 0.25），
    导致 valence 系统性偏负 —— 523 首真实曲目里 264 首被判"低沉/忧郁"，
    4 个情感象限只有 2 个能出现，二维模型退化成一条线。

所以本文件的核心是"中性不变量"：**没有证据时必须输出中性**。
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, ".")

from music_emotion import vamodel as vm  # noqa: E402


#: 构造一个"每个通道归一值都恰好 0.5"的输入。
#: 数值都取各自 FEATURE_RANGES 的区间中点，mode_score 取区间中点 0。
NEUTRAL_PROBE = {
    "tempo_bpm": (60.0 + 170.0) / 2,
    "onset_rate": (0.3 + 6.0) / 2,
    "rms_db": (-26.0 + -4.0) / 2,
    "spectral_flux": (0.08 + 0.45) / 2,
    "percussive_ratio": (0.02 + 0.30) / 2,
    "spectral_centroid_hz": (600.0 + 3500.0) / 2,
    "key_clarity": (0.3 + 0.9) / 2,
    "pulse_clarity": 0.5,
    "mode_score": 0.0,
    "harmonic_complexity": 0.5,
    "duration_sec": 200.0,
}

#: 一组"偏负面"的输入：小调、慢、暗、噪。
NEGATIVE_PROBE = {
    "tempo_bpm": 70.0,
    "onset_rate": 0.8,
    "rms_db": -22.0,
    "spectral_flux": 0.12,
    "percussive_ratio": 0.05,
    "spectral_centroid_hz": 900.0,
    "key_clarity": 0.4,
    "pulse_clarity": 0.2,
    "mode_score": -0.25,
    "harmonic_complexity": 0.85,
    "duration_sec": 200.0,
}

#: 一组"偏正面"的输入：大调、快、亮、谐和。
POSITIVE_PROBE = {
    "tempo_bpm": 145.0,
    "onset_rate": 4.5,
    "rms_db": -8.0,
    "spectral_flux": 0.38,
    "percussive_ratio": 0.25,
    "spectral_centroid_hz": 2900.0,
    "key_clarity": 0.85,
    "pulse_clarity": 0.8,
    "mode_score": 0.25,
    "harmonic_complexity": 0.15,
    "duration_sec": 200.0,
}


def _check(name: str, ok: bool, detail: str = "") -> bool:
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, ("  " + detail) if detail else ""))
    return ok


def test_neutral_invariant() -> bool:
    """核心不变量：每通道归一值 0.5 -> V=A=0，标签 neutral。"""
    print("\n[t1] 中性不变量")
    # 先确认探针确实让每个通道归一值落在 0.5
    bad = [
        k for k in set(vm.AROUSAL_WEIGHTS) | set(vm.VALENCE_WEIGHTS)
        if abs(vm._feature_norm(k, NEUTRAL_PROBE) - 0.5) > 1e-9
    ]
    ok = _check("探针各通道归一值均为 0.5", not bad, str(bad) if bad else "")

    r = vm.estimate_va(NEUTRAL_PROBE)
    ok &= _check("valence == 0", abs(r["valence"]) < 1e-9, "%.2e" % r["valence"])
    ok &= _check("arousal == 0", abs(r["arousal"]) < 1e-9, "%.2e" % r["arousal"])
    ok &= _check("label == neutral", r["label"] == "neutral", r["label"])
    return ok


def test_constants_are_fixed_points() -> bool:
    """常量必须等于"权重和 × 0.5"，否则就不是不动点。"""
    print("\n[t2] 常量可验算性")
    expect_v = sum(vm.VALENCE_WEIGHTS.values()) * 0.5
    expect_a = sum(vm.AROUSAL_WEIGHTS.values()) * 0.5
    ok = _check("VALENCE_NEUTRAL == Σw*0.5", abs(vm.VALENCE_NEUTRAL - expect_v) < 1e-9,
                "%.4f vs %.4f" % (vm.VALENCE_NEUTRAL, expect_v))
    ok &= _check("AROUSAL_NEUTRAL == Σw*0.5", abs(vm.AROUSAL_NEUTRAL - expect_a) < 1e-9,
                 "%.4f vs %.4f" % (vm.AROUSAL_NEUTRAL, expect_a))
    return ok


def test_directionality() -> bool:
    """方向性：负面探针与正面探针必须在预期方向上分开。"""
    print("\n[t3] 方向性")
    neg = vm.estimate_va(NEGATIVE_PROBE)
    pos = vm.estimate_va(POSITIVE_PROBE)
    print("     negative: V=%+.3f A=%+.3f %s" % (neg["valence"], neg["arousal"], neg["label"]))
    print("     positive: V=%+.3f A=%+.3f %s" % (pos["valence"], pos["arousal"], pos["label"]))
    ok = _check("valence 正探针 > 负探针", pos["valence"] > neg["valence"],
                "%+.3f vs %+.3f" % (pos["valence"], neg["valence"]))
    ok &= _check("arousal 正探针 > 负探针", pos["arousal"] > neg["arousal"],
                 "%+.3f vs %+.3f" % (pos["arousal"], neg["arousal"]))
    return ok


def test_quadrant_reachability() -> bool:
    """四象限都必须可达，否则模型退化成一条线。"""
    print("\n[t4] 四象限可达性")
    from collections import Counter

    rng = np.random.default_rng(20261024)
    feats = []
    for _ in range(4000):
        feats.append({
            "tempo_bpm": rng.uniform(60, 170), "onset_rate": rng.uniform(0.3, 6),
            "rms_db": rng.uniform(-26, -4), "spectral_flux": rng.uniform(0.08, 0.45),
            "percussive_ratio": rng.uniform(0.02, 0.30),
            "spectral_centroid_hz": rng.uniform(600, 3500),
            "key_clarity": rng.uniform(0.3, 0.9), "pulse_clarity": rng.uniform(0, 1),
            "mode_score": rng.uniform(-0.3, 0.3),
            "harmonic_complexity": rng.uniform(0, 1), "duration_sec": 200.0,
        })
    labels = Counter(vm.estimate_va(f)["label"] for f in feats)
    quads = [k for k in labels if k != "neutral"]
    print("     labels: %s" % dict(labels))
    return _check("四个非中性象限均可达", len(quads) == 4, "reached %d/4" % len(quads))


def test_calibration_modes_differ() -> bool:
    """按库标定必须真的改变中性点，且不破坏有序性。"""
    print("\n[t5] 按库标定模式")
    rng = np.random.default_rng(7)
    feats = [{
        "tempo_bpm": rng.uniform(130, 170),   # 故意造一个"整体偏快偏亮"的库
        "onset_rate": rng.uniform(3, 6),
        "rms_db": rng.uniform(-10, -4),
        "spectral_flux": rng.uniform(0.3, 0.45),
        "percussive_ratio": rng.uniform(0.15, 0.3),
        "spectral_centroid_hz": rng.uniform(2500, 3500),
        "key_clarity": rng.uniform(0.7, 0.9),
        "pulse_clarity": rng.uniform(0.6, 1.0),
        "mode_score": rng.uniform(0.1, 0.3),
        "harmonic_complexity": rng.uniform(0, 0.3),
        "duration_sec": 200.0,
    } for _ in range(200)]

    cal = vm.calibrate(feats, "median")
    fixed_v = np.mean([vm.estimate_va(f)["valence"] for f in feats])
    rel_v = np.mean([vm.estimate_va(f, cal["neutral"])["valence"] for f in feats])
    print("     fixed mean V=%+.3f   corpus-relative mean V=%+.3f" % (fixed_v, rel_v))
    ok = _check("固定模式保留绝对倾向(偏正)", fixed_v > 0.15, "%+.3f" % fixed_v)
    ok &= _check("按库标定后方差居中", abs(rel_v) < 0.05, "%+.3f" % rel_v)
    ok &= _check("calibrate 返回可用中性点", len(cal["neutral"]) == 2 and cal["n"] == len(feats))
    return ok


def test_determinism() -> bool:
    """同输入必须完全一致，否则断点续跑的结果不可复现。"""
    print("\n[t6] 确定性")
    a = vm.estimate_va(POSITIVE_PROBE)
    b = vm.estimate_va(dict(POSITIVE_PROBE))
    ok = _check("两次调用一致",
                a["valence"] == b["valence"] and a["arousal"] == b["arousal"])
    return ok


def test_label_deadzone_uses_both_axes() -> bool:
    """neutral 必须同时要求两个维度都在死区内。

    历史 bug：早期只检查 valence，导致"效价模糊但唤醒明确"的曲子被判中性。
    本库实测有 45 首如此（如 162BPM、arousal=+0.47 的 Runaway Baby）。
    """
    print("\n[t7] neutral 必须使用两个维度")
    # 效价模糊、唤醒很高 -> 不能是 neutral
    a = vm.label_for(0.0, 0.45)[0]
    ok = _check("V=0, A=+0.45 不判 neutral", a != "neutral", a)

    # 效价模糊、唤醒很低 -> 不能是 neutral
    b = vm.label_for(0.0, -0.45)[0]
    ok &= _check("V=0, A=-0.45 不判 neutral", b != "neutral", b)

    # 两个维度都在死区内 -> 必须是 neutral
    c = vm.label_for(0.10, 0.10)[0]
    ok &= _check("V=0.10, A=0.10 判 neutral", c == "neutral", c)

    # 边界：都恰好等于 deadzone 不算严格超出 -> neutral
    d = vm.label_for(0.18, 0.18)[0]
    ok &= _check("|V|=|A|=0.18 判 neutral", d == "neutral", d)

    # 一个超出死区 -> 不能 neutral
    e = vm.label_for(0.19, 0.0)[0]
    ok &= _check("|V|=0.19 判非 neutral", e != "neutral", e)
    return ok


def main() -> int:
    print("=" * 66)
    print("V-A 模型标定性验证")
    print("=" * 66)
    results = [
        test_neutral_invariant(),
        test_constants_are_fixed_points(),
        test_directionality(),
        test_quadrant_reachability(),
        test_calibration_modes_differ(),
        test_determinism(),
        test_label_deadzone_uses_both_axes(),
    ]
    passed = sum(bool(r) for r in results)
    print("\n" + "=" * 66)
    print("结果：%d/%d 项通过" % (passed, len(results)))
    print("=" * 66)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
