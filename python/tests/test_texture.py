# -*- coding: utf-8 -*-
"""音色张力（第三维）测试。

这一维的存在理由是两个具体的错误认知，必须用测试钉死：

    "低沉 = 悲伤"   -> 低沉 + 高张力 应该是"悲壮/沉重"，不是"悲伤"
    "快 = 欢快"     -> 快 + 高张力 应该是"热烈/爆发"，不是"欢快"

历史 bug（本文件诞生前）：
    1. 只有 V-A 两维，`emotion` 由 (valence, arousal) 硬切，全库只有 7 种可能，
       小调钢琴独奏与低频失真电子只要 V/A 相近就得到同一个词。
    2. 首次加 texture 时忘了让 arousal 参与，导致 Coldplay《Yellow》
       （V=+0.21、86BPM、texture=0.361）从"平静"被错判成"欢快"。
       慢速大调应当是"平静" —— 速度决定活泼程度。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from music_emotion.texture import (  # noqa: E402
    BAND_HIGH, BAND_LOW, TEXTURE_RANGES, TEXTURE_WEIGHTS,
    describe_three_axis, texture_band, texture_score,
)


def _check(name: str, ok: bool, detail: str = "") -> bool:
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, ("  " + detail) if detail else ""))
    return ok


#: 三首已人工核过声学量的真实曲目的特征快照。
#: 用来防止改权重后这些"已知答案"翻车。
PROFILES = {
    "somber_slow": {          # 英雄主义 / 诀别书 一类：暗、慢、柔
        "spectral_flatness": 0.002, "percussive_ratio": 0.047,
        "spectral_flux": 0.120, "rms_db": -20.0, "spectral_centroid_hz": 937.0,
    },
    "driving_loud": {         # 燃/Die For You 一类：亮、响、有打击
        "spectral_flatness": 0.033, "percussive_ratio": 0.163,
        "spectral_flux": 0.209, "rms_db": -11.2, "spectral_centroid_hz": 2451.0,
    },
    "mid_slow": {             # Coldplay - Yellow：不闷但很慢
        "spectral_flatness": 0.010, "percussive_ratio": 0.090,
        "spectral_flux": 0.150, "rms_db": -15.0, "spectral_centroid_hz": 1800.0,
    },
}


def test_texture_separates_somber_from_driving() -> bool:
    print("\n[t1] 同样的低沉，张力必须不同")
    quiet = texture_score(PROFILES["somber_slow"])["texture"]
    loud = texture_score(PROFILES["driving_loud"])["texture"]
    ok = _check("安静悲伤类 落在 low 档", quiet < BAND_LOW, f"{quiet:.3f}")
    ok &= _check("冲击力强类 落在 high 档", loud >= BAND_HIGH, f"{loud:.3f}")
    ok &= _check("两者差距 > 0.25", (loud - quiet) > 0.25, f"Δ={loud - quiet:.3f}")
    return ok


def test_texture_monotonic() -> bool:
    """任一张力线索单独升高，总分都不能下降。"""
    print("\n[t2] 张力随各线索单调")
    base = dict(PROFILES["mid_slow"])
    lo = texture_score(base)["texture"]
    ok = True
    for key, (rlo, rhi) in TEXTURE_RANGES.items():
        bumped = dict(base)
        bumped[key] = rhi
        hi = texture_score(bumped)["texture"]
        ok &= _check(f"提高 {key} 不降低张力", hi >= lo - 1e-9, f"{lo:.3f} -> {hi:.3f}")
    return ok


def test_slow_major_is_calm_not_cheerful() -> bool:
    """回归：慢速大调必须是"平静"，不能因为音色不闷就变"欢快"。"""
    print("\n[t3] 慢速大调 = 平静（防止旧 bug 复发）")
    tex = texture_score(PROFILES["mid_slow"])["texture"]
    # Yellow 实测 V=+0.21, A=-0.24
    d = describe_three_axis(0.21, -0.24, tex)
    ok = _check("V=+0.21 A=-0.24 -> 平静", d["emotion"] == "平静", d["emotion"])
    ok &= _check("唤醒档判为 low", d["arousal_band"] == "low", d["arousal_band"])
    return ok


def test_fast_high_texture_is_passionate() -> bool:
    print("\n[t4] 快 + 高张力 = 热烈，不是欢快")
    tex = texture_score(PROFILES["driving_loud"])["texture"]
    # Runaway Baby 实测 V=+0.17, A=+0.47
    d = describe_three_axis(0.17, 0.47, tex)
    ok = _check("V=+0.17 A=+0.47 tex高 -> 热烈", d["emotion"] == "热烈", d["emotion"])
    ok &= _check("与'欢快'区分开", d["emotion"] != "欢快", d["emotion"])
    return ok


def test_low_arousal_high_texture_is_not_sad() -> bool:
    print("\n[t5] 低沉 + 高张力 不应该叫悲伤")
    d = describe_three_axis(-0.25, -0.15, 0.70)
    ok = _check("V=-0.25 A=-0.15 tex=0.70 -> 沉重", d["emotion"] == "沉重", d["emotion"])
    ok &= _check("不叫'悲伤'", d["emotion"] != "悲伤", d["emotion"])
    return ok


def test_weak_valence_refuses_direction() -> bool:
    print("\n[t6] 效价接近 0 时不下情绪方向判断")
    ok = True
    for t in (0.10, 0.45, 0.75):
        d = describe_three_axis(0.03, 0.05, t)
        ok &= _check(f"V=0.03 tex={t} 不判方向", d["directional"] is False,
                     d["emotion"])
    ok &= _check("高张力时报'强烈'而不是'欢快'",
                 describe_three_axis(0.03, 0.05, 0.75)["emotion"] != "欢快")
    return ok


def test_missing_features_flagged() -> bool:
    """缺特征时要明说，不能假装算准。"""
    print("\n[t7] 缺失特征被标记")
    r = texture_score({"rms_db": -15.0})
    ok = _check("missing 非空", len(r["missing"]) == 4, str(r["missing"]))
    ok &= _check("缺失时仍返回可用分数", 0.0 <= r["texture"] <= 1.0, f"{r['texture']:.3f}")
    return ok


def test_weights_normalized() -> bool:
    print("\n[t8] 权重与分档配置自洽")
    s = sum(TEXTURE_WEIGHTS.values())
    # 用真实分档边界构造三段代表值，而不是写死数字——
    # 这样以后调 BAND_LOW/BAND_HIGH 时测试不会跟着过期。
    below = (BAND_LOW + BAND_HIGH) / 2 - 0.05
    mid_lo = BAND_HIGH
    mid_hi = min(BAND_HIGH + 0.25, 0.99)
    above = min(BAND_HIGH + 0.30, 0.99)

    ok = _check("权重和为 1", abs(s - 1.0) < 1e-9, f"{s:.4f}")
    ok &= _check("每个权重项都有归一区间",
                 all(k in TEXTURE_RANGES for k in TEXTURE_WEIGHTS))
    ok &= _check("分档边界有序", BAND_LOW < BAND_HIGH, f"{BAND_LOW} < {BAND_HIGH}")
    ok &= _check("low 段正确", texture_band(0.0) == "low", texture_band(0.0))
    ok &= _check("just below high 仍是 mid",
                 texture_band(mid_lo - 0.001) == "mid", f"{mid_lo - 0.001:.3f}")
    ok &= _check("high 段正确", texture_band(above) == "high", f"{above:.3f}")
    return ok


def main() -> int:
    print("=" * 66)
    print("音色张力（第三维）测试")
    print("=" * 66)
    results = [
        test_texture_separates_somber_from_driving(),
        test_texture_monotonic(),
        test_slow_major_is_calm_not_cheerful(),
        test_fast_high_texture_is_passionate(),
        test_low_arousal_high_texture_is_not_sad(),
        test_weak_valence_refuses_direction(),
        test_missing_features_flagged(),
        test_weights_normalized(),
    ]
    passed = sum(bool(r) for r in results)
    print("\n" + "=" * 66)
    print("结果：%d/%d 项通过" % (passed, len(results)))
    print("=" * 66)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
