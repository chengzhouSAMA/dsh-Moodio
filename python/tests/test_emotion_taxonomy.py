# -*- coding: utf-8 -*-
"""离散情绪词表的行为测试。

这套词表是给人看的第一界面，错一个字就会让人误解整首歌的结论，
所以边界必须钉死：
    - 四象限主词必须与 V/A 符号一致；
    - "略偏"不能被说成强情绪词（历史 bug：258 首被判"紧张"）；
    - 真正平坦的音轨才允许叫"平淡"。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from music_emotion.emotion_taxonomy import (  # noqa: E402
    EMOTIONS, FLAT_RADIUS, VALUE_WEAK_BAND, emotion_for, emotion_profile,
)


def _check(name: str, ok: bool, detail: str = "") -> bool:
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, ("  " + detail) if detail else ""))
    return ok


def test_quadrant_mapping() -> bool:
    print("\n[t1] 四象限主词与 V/A 符号一致")
    cases = [
        (0.50, 0.40, "欢快"), (0.40, -0.30, "平静"),
        (-0.50, 0.40, "紧张"), (-0.50, -0.40, "悲伤"),
    ]
    ok = True
    for v, a, expect in cases:
        got = emotion_for(v, a)[0]
        ok &= _check(f"V={v:+.2f} A={a:+.2f} -> {expect}", got == expect, got)
    return ok


def test_weak_band_does_not_overclaim() -> bool:
    """效价接近 0 时不能给出方向性情绪词。

    历史 bug：没有弱带时，V=-0.05/A=+0.40 被判"紧张"，
    导致全库 258 首（超过"平静"的 115 首）被判紧张，明显不合理。
    """
    print("\n[t2] 弱效应带内不下强判断")
    ok = True
    for v in (-0.08, -0.03, 0.0, 0.03, 0.08):
        prof = emotion_profile(v, 0.40)
        ok &= _check(
            f"V={v:+.2f} A=+0.40 不判四象限词",
            prof["emotion"] not in ("紧张", "欢快", "悲伤", "平静"),
            prof["emotion"],
        )
    # 明显偏负时才允许叫"紧张"
    ok &= _check("V=-0.30 A=+0.40 可以是紧张",
                 emotion_profile(-0.30, 0.40)["emotion"] == "紧张")
    # 弱带边界要一致
    ok &= _check("|V| 刚好等于弱带边界仍算弱",
                 emotion_profile(VALUE_WEAK_BAND, 0.40)["emotion"] not in
                 ("紧张", "欢快", "悲伤", "平静"),
                 emotion_profile(VALUE_WEAK_BAND, 0.40)["emotion"])
    return ok


def test_flat_only_near_origin() -> bool:
    print("\n[t3] 只有贴近原点才算平淡")
    ok = _check("V=0.01 A=0.01 -> 平淡",
                emotion_profile(0.01, 0.01)["emotion_id"] == "flat")
    # 《诀别书》这类"安静地悲伤"不能被吞成平淡
    prof = emotion_profile(-0.163, -0.129)
    ok &= _check("V=-0.16 A=-0.13 不判平淡", prof["emotion_id"] != "flat", prof["emotion"])
    ok &= _check("V=-0.16 A=-0.13 判悲伤", prof["emotion"] == "悲伤", prof["emotion"])
    return ok


def test_magnitude_monotonic() -> bool:
    """强度分档必须随模长单调，不能出现"更强烈反而标 mild"。"""
    print("\n[t4] 强度分档随模长单调")
    order = {"flat": 0, "mild": 1, "moderate": 2, "strong": 3}
    seq = [(0.0, 0.0), (-0.12, -0.12), (-0.25, -0.25), (-0.45, -0.45)]
    levels = [order[emotion_profile(v, a)["intensity"]] for v, a in seq]
    ok = _check("阶梯不下降", levels == sorted(levels), str(levels))
    ok &= _check("最强档确实到 strong", levels[-1] == 3)
    return ok


def test_all_quadrants_have_words() -> bool:
    print("\n[t5] 词表完整性")
    ok = _check("四象限齐全", len(EMOTIONS) == 4)
    ok &= _check("每个象限都有中英文",
                 all(k in v for v in EMOTIONS.values() for k in ("zh", "id")))
    ok &= _check("平淡半径小于弱带", FLAT_RADIUS < VALUE_WEAK_BAND,
                 f"{FLAT_RADIUS} < {VALUE_WEAK_BAND}")
    return ok


def main() -> int:
    print("=" * 66)
    print("离散情绪词表测试")
    print("=" * 66)
    results = [
        test_quadrant_mapping(),
        test_weak_band_does_not_overclaim(),
        test_flat_only_near_origin(),
        test_magnitude_monotonic(),
        test_all_quadrants_have_words(),
    ]
    passed = sum(bool(r) for r in results)
    print("\n" + "=" * 66)
    print("结果：%d/%d 项通过" % (passed, len(results)))
    print("=" * 66)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
