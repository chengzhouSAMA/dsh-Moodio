# -*- coding: utf-8 -*-
"""四轴判定测试：texture（攻击性）与 grandeur（规模感）必须分开。

本文件是两次真实误判的回归护栏：

    误判 1：只有 V-A 两维 -> "低沉"被当成"悲伤"、"快"被当成"欢快"
    误判 2：只有 texture  -> 《英雄主义》被判"悲伤"，实际是"悲壮"。
            它动态 34.6dB、40.9% 时间满响、对比度 27.8，是实测最宏大的一首，
            但没有失真、没有高频噪声，于是 texture 只有 0.10，
            被误读成"安静"。

核心断言：**"力量"有两种，管弦的宏大 ≠ 电子的粗糙。**
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from music_emotion.four_axis import classify_four_axis  # noqa: E402
from music_emotion.texture import (  # noqa: E402
    GRANDEUR_HIGH, GRANDEUR_MID, grandeur_band, grandeur_score,
)

#: 实测特征快照（来自真实曲目，已人工核对声学量）。
SNAPSHOTS = {
    # 英雄主义：动态巨大、持续满响、厚织体，但无失真
    "heroic_somber": {
        "dynamic_range_db": 34.6, "full_loud_share": 40.9,
        "spectral_contrast": 27.8, "active_pitches": 9.0,
    },
    # 诀别书：动态也不小，但长时间是轻的，规模感弱
    "intimate_sad": {
        "dynamic_range_db": 28.9, "full_loud_share": 25.7,
        "spectral_contrast": 26.9, "active_pitches": 10.0,
    },
    # Victory：持续强奏最久，最"满"
    "epic_triumphant": {
        "dynamic_range_db": 26.5, "full_loud_share": 60.1,
        "spectral_contrast": 22.2, "active_pitches": 12.0,
    },
    # Yellow：动态很小，平
    "flat_calm": {
        "dynamic_range_db": 12.5, "full_loud_share": 53.8,
        "spectral_contrast": 25.1, "active_pitches": 12.0,
    },
}


def _check(name: str, ok: bool, detail: str = "") -> bool:
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, ("  " + detail) if detail else ""))
    return ok


def test_grandeur_separates_heroic_from_intimate() -> bool:
    print("\n[t1] 宏大 vs 私密：同样悲伤，规模感必须不同")
    hero = grandeur_score(SNAPSHOTS["heroic_somber"])["grandeur"]
    intimate = grandeur_score(SNAPSHOTS["intimate_sad"])["grandeur"]
    ok = _check("英雄主义 grandeur >= GRANDEUR_HIGH", hero >= GRANDEUR_HIGH, f"{hero:.3f}")
    ok &= _check("诀别书 grandeur 落在 mid/low", intimate < GRANDEUR_HIGH, f"{intimate:.3f}")
    ok &= _check("两者差距 >= 0.15", (hero - intimate) >= 0.15, f"Δ={hero - intimate:.3f}")
    return ok


def test_heroic_is_somber_not_sad() -> bool:
    """核心回归：负效价 + 低唤醒 + 低攻击性 + 高宏大 = 悲壮。"""
    print("\n[t2] 英雄主义必须是'悲壮'而不是'悲伤'")
    g = grandeur_score(SNAPSHOTS["heroic_somber"])["grandeur"]
    # 实测 V=-0.225, A=-0.114, texture=0.103
    d = classify_four_axis(-0.225, -0.114, 0.103, g)
    ok = _check("判为'悲壮'", d["emotion"] == "悲壮", d["emotion"])
    ok &= _check("不再是'悲伤'", d["emotion"] != "悲伤", d["emotion"])
    ok &= _check("grandeur 档位为 high", d["grandeur_band"] == "high", d["grandeur_band"])
    return ok


def test_intimate_sad_stays_sad() -> bool:
    print("\n[t3] 诀别书仍应保持'悲伤'")
    g = grandeur_score(SNAPSHOTS["intimate_sad"])["grandeur"]
    d = classify_four_axis(-0.163, -0.129, 0.150, g)
    ok = _check("仍判'悲伤'", d["emotion"] == "悲伤", d["emotion"])
    ok &= _check("没有误升为'悲壮'", d["emotion"] != "悲壮", d["emotion"])
    return ok


def test_epic_triumphant() -> bool:
    print("\n[t4] 正效价 + 宏大 = 昂扬/壮丽，不是欢快")
    g = grandeur_score(SNAPSHOTS["epic_triumphant"])["grandeur"]
    d = classify_four_axis(0.28, 0.04, 0.326, g)
    ok = _check("判为昂扬或壮丽", d["emotion"] in ("昂扬", "壮丽"), d["emotion"])
    ok &= _check("不是'欢快'", d["emotion"] != "欢快", d["emotion"])
    return ok


def test_slow_major_still_calm() -> bool:
    """回归：防止再犯"慢速大调被判欢快"的错。"""
    print("\n[t5] 慢速大调仍应'平静'")
    g = grandeur_score(SNAPSHOTS["flat_calm"])["grandeur"]
    d = classify_four_axis(0.21, -0.24, 0.361, g)
    ok = _check("判'平静'", d["emotion"] == "平静", d["emotion"])
    ok &= _check("唤醒档 low", d["arousal_band"] == "low", d["arousal_band"])
    return ok


def test_high_attack_mid_scale() -> bool:
    print("\n[t6] 高攻击性 + 中等规模 = 热烈（不是悲壮）")
    g = grandeur_score(SNAPSHOTS["flat_calm"])["grandeur"]
    d = classify_four_axis(0.17, 0.47, 0.569, g)
    ok = _check("判'热烈'", d["emotion"] == "热烈", d["emotion"])
    return ok


def test_negative_high_both_is_somber_majestic() -> bool:
    print("\n[t7] 负效价 + 高宏大 + 高攻击性 = 仍是悲壮（规模压过粗糙）")
    d = classify_four_axis(-0.30, 0.20, 0.70, 0.75)
    ok = _check("判'悲壮'", d["emotion"] == "悲壮", d["emotion"])
    return ok


def test_neutral_valence_refuses_direction() -> bool:
    print("\n[t8] 效价接近 0 时不宣称情绪方向")
    ok = True
    for g in (0.2, 0.5, 0.8):
        d = classify_four_axis(0.04, 0.02, 0.5, g)
        ok &= _check(f"grandeur={g} 时 directional=False",
                     d["directional"] is False, d["emotion"])
    ok &= _check("高宏大时报'宏伟/强烈'而非情绪词",
                 classify_four_axis(0.04, 0.02, 0.2, 0.8)["emotion"] in ("宏伟", "辽阔", "强烈"))
    return ok


def test_band_boundaries() -> bool:
    print("\n[t9] 分档边界")
    ok = _check("0.0 -> low", grandeur_band(0.0) == "low")
    ok &= _check("GRANDEUR_MID-0.01 -> low", grandeur_band(GRANDEUR_MID - 0.01) == "low")
    ok &= _check("GRANDEUR_MID -> mid", grandeur_band(GRANDEUR_MID) == "mid")
    ok &= _check("GRANDEUR_HIGH-0.01 -> mid", grandeur_band(GRANDEUR_HIGH - 0.01) == "mid")
    ok &= _check("GRANDEUR_HIGH -> high", grandeur_band(GRANDEUR_HIGH) == "high")
    ok &= _check("边界有序", GRANDEUR_MID < GRANDEUR_HIGH, f"{GRANDEUR_MID} < {GRANDEUR_HIGH}")
    return ok


def test_missing_grandeur_features_flagged() -> bool:
    print("\n[t10] 缺失特征被标记")
    r = grandeur_score({"dynamic_range_db": 20.0})
    ok = _check("missing 非空", len(r["missing"]) == 3, str(r["missing"]))
    ok &= _check("仍返回可用分数", 0.0 <= r["grandeur"] <= 1.0, f"{r['grandeur']:.3f}")
    return ok


def test_flat_audio_is_not_grand() -> bool:
    """恒定电平不能被判成宏大。

    实测：一段 12 秒恒定 RMS 的合成音，full_loud_share 按定义就是 100%，
    于是初版 grandeur=0.65（high）被说成"辽阔"。但"持续强奏"只有在真有
    动态落差时才有意义，所以按实测动态范围对 full_loud_share 做渐变抑制。
    """
    print("\n[t11] 恒定电平不判宏大")
    flat = {"dynamic_range_db": 0.0, "full_loud_share": 100.0,
            "spectral_contrast": 26.0, "active_pitches": 9.0}
    g = grandeur_score(flat)
    ok = _check("grandeur < GRANDEUR_MID", g["grandeur"] < GRANDEUR_MID, f"{g['grandeur']:.3f}")
    ok &= _check("band 为 low", g["band"] == "low", g["band"])
    ok &= _check("full_loud_share 被抑制到 0", g["parts"]["full_loud_share"] == 0.0,
                 f"{g['parts']['full_loud_share']:.3f}")

    # 真实曲目的动态范围应完全不受影响（实测最低 12.5 dB）。
    hero = dict(SNAPSHOTS["heroic_somber"])
    g2 = grandeur_score(hero)
    ok &= _check("英雄主义仍为 high", g2["band"] == "high", f"{g2['grandeur']:.3f}")
    ok &= _check("英雄主义 full_loud_share 未被抑制",
                 g2["parts"]["full_loud_share"] > 0.4,
                 f"{g2['parts']['full_loud_share']:.3f}")
    return ok


def main() -> int:
    print("=" * 66)
    print("四轴情绪判定测试")
    print("=" * 66)
    results = [
        test_grandeur_separates_heroic_from_intimate(),
        test_heroic_is_somber_not_sad(),
        test_intimate_sad_stays_sad(),
        test_epic_triumphant(),
        test_slow_major_still_calm(),
        test_high_attack_mid_scale(),
        test_negative_high_both_is_somber_majestic(),
        test_neutral_valence_refuses_direction(),
        test_band_boundaries(),
        test_missing_grandeur_features_flagged(),
        test_flat_audio_is_not_grand(),
    ]
    passed = sum(bool(r) for r in results)
    print("\n" + "=" * 66)
    print("结果：%d/%d 项通过" % (passed, len(results)))
    print("=" * 66)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
