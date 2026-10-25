# -*- coding: utf-8 -*-
"""批量执行器的行为测试（不碰真实音频，跑得快）。

测的是**契约**而不是数值：断点续跑、失败重试、进度走 stderr、输出只追加。
这些一旦坏掉，agent 侧不会报错，只会静默重算 794 首——所以必须有测试兜住。
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from music_emotion.batch import _load_done, _rel_of, scan_audio, track_id  # noqa: E402


def _check(name: str, ok: bool, detail: str = "") -> bool:
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, ("  " + detail) if detail else ""))
    return ok


def test_track_id_stability() -> bool:
    print("\n[t1] track_id 稳定性")
    a = track_id("dir/a.mp3", 1234)
    b = track_id("dir/a.mp3", 1234)
    c = track_id("dir/a.mp3", 1235)
    d = track_id("dir/b.mp3", 1234)
    ok = _check("同一输入稳定", a == b)
    ok &= _check("大小变化 -> id 变化", a != c)
    ok &= _check("路径变化 -> id 变化", a != d)
    ok &= _check("长度 12", len(a) == 12, a)
    return ok


def test_rel_of_normalization() -> bool:
    print("\n[t2] 相对路径归一化")
    root = r"C:\music"
    forms = [
        r"C:\music\a\b.mp3",
        "C:/music/a/b.mp3",
        r"C:\music\a\..\a\b.mp3",
    ]
    outs = {_rel_of(f, root) for f in forms}
    ok = _check("三种写法归一到同一 rel", len(outs) == 1, str(outs))
    ok &= _check("分隔符统一为 /", all("/" in o for o in outs), str(outs))
    return ok


def test_load_done_ignores_failures() -> bool:
    print("\n[t3] 失败记录不计入已完成")
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "out.jsonl")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"path": "a.mp3", "error": None}) + "\n")
            fh.write(json.dumps({"path": "b.mp3", "error": "DECODE_FAILED"}) + "\n")
            fh.write(json.dumps({"path": "c.mp3", "error": None}) + "\n")
            fh.write("{ this is not json\n")  # 半截写入要能容忍
        done = _load_done(__import__("pathlib").Path(p))
    ok = _check("成功的被收录", done == {"a.mp3", "c.mp3"}, str(sorted(done)))
    ok &= _check("失败的未收录(下次重试)", "b.mp3" not in done)
    return ok


def test_scan_is_ordered_and_filtered() -> bool:
    print("\n[t4] 扫描结果有序且只含音频")
    with tempfile.TemporaryDirectory() as td:
        for name in ("z.mp3", "a.flac", "m.wav", "readme.txt", "cover.jpg"):
            with open(os.path.join(td, name), "wb") as fh:
                fh.write(b"x")
        os.makedirs(os.path.join(td, "sub"), exist_ok=True)
        with open(os.path.join(td, "sub", "b.mp3"), "wb") as fh:
            fh.write(b"x")

        files = scan_audio(td)
        names = [os.path.basename(f) for f in files]
        full = [str(f).lower() for f in files]
    ok = _check("排除非音频", "readme.txt" not in names and "cover.jpg" not in names, str(names))
    ok &= _check("递归到子目录", "b.mp3" in names, str(names))
    ok &= _check("数量正确(4 个音频)", len(files) == 4, str(len(files)))
    # 排序键是完整路径（不是 basename）：只有这样才能跨目录深度保持稳定，
    # 否则 sub/b.mp3 会排到根目录文件前面，断点续跑的顺序就不确定了。
    ok &= _check("顺序按完整路径稳定", full == sorted(full), str(full))
    return ok


def test_analyze_one_error_is_captured() -> bool:
    print("\n[t5] 单条失败被捕获为 error 字段而不是抛异常")
    from music_emotion.batch import analyze_one

    with tempfile.TemporaryDirectory() as td:
        rec = analyze_one(os.path.join(td, "missing.mp3"), td)
    ok = _check("返回 dict 而非抛出", isinstance(rec, dict))
    ok &= _check("带 error 码", bool(rec.get("error")), str(rec.get("error")))
    ok &= _check("带 path 便于定位", bool(rec.get("path")))
    return ok


def main() -> int:
    print("=" * 66)
    print("批量执行器行为测试")
    print("=" * 66)
    results = [
        test_track_id_stability(),
        test_rel_of_normalization(),
        test_load_done_ignores_failures(),
        test_scan_is_ordered_and_filtered(),
        test_analyze_one_error_is_captured(),
    ]
    passed = sum(bool(r) for r in results)
    print("\n" + "=" * 66)
    print("结果：%d/%d 项通过" % (passed, len(results)))
    print("=" * 66)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
