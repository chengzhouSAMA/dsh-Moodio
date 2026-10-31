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

    # Build the three spellings from a real directory instead of hard-coding a
    # Windows drive path. The previous version used r"C:\music", which on a
    # POSIX runner is a *relative* path: os.path.abspath prepends the cwd and
    # relpath then returns "../C:/music/...", so CI failed while local Windows
    # runs passed. Deriving the forms from a temp directory keeps the assertion
    # meaningful on both platforms.
    with tempfile.TemporaryDirectory() as td:
        root = os.path.join(td, "music")
        target = os.path.join(root, "a", "b.mp3")
        forms = [
            target,                                            # native separators
            target.replace(os.sep, "/"),                        # forward slashes
            os.path.join(root, "a", "..", "a", "b.mp3"),        # contains ..
        ]
        outs = {_rel_of(f, root) for f in forms}

    ok = _check("三种写法归一到同一 rel", len(outs) == 1, str(outs))
    ok &= _check("归一结果为 a/b.mp3", outs == {"a/b.mp3"}, str(outs))
    ok &= _check("分隔符统一为 /", all("\\" not in o for o in outs), str(outs))
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


def test_cli_defaults_and_invalid_limit() -> bool:
    print("\n[t6] CLI 默认时间窗与负 limit 校验")
    from moodio import CliError, build_parser, cmd_batch

    parser = build_parser()
    analyze = parser.parse_args(["analyze", "--path", __file__])
    batch = parser.parse_args(["batch", "--path", os.path.dirname(__file__)])
    ok = _check("analyze 默认 10s/10s", (analyze.segment_sec, analyze.hop_sec) == (10.0, 10.0))
    ok &= _check("batch 默认 10s/10s", (batch.segment_sec, batch.hop_sec) == (10.0, 10.0))

    batch.limit = -1
    try:
        cmd_batch(batch)
    except CliError as exc:
        ok &= _check("负 limit 返回参数错误", exc.code == "INVALID_ARGUMENT" and exc.exit_code == 2)
    else:
        ok &= _check("负 limit 返回参数错误", False)
    return ok


def test_human_fields_does_not_crash() -> bool:
    print("\n[t7] --human 配合 --fields 不丢失必需数据")
    import io
    from contextlib import redirect_stdout
    from types import SimpleNamespace
    from unittest.mock import patch

    from moodio import build_parser, cmd_analyze

    payload = {
        "emotion": {"emotion": "悲伤", "valence": -0.2, "arousal": -0.1},
        "clip": {"path": __file__, "duration_sec": 12.0},
        "quality": {"confidence": 0.5},
        "emotion_4d": {},
        "arc": "stable",
    }
    args = build_parser().parse_args([
        "analyze", "--path", __file__, "--human", "--fields", "arc",
    ])
    output = io.StringIO()
    fake_report = SimpleNamespace(to_dict=lambda **_kwargs: payload)
    with patch("music_emotion.pipeline.analyze_file", return_value=fake_report):
        with redirect_stdout(output):
            code = cmd_analyze(args)
    return _check("仍输出完整情绪坐标", code == 0 and "valence -0.20" in output.getvalue())


def test_online_preflight_matches_pending_tracks() -> bool:
    print("\n[t8] 批量联网清单与实际处理曲目一致")
    from music_emotion.batch import pending_audio, run_batch

    with tempfile.TemporaryDirectory() as td:
        root = os.path.join(td, "songs")
        os.makedirs(root)
        for name in ("a.mp3", "b.mp3"):
            with open(os.path.join(root, name), "wb") as fh:
                fh.write(b"x")
        out = os.path.join(td, "out.jsonl")
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"path": "a.mp3", "error": None}) + "\n")
        ok = _check("续跑只列出未分析曲目", pending_audio(root, out)["pending"] == ["b.mp3"])
        from contextlib import redirect_stdout
        from io import StringIO
        from moodio import main as moodio_main
        stream = StringIO()
        with redirect_stdout(stream):
            listed_code = moodio_main(["batch", "--path", root, "--out", out, "--list-only"])
        ok &= _check("CLI 只列出同一批待处理歌曲",
                     listed_code == 0 and json.loads(stream.getvalue())["pending"] == ["b.mp3"])
        try:
            run_batch(root, out, workers=1, resume=False, preflight_files=["wrong.mp3"])
        except ValueError as exc:
            ok &= _check("曲目集合变化时拒绝分析", "preflight" in str(exc).lower())
        else:
            ok &= _check("曲目集合变化时拒绝分析", False)
        with open(out, "r", encoding="utf-8") as fh:
            ok &= _check("拒绝时不清除原结果", "a.mp3" in fh.read())
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
        test_cli_defaults_and_invalid_limit(),
        test_human_fields_does_not_crash(),
        test_online_preflight_matches_pending_tracks(),
    ]
    passed = sum(bool(r) for r in results)
    print("\n" + "=" * 66)
    print("结果：%d/%d 项通过" % (passed, len(results)))
    print("=" * 66)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
