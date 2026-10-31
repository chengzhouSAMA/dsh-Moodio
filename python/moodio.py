# -*- coding: utf-8 -*-
"""moodio: 给 agent 用的音乐情感分析 CLI。

默认输出 JSON（agent 模式），--human 才给人看。
错误永远走 stderr + 非零退出码，绝不交互。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

# stdout 只放数据；日志与第三方库噪声一律压到 stderr 或丢弃。
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# ---------------------------------------------------------------- 约定

VERSION = "2.2.0"
AGENT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent")

EXIT_OK = 0
EXIT_GENERAL = 1
EXIT_USAGE = 2
EXIT_AUTH = 10
EXIT_PERMISSION = 11
EXIT_NOT_FOUND = 20
EXIT_CONFLICT = 30


class CliError(Exception):
    def __init__(self, code: str, message: str, suggestion: str = "", exit_code: int = EXIT_GENERAL):
        super().__init__(message)
        self.code = code
        self.message = message
        self.suggestion = suggestion
        self.exit_code = exit_code


def _emit_error(err: CliError) -> int:
    payload = {
        "error": True,
        "code": err.code,
        "message": err.message,
        "suggestion": err.suggestion,
    }
    sys.stderr.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return err.exit_code


def _read_agent_file(*parts: str) -> str:
    path = os.path.join(AGENT_DIR, *parts)
    if not os.path.isfile(path):
        return ""
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read().strip()


def _strip_frontmatter(text: str) -> str:
    """去掉 agent/*.md 的 YAML frontmatter，只留正文。

    frontmatter 是给人类维护文档用的元数据，agent 拿到的是纯正文更有用——
    否则 --brief 返回的开头是 `---\\nname: brief\\n...` 这些噪声。
    """
    if not text.startswith("---"):
        return text
    end = text.find("\n---", 3)
    if end == -1:
        return text
    return text[end + 4:].strip()


def _skill_index() -> List[Dict[str, str]]:
    """技能索引：只给 name + description + 取用命令，正文按需再取。

    这是"渐进披露"：把 6KB 的技能正文塞进每条响应是不划算的，
    agent 需要时用 `moodio skills <name>` 取。
    """
    out = []
    skills_dir = os.path.join(AGENT_DIR, "skills")
    if not os.path.isdir(skills_dir):
        return out
    for name in sorted(os.listdir(skills_dir)):
        if not name.endswith(".md"):
            continue
        stem = os.path.splitext(name)[0]
        text = _read_agent_file("skills", name)
        desc = ""
        for line in text.splitlines():
            if line.lower().startswith("description:"):
                desc = line.split(":", 1)[1].strip()
                break
        out.append({
            "name": stem,
            "description": desc,
            "command": f"moodio skills {stem}",
        })
    return out


def _agent_context(include_rules: bool = False) -> Dict[str, Any]:
    """响应附带的上下文。

    默认**不**注入规则正文。此前每条响应都带 2119 字符（约 660 token）的
    rules 全文，agent 连续分析 20 首就白烧 13000 token，而其中真正需要
    "随时可见"的只有触发条件与退出码。规则正文改为 `moodio --help` 与
    `moodio rules <name>` 按需获取。
    """
    ctx: Dict[str, Any] = {
        "skills": _skill_index(),
        "issue": "Report problems with: moodio issue create --type bug --message \"...\"",
    }
    if include_rules:
        rules = []
        rules_dir = os.path.join(AGENT_DIR, "rules")
        if os.path.isdir(rules_dir):
            for name in sorted(os.listdir(rules_dir)):
                if name.endswith(".md"):
                    rules.append({
                        "name": os.path.splitext(name)[0],
                        "content": _strip_frontmatter(_read_agent_file("rules", name)),
                    })
        ctx["rules"] = rules
    return ctx


def _print_json(payload: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


# ---------------------------------------------------------------- 命令

def cmd_analyze(args: argparse.Namespace) -> int:
    from music_emotion.audio_io import ANALYSIS_SR
    from music_emotion.pipeline import analyze_file
    from music_emotion.types import MusicEmotionError

    if args.segment_sec <= 0:
        raise CliError("INVALID_ARGUMENT", "--segment-sec must be > 0", "try --segment-sec 10", EXIT_USAGE)
    if args.hop_sec <= 0:
        raise CliError("INVALID_ARGUMENT", "--hop-sec must be > 0", "try --hop-sec 10", EXIT_USAGE)
    if not os.path.exists(args.path):
        raise CliError(
            "FILE_NOT_FOUND",
            f"audio file not found: {args.path}",
            "check the path; quote it if it contains spaces or CJK characters",
            EXIT_NOT_FOUND,
        )

    # 参数在跑分析之前就校验，避免"跑完 20 秒才告诉你字段名写错"
    wanted = _validate_fields(args.fields)

    try:
        report = analyze_file(
            args.path,
            sr=ANALYSIS_SR,
            max_seconds=args.max_seconds,
            segment_sec=args.segment_sec,
            hop_sec=args.hop_sec,
            with_neural=bool(args.neural) and not args.no_neural,
            neural_model_dir=args.neural_model_dir,
            with_axes_v2=bool(args.axes_v2),
        )
    except MusicEmotionError as exc:
        raise CliError(exc.code, exc.message, exc.suggestion) from exc

    payload = report.to_dict(include_segments=not args.no_segments)
    # --human 走完整报告：字段裁剪只服务 agent 模式，否则 _print_human 会
    # 拿到被裁掉的 None（如 emo['valence']）并在格式化时炸成 INTERNAL_ERROR。
    if wanted and not args.human:
        # 用 "取值 + 补空" 而不是直接下标：`warnings`、`neural` 这类键在"没有内容"时
        # 根本不会出现在 payload 里，直接下标会抛 KeyError 并被顶层兜底成 INTERNAL_ERROR
        # —— 用户只是多问了一个合法字段，却收到"内部错误"。缺失时补明确的空值，
        # 让调用方看到 "这个维度确实为空" 而不是 "这个字段不存在"。
        absent_defaults = {"warnings": [], "neural": None, "segments": []}
        payload = {k: payload.get(k, absent_defaults.get(k)) for k in wanted}

    if args.human:
        _print_human(payload)
    else:
        payload.update(_agent_context())
        _print_json(payload)
    return EXIT_OK


# ---------------------------------------------------------------- batch

def cmd_batch(args: argparse.Namespace) -> int:
    """批量分析整个目录，断点续跑，逐条追加写入 JSONL。"""
    from music_emotion.batch import pending_audio, run_batch

    if not os.path.isdir(args.path):
        if os.path.exists(args.path):
            raise CliError(
                "NOT_A_DIRECTORY",
                f"--path must be a directory for batch mode: {args.path}",
                "use `analyze --path <file>` for a single file",
                EXIT_USAGE,
            )
        raise CliError(
            "DIR_NOT_FOUND",
            f"directory not found: {args.path}",
            "check the path; quote it if it contains spaces or CJK characters",
            EXIT_NOT_FOUND,
        )
    if args.segment_sec <= 0:
        raise CliError("INVALID_ARGUMENT", "--segment-sec must be > 0", "try --segment-sec 10", EXIT_USAGE)
    if args.hop_sec <= 0:
        raise CliError("INVALID_ARGUMENT", "--hop-sec must be > 0", "try --hop-sec 10", EXIT_USAGE)
    if args.workers < 1:
        raise CliError("INVALID_ARGUMENT", "--workers must be >= 1", "try --workers 4", EXIT_USAGE)
    if args.limit < 0:
        # 负 limit 会让 todo[:limit] 静默丢掉尾部曲目，必须在这里拦住。
        raise CliError("INVALID_ARGUMENT", "--limit must be >= 0", "0 means all pending tracks", EXIT_USAGE)

    out_path = args.out or os.path.join(os.getcwd(), "moodio_batch.jsonl")
    if args.list_only:
        _print_json(pending_audio(args.path, out_path, limit=args.limit, resume=not args.no_resume))
        return EXIT_OK
    preflight_files = None
    if args.preflight_files:
        try:
            preflight_files = json.loads(args.preflight_files)
        except json.JSONDecodeError as exc:
            raise CliError("INVALID_ARGUMENT", "--preflight-files must be a JSON array", exit_code=EXIT_USAGE) from exc
        if not isinstance(preflight_files, list) or not all(isinstance(p, str) for p in preflight_files):
            raise CliError("INVALID_ARGUMENT", "--preflight-files must be an array of paths", exit_code=EXIT_USAGE)

    def _log(msg: str) -> None:
        # 进度一律走 stderr：stdout 只留最终汇总 JSON，agent 才能安全重定向
        if not args.quiet:
            sys.stderr.write(msg + "\n")
            sys.stderr.flush()

    try:
        summary = run_batch(
            root=args.path,
            out_path=out_path,
            workers=args.workers,
            limit=args.limit,
            segment_sec=args.segment_sec,
            hop_sec=args.hop_sec,
            max_seconds=args.max_seconds,
            with_neural=bool(args.neural) and not args.no_neural,
            neural_model_dir=args.neural_model_dir,
            resume=not args.no_resume,
            log=_log,
            preflight_files=preflight_files,
            with_axes_v2=bool(args.axes_v2),
        )
    except FileNotFoundError as exc:
        raise CliError("DIR_NOT_FOUND", str(exc), "check --path", EXIT_NOT_FOUND) from exc
    except ValueError as exc:
        raise CliError("BATCH_PREFLIGHT_CHANGED", str(exc), "retry the lookup before batch analysis", EXIT_CONFLICT) from exc

    summary.update(_agent_context())
    _print_json(summary)
    # 有失败项时不要假装全成功：非零退出码 + 失败明细已在 JSON 里
    return EXIT_OK if summary.get("failed", 0) == 0 else EXIT_GENERAL


# ---------------------------------------------------------------- lyrics

def cmd_lyrics(args: argparse.Namespace) -> int:
    """仅在显式调用时用本地模型猜测唱词，不改变声学情绪模型。"""
    from music_emotion.lyrics import transcribe_lyrics
    from music_emotion.types import MusicEmotionError

    try:
        result = transcribe_lyrics(
            args.path, model_dir=args.model_dir, language=args.language, device=args.device,
        )
    except MusicEmotionError as exc:
        if exc.code in ("FILE_NOT_FOUND", "LYRICS_MODEL_NOT_FOUND"):
            exit_code = EXIT_NOT_FOUND
        elif exc.code == "INVALID_ARGUMENT":
            exit_code = EXIT_USAGE
        else:
            exit_code = EXIT_GENERAL
        raise CliError(exc.code, exc.message, exc.suggestion, exit_code) from exc
    _print_json(result)
    return EXIT_OK


# ---------------------------------------------------------------- report

def cmd_report(args: argparse.Namespace) -> int:
    """把 batch 产出的 JSONL 汇总成 Markdown + JSON 报告。"""
    from music_emotion import report as report_mod

    if not os.path.isfile(args.path):
        raise CliError(
            "FILE_NOT_FOUND",
            f"batch result file not found: {args.path}",
            "run `moodio batch --path <dir> --out <file>` first",
            EXIT_NOT_FOUND,
        )

    rows, errors = report_mod.load(args.path)
    if not rows:
        raise CliError(
            "NO_RECORDS",
            f"no successful records in {args.path}",
            "the batch may have failed entirely; inspect the JSONL error lines",
            EXIT_CONFLICT,
        )

    summary = report_mod.summarize(rows, errors)
    if args.calibration == "corpus-relative":
        calib = {"mode": "corpus-relative", "note": "0 点 = 本曲库中位数，跨库不可比"}
    else:
        calib = {"mode": "fixed", "note": "0 点 = 中性输入不动点，跨库可比"}
    summary["calibration"] = calib
    summary["source"] = os.path.abspath(args.path)

    md_path = args.out or os.path.splitext(args.path)[0] + "_report.md"
    json_path = os.path.splitext(md_path)[0] + ".json"

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(report_mod.render_markdown(summary, calib))

    payload = {
        "report_md": os.path.abspath(md_path),
        "report_json": os.path.abspath(json_path),
        "n_ok": summary["n_ok"],
        "n_failed": summary["n_failed"],
        "quadrants_present": summary["quadrants_present"],
        "sanity_tempo_arousal_r": summary["sanity_tempo_arousal_r"],
        "calibration": calib,
    }
    payload.update(_agent_context())
    _print_json(payload)
    return EXIT_OK



#: 合法顶层字段，用于参数预校验。
_TOP_FIELDS = (
    "arc", "clip", "deltas", "emotion", "emotion_4d", "texture", "grandeur",
    "features", "modes", "neural", "quality", "schema_version", "axes_v2",
    "segments", "warnings",
)


def _validate_fields(fields: str) -> List[str]:
    wanted = [f.strip() for f in (fields or "").split(",") if f.strip()]
    if not wanted:
        return []
    unknown = [f for f in wanted if f not in _TOP_FIELDS]
    if unknown:
        raise CliError(
            "UNKNOWN_FIELD",
            f"unknown field(s): {', '.join(unknown)}",
            "available: " + ", ".join(_TOP_FIELDS),
            EXIT_USAGE,
        )
    return wanted


def _print_human(payload: Dict[str, Any]) -> None:
    emo = payload.get("emotion", {})
    clip = payload.get("clip", {})
    q = payload.get("quality", {})
    print(f"{os.path.basename(clip.get('path') or '')}  ({clip.get('duration_sec')}s)")
    # 先给人类语言，再给坐标
    word = emo.get("emotion") or emo.get("label")
    detail = emo.get("detail")
    extra = f"（更贴近说：{detail}）" if detail and detail != word else ""
    print(f"  情绪：{word}{extra}  强度={emo.get('intensity')}  "
          f"magnitude={emo.get('magnitude')}")
    print(f"  valence {emo.get('valence'):+.2f}   arousal {emo.get('arousal'):+.2f}   "
          f"quadrant={emo.get('label')}")
    if emo.get("desc"):
        print(f"  {emo['desc']}")
    # 四轴判断：把"悲壮/悲伤"、"昂扬/欢快"分开的那一层
    e4 = payload.get("emotion_4d") or {}
    if e4:
        print(f"  四轴判断：{e4.get('emotion')}  "
              f"（唤醒={e4.get('arousal_band')}, 攻击性={e4.get('texture_band')}, "
              f"宏大度={e4.get('grandeur_band')}）")
        print(f"    texture={e4.get('texture')}  grandeur={e4.get('grandeur')}")
        if e4.get("desc"):
            print(f"    {e4['desc']}")
    print(f"  arc={payload.get('arc')}  confidence={q.get('confidence')}")
    segs = payload.get("segments")
    if isinstance(segs, list) and segs:
        print("  timeline:")
        for s in segs[:12]:
            bar = "#" * int(round((s["arousal"] + 1) * 10))
            print(f"    {s['start_sec']:6.1f}s A={s['arousal']:+.2f} V={s['valence']:+.2f} {bar}")
    warnings = payload.get("warnings") or []
    for w in warnings:
        print(f"  ! {w}")


def cmd_brief(args: argparse.Namespace) -> int:
    text = _read_agent_file("brief.md")
    if not text:
        raise CliError("MISSING_BRIEF", "agent/brief.md not found", "restore the agent/ directory")
    # 去掉 frontmatter：--brief 是"一句话自我介绍"，塞给 agent 时不该带 --- name: ---
    body = _strip_frontmatter(text)
    if args.human:
        print(body)
    else:
        _print_json({"brief": body})
    return EXIT_OK


def cmd_help(args: argparse.Namespace) -> int:
    # --help 是"完整自描述"，这里显式要求带上规则正文；
    # 普通命令响应不带（见 _agent_context 的注释）。
    ctx = _agent_context(include_rules=True)
    payload = {
        "help": {
            "name": "moodio",
            "version": VERSION,
            "summary": "Music emotion (valence/arousal) analysis for agents.",
            "usage": ("moodio analyze --path <audio> [options] | "
                      "moodio lyrics --path <audio> --model-dir <local-model> [options] | "
                      "moodio batch --path <dir> [options]"),
            "output_mode": "JSON by default; --human for readable text",
            "exit_codes": {
                "0": "success",
                "1": "general error / batch finished with failures",
                "2": "usage error",
                "10": "auth failed",
                "11": "permission denied",
                "20": "resource not found",
                "30": "conflict / precondition failed",
            },
        },
        "commands": _command_specs(),
        "rules": ctx.get("rules", []),
        "skills": ctx.get("skills", []),
        "issue": ctx["issue"],
    }
    _print_json(payload)
    return EXIT_OK


def _command_specs() -> List[Dict[str, Any]]:
    return [
        {
            "name": "analyze",
            "description": "Estimate valence/arousal, per-segment timeline and arc for one audio file.",
            "parameters": [
                {"name": "path", "flag": "--path", "type": "string", "required": True,
                 "description": "audio file path (mp3/flac/wav/ogg/aiff)"},
                {"name": "segment_sec", "flag": "--segment-sec", "type": "number", "required": False,
                 "default": 10.0, "description": "segment length in seconds"},
                {"name": "hop_sec", "flag": "--hop-sec", "type": "number", "required": False,
                 "default": 10.0, "description": "segment hop in seconds"},
                {"name": "max_seconds", "flag": "--max-seconds", "type": "number", "required": False,
                 "default": 0, "description": "analyze only the first N seconds (0 = all)"},
                {"name": "axes_v2", "flag": "--axes-v2", "type": "boolean", "required": False,
                 "description": "add independent 12-axis diagnostics over the full analyzed audio"},
                {"name": "no_neural", "flag": "--no-neural", "type": "boolean", "required": False,
                 "description": "skip the wav2small neural sidecar"},
                {"name": "no_segments", "flag": "--no-segments", "type": "boolean", "required": False,
                 "description": "omit the per-segment timeline (saves tokens)"},
                {"name": "fields", "flag": "--fields", "type": "string", "required": False,
                 "description": "comma-separated top-level fields to keep"},
            ],
        },
        {
            "name": "batch",
            "description": (
                "Analyze every audio file under a directory, appending one JSON record "
                "per track to a JSONL file. Resumable: re-running skips tracks that "
                "already succeeded; failed tracks are retried."
            ),
            "parameters": [
                {"name": "path", "flag": "--path", "type": "string", "required": True,
                 "description": "directory to scan recursively"},
                {"name": "out", "flag": "--out", "type": "string", "required": False,
                 "default": "./moodio_batch.jsonl",
                 "description": "JSONL output path (appended, never rewritten)"},
                {"name": "workers", "flag": "--workers", "type": "integer", "required": False,
                 "default": 4, "description": "parallel worker processes"},
                {"name": "limit", "flag": "--limit", "type": "integer", "required": False,
                 "default": 0, "description": "only process the first N pending tracks"},
                {"name": "max_seconds", "flag": "--max-seconds", "type": "number", "required": False,
                 "default": 0, "description": "analyze only the first N seconds per track"},
                {"name": "axes_v2", "flag": "--axes-v2", "type": "boolean", "required": False,
                 "description": "add independent 12-axis diagnostics to each record"},
                {"name": "no_resume", "flag": "--no-resume", "type": "boolean", "required": False,
                 "description": "discard previous results and start from scratch"},
                {"name": "quiet", "flag": "--quiet", "type": "boolean", "required": False,
                 "description": "suppress progress output on stderr"},
            ],
        },
        {
            "name": "lyrics",
            "description": "Transcribe singing from a local audio file; estimates may differ from official lyrics.",
            "parameters": [
                {"name": "path", "flag": "--path", "type": "string", "required": True,
                 "description": "audio file path"},
                {"name": "model_dir", "flag": "--model-dir", "type": "string", "required": False,
                 "description": "local model directory containing model.bin; or set MOODIO_LYRICS_MODEL_DIR"},
                {"name": "language", "flag": "--language", "type": "string", "required": False,
                 "description": "language code, or omit to auto-detect"},
                {"name": "device", "flag": "--device", "type": "string", "required": False,
                 "enum": ["cpu", "cuda"], "default": "cpu"},
            ],
        },
        {
            "name": "report",
            "description": (
                "Summarize a batch JSONL into a Markdown report plus machine-readable JSON: "
                "V/A distribution, quadrant shares, confidence, arc types, error breakdown."
            ),
            "parameters": [
                {"name": "path", "flag": "--path", "type": "string", "required": True,
                 "description": "JSONL produced by `batch`"},
                {"name": "out", "flag": "--out", "type": "string", "required": False,
                 "description": "markdown output path; JSON is written next to it"},
                {"name": "calibration", "flag": "--calibration", "type": "string", "required": False,
                 "enum": ["fixed", "corpus-relative"], "default": "fixed",
                 "description": "how to describe the zero point in the report"},
            ],
        },
        {
            "name": "brief",
            "description": "One-paragraph identity statement for agent config sync.",
            "parameters": [],
        },
        {
            "name": "skills",
            "description": "List or show extended capability docs.",
            "parameters": [
                {"name": "name", "flag": "positional", "type": "string", "required": False},
            ],
        },
        {
            "name": "issue",
            "description": "Local feedback log (create/list/show).",
            "parameters": [
                {"name": "action", "flag": "positional", "type": "string", "required": True,
                 "enum": ["create", "list", "show"]},
            ],
        },
    ]


def cmd_skills(args: argparse.Namespace) -> int:
    skills_dir = os.path.join(AGENT_DIR, "skills")
    names = []
    if os.path.isdir(skills_dir):
        names = sorted(os.path.splitext(n)[0] for n in os.listdir(skills_dir) if n.endswith(".md"))

    if not args.name:
        _print_json({"skills": [{"name": n, "command": f"moodio skills {n}"} for n in names]})
        return EXIT_OK

    if args.name not in names:
        raise CliError(
            "SKILL_NOT_FOUND",
            f"no such skill: {args.name}",
            "available: " + (", ".join(names) if names else "(none)"),
            EXIT_NOT_FOUND,
        )
    _print_json({"name": args.name, "content": _read_agent_file("skills", args.name + ".md")})
    return EXIT_OK


# ---------------------------------------------------------------- issue

def _issue_path() -> str:
    base = os.environ.get("MOODIO_HOME") or os.path.join(os.path.expanduser("~"), ".moodio")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, "issues.json")


def _load_issues() -> List[Dict[str, Any]]:
    path = _issue_path()
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _save_issues(items: List[Dict[str, Any]]) -> None:
    with open(_issue_path(), "w", encoding="utf-8") as fh:
        json.dump(items, fh, ensure_ascii=False, indent=2)


def cmd_issue(args: argparse.Namespace) -> int:
    import datetime

    items = _load_issues()

    if args.action == "list":
        _print_json({"issues": items, "count": len(items), "path": _issue_path()})
        return EXIT_OK

    if args.action == "show":
        if not args.id:
            raise CliError("INVALID_ARGUMENT", "--id is required for show", "moodio issue show --id 1", EXIT_USAGE)
        for it in items:
            if it.get("id") == args.id:
                _print_json(it)
                return EXIT_OK
        raise CliError("ISSUE_NOT_FOUND", f"no issue with id {args.id}", "run: moodio issue list", EXIT_NOT_FOUND)

    # create
    if not args.message:
        raise CliError("INVALID_ARGUMENT", "--message is required for create", "add --message \"...\"", EXIT_USAGE)
    allowed = {"bug", "requirement", "suggestion", "bad-output"}
    if args.type not in allowed:
        raise CliError(
            "INVALID_ARGUMENT",
            f"--type must be one of {sorted(allowed)}",
            "e.g. --type bad-output",
            EXIT_USAGE,
        )

    now = datetime.datetime.now().isoformat(timespec="seconds")
    new_id = max([it.get("id", 0) for it in items], default=0) + 1
    item = {
        "id": new_id,
        "type": args.type,
        "status": "open",
        "message": args.message,
        "version": VERSION,
        "created_at": now,
        "updated_at": now,
    }
    items.append(item)
    _save_issues(items)
    _print_json(item)
    return EXIT_OK


# ---------------------------------------------------------------- 参数

class _Parser(argparse.ArgumentParser):
    """argparse 默认 exit(2) + 人类文本 stderr，这里改成结构化 JSON。"""

    def error(self, message: str):  # type: ignore[override]
        raise CliError("INVALID_ARGUMENT", message, "run: moodio --help", EXIT_USAGE)


def build_parser() -> argparse.ArgumentParser:
    p = _Parser(prog="moodio", description="Music emotion analysis for agents",
                add_help=False)
    p.add_argument("--version", action="store_true", help="print version and exit")
    p.add_argument("--help", action="store_true", dest="full_help", help="JSON self-description")
    p.add_argument("--brief", action="store_true", help="one-paragraph identity")
    p.add_argument("--human", action="store_true", help="human-readable output")
    p.add_argument("--agent", action="store_true", help="explicit agent mode (default)")

    sub = p.add_subparsers(dest="command")

    a = sub.add_parser("analyze", add_help=False)
    a.add_argument("--path", required=True)
    # 10s 窗口/步长与工具描述、README 一致；需要更细的时间线可显式传参。
    a.add_argument("--segment-sec", type=float, default=10.0)
    a.add_argument("--hop-sec", type=float, default=10.0)
    a.add_argument("--max-seconds", type=float, default=0.0)
    a.add_argument("--axes-v2", action="store_true", help="append optional 12-axis full-audio diagnostics")
    a.add_argument("--no-neural", action="store_true", help="kept for compatibility; neural is off by default")
    a.add_argument("--neural", action="store_true", help="also run the wav2small sidecar")
    a.add_argument("--no-segments", action="store_true")
    a.add_argument("--neural-model-dir", default=None)
    a.add_argument("--fields", default="")
    a.add_argument("--human", action="store_true")
    a.add_argument("--agent", action="store_true")

    ly = sub.add_parser("lyrics", add_help=False)
    ly.add_argument("--path", required=True)
    ly.add_argument("--model-dir", default=None)
    ly.add_argument("--language", default=None)
    ly.add_argument("--device", choices=["cpu", "cuda"], default="cpu")

    s = sub.add_parser("skills", add_help=False)
    s.add_argument("name", nargs="?", default="")
    s.add_argument("--human", action="store_true")

    i = sub.add_parser("issue", add_help=False)
    i.add_argument("action", nargs="?", default="list")
    i.add_argument("--id", type=int, default=0)
    i.add_argument("--type", default="bug")
    i.add_argument("--message", default="")

    b = sub.add_parser("brief", add_help=False)
    b.add_argument("--human", action="store_true")

    bt = sub.add_parser("batch", add_help=False)
    bt.add_argument("--path", required=True)
    bt.add_argument("--out", default="")
    bt.add_argument("--workers", type=int, default=4)
    # limit 必须显式校验：todo[:limit] 对负数会静默丢尾部曲目。
    bt.add_argument("--limit", type=int, default=0)
    bt.add_argument("--segment-sec", type=float, default=10.0)
    bt.add_argument("--hop-sec", type=float, default=10.0)
    bt.add_argument("--max-seconds", type=float, default=0.0)
    bt.add_argument("--axes-v2", action="store_true")
    bt.add_argument("--no-neural", action="store_true",
                    help="kept for compatibility; neural is off by default")
    bt.add_argument("--neural", action="store_true", help="also run the wav2small sidecar")
    bt.add_argument("--neural-model-dir", default=None)
    bt.add_argument("--no-resume", action="store_true")
    bt.add_argument("--list-only", action="store_true")
    bt.add_argument("--preflight-files", default="")
    bt.add_argument("--quiet", action="store_true")
    bt.add_argument("--human", action="store_true")
    bt.add_argument("--agent", action="store_true")

    rp = sub.add_parser("report", add_help=False)
    rp.add_argument("--path", required=True, help="batch JSONL output")
    rp.add_argument("--out", default="", help="markdown output path")
    rp.add_argument("--calibration", default="fixed",
                    choices=["fixed", "corpus-relative"])
    rp.add_argument("--human", action="store_true")
    rp.add_argument("--agent", action="store_true")

    return p


def main(argv: List[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()

    try:
        args = parser.parse_args(argv)
    except CliError as exc:
        return _emit_error(exc)

    try:
        if getattr(args, "version", False):
            _print_json({"version": VERSION})
            return EXIT_OK
        if getattr(args, "full_help", False):
            return cmd_help(args)
        if getattr(args, "brief", False):
            return cmd_brief(args)

        command = getattr(args, "command", None)
        if command == "analyze":
            return cmd_analyze(args)
        if command == "batch":
            return cmd_batch(args)
        if command == "lyrics":
            return cmd_lyrics(args)
        if command == "report":
            return cmd_report(args)
        if command == "skills":
            return cmd_skills(args)
        if command == "issue":
            return cmd_issue(args)
        if command == "brief":
            return cmd_brief(args)

        return cmd_help(args)
    except CliError as exc:
        return _emit_error(exc)
    except KeyboardInterrupt:
        return _emit_error(CliError("INTERRUPTED", "interrupted by user", "retry or reduce --max-seconds"))
    except Exception as exc:  # noqa: BLE001 - 顶层兜底，保证不裸抛 traceback 给 agent
        return _emit_error(
            CliError("INTERNAL_ERROR", f"{type(exc).__name__}: {exc}",
                     "rerun with --max-seconds 30 to isolate; report via: moodio issue create")
        )


if __name__ == "__main__":
    raise SystemExit(main())
