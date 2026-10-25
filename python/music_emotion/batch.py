# -*- coding: utf-8 -*-
"""曲库级批量分析。

设计约束（都是给 agent 用带来的）：

1. **断点续跑**：794 首跑 1 小时，进程被杀是常态。已有结果按 path 去重，
   重跑只处理缺失项，绝不重算。
2. **单条失败不中断全局**：解码失败/损坏文件写进 error 记录，继续下一首。
   agent 需要的是"794 首里 791 首有结果"，不是第一首坏文件就整体崩掉。
3. **进度只走 stderr**：stdout 留给最终 JSON，agent 才能安全地重定向。
4. **子进程隔离**：librosa 在大文件上有内存峰值，进程池比线程池更能兜住，
   也让 Ctrl-C 能真正停下来。
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

__all__ = [
    "AUDIO_EXTS",
    "scan_audio",
    "track_id",
    "analyze_one",
    "run_batch",
]

#: 批量扫描时认可的扩展名。与 audio_io._SUPPORTED_EXT 保持一致，
#: 但这里显式排除仅在有 mpg123 时才可用的格式判断——scan 阶段不做解码试探。
AUDIO_EXTS = {".mp3", ".flac", ".wav", ".ogg", ".aiff", ".aif", ".au", ".w64", ".caf"}


def track_id(rel_path: str, size: int) -> str:
    """稳定 id：相对路径 + 字节数。改文件名或换码率都会变，这是有意的。"""
    h = hashlib.sha1(f"{rel_path}|{size}".encode("utf-8")).hexdigest()
    return h[:12]


def scan_audio(root: str, exts: Optional[Iterable[str]] = None) -> List[Path]:
    """递归扫描音频文件，按路径排序保证顺序稳定（断点续跑依赖它）。"""
    want = {e.lower() for e in (exts or AUDIO_EXTS)}
    root_p = Path(root)
    if not root_p.exists():
        raise FileNotFoundError(f"directory not found: {root}")
    out = [p for p in root_p.rglob("*") if p.is_file() and p.suffix.lower() in want]
    out.sort(key=lambda p: str(p).lower())
    return out


def analyze_one(
    path: str,
    root: str,
    segment_sec: float = 5.0,
    hop_sec: float = 1.0,
    max_seconds: float = 0.0,
    with_neural: bool = False,
    neural_model_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """分析单个文件，返回紧凑记录。异常在此吞掉并转成 error 字段。

    放在模块顶层是为了能被 multiprocessing 以 spawn 方式 pickle。
    """
    # 单进程内不要开多线程：进程池已经占满核心，OMP 再开线程只会互相抢。
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    from .pipeline import analyze_file
    from .types import MusicEmotionError

    p = Path(path)
    rel = _rel_of(str(p), root)

    try:
        size = p.stat().st_size
    except OSError:
        size = 0
    tid = track_id(rel, size)

    t0 = time.time()
    try:
        report = analyze_file(
            path,
            max_seconds=max_seconds,
            segment_sec=segment_sec,
            hop_sec=hop_sec,
            with_neural=with_neural,
            neural_model_dir=neural_model_dir,
        )
        d = report.to_dict(include_segments=False)
        feats = d.get("features") or {}
        emo = d.get("emotion") or {}
        return {
            "id": tid,
            "path": rel,
            "error": None,
            # 离散情绪词（悲伤/欢快/平静/紧张…）—— 这是给人的主字段
            "emotion": emo.get("emotion"),
            "emotion_id": emo.get("emotion_id"),
            "emotion_detail": emo.get("detail"),
            "emotion_intensity": emo.get("intensity"),
            "valence": emo.get("valence"),
            "arousal": emo.get("arousal"),
            # 四象限 id 保留给程序判断，与 emotion 同源不冲突
            "label": emo.get("label"),
            "arc": d.get("arc"),
            "confidence": d["quality"]["confidence"],
            "duration_sec": d["clip"]["duration_sec"],
            "n_segments": d["quality"]["n_segments"],
            # 保留少量可用于排序/分组的原始特征，避免 agent 再跑一遍
            "tempo_bpm": round(float(feats.get("tempo_bpm", 0.0)), 1),
            "key": feats.get("key"),
            "mode_score": round(float(feats.get("mode_score", 0.0)), 4),
            "rms_db": round(float(feats.get("rms_db", 0.0)), 2),
            "elapsed_sec": round(time.time() - t0, 2),
        }
    except MusicEmotionError as exc:
        return {
            "id": tid, "path": rel, "error": exc.code,
            "message": exc.message, "suggestion": exc.suggestion,
            "elapsed_sec": round(time.time() - t0, 2),
        }
    except Exception as exc:  # noqa: BLE001 - 批量场景绝不能被单条异常打断
        return {
            "id": tid, "path": rel, "error": "INTERNAL_ERROR",
            "message": f"{type(exc).__name__}: {exc}",
            "elapsed_sec": round(time.time() - t0, 2),
        }


def _rel_of(path: str, root: str) -> str:
    """统一相对路径口径。

    必须先把两边都 normpath/casefold，否则同一个文件会算出不同的 rel：
    `D:/a/b.mp3` 与 `D:\\a\\b.mp3`、以及大小写差异，都会让断点续跑失效。
    """
    try:
        rel = os.path.relpath(os.path.abspath(path), os.path.abspath(root))
    except ValueError:
        rel = os.path.basename(path)
    return rel.replace("\\", "/")


def _load_done(out_path: Path) -> set:
    """已成功分析过的 path 集合。失败记录不算完成，下次会重试。"""
    done = set()
    if not out_path.exists():
        return done
    with open(out_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not rec.get("error") and rec.get("path"):
                done.add(rec["path"])
    return done


def run_batch(
    root: str,
    out_path: str,
    workers: int = 4,
    limit: int = 0,
    segment_sec: float = 5.0,
    hop_sec: float = 1.0,
    max_seconds: float = 0.0,
    with_neural: bool = False,
    neural_model_dir: Optional[str] = None,
    resume: bool = True,
    log: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """跑完整个目录，逐条追加写入 out_path，返回汇总。"""
    say = log or (lambda _m: None)
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    files = scan_audio(root)
    root_abs = os.path.abspath(root)

    done = _load_done(out_p) if resume else set()
    if not resume and out_p.exists():
        out_p.unlink()

    todo = []
    for f in files:
        rel = _rel_of(str(f), root_abs)
        if rel not in done:
            todo.append((str(f), root_abs))
    if limit:
        todo = todo[:limit]

    say(f"scanned={len(files)} already_done={len(done)} todo={len(todo)} workers={workers}")
    if not todo:
        return {
            "scanned": len(files), "done_before": len(done),
            "processed": 0, "ok": 0, "failed": 0,
            "out": str(out_p), "elapsed_sec": 0.0,
        }

    t_all = time.time()
    ok = 0
    failed = 0
    err_counter: Dict[str, int] = {}

    from concurrent.futures import ProcessPoolExecutor, as_completed

    with open(out_p, "a", encoding="utf-8") as fh:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {
                ex.submit(
                    analyze_one, path, r, segment_sec, hop_sec,
                    max_seconds, with_neural, neural_model_dir,
                ): path
                for path, r in todo
            }
            for i, fut in enumerate(as_completed(futs), 1):
                src = futs[fut]
                try:
                    rec = fut.result()
                except Exception as exc:  # noqa: BLE001 - worker 崩溃也不放弃整批
                    rec = {
                        "path": os.path.basename(src),
                        "error": "WORKER_CRASH",
                        "message": f"{type(exc).__name__}: {exc}",
                    }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()  # 立刻落盘，进程被杀也不丢已完成的

                if rec.get("error"):
                    failed += 1
                    err_counter[rec["error"]] = err_counter.get(rec["error"], 0) + 1
                    say(f"  [{i}/{len(todo)}] FAIL {rec['error']} {os.path.basename(src)[:50]}")
                else:
                    ok += 1
                    if i % 10 == 0 or i == len(todo):
                        el = time.time() - t_all
                        eta = (len(todo) - i) * el / max(i, 1) / 60
                        say(
                            f"  [{i}/{len(todo)}] {el/60:.1f}m eta {eta:.1f}m  "
                            f"{rec['valence']:+.2f}/{rec['arousal']:+.2f} "
                            f"{rec['label']:<10s} {os.path.basename(src)[:40]}"
                        )

    return {
        "scanned": len(files),
        "done_before": len(done),
        "processed": len(todo),
        "ok": ok,
        "failed": failed,
        "error_breakdown": err_counter,
        "out": str(out_p),
        "elapsed_sec": round(time.time() - t_all, 1),
    }
