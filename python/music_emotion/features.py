# -*- coding: utf-8 -*-
"""声学特征提取。

只提取“有心理学依据、且可稳定计算”的特征，每个特征都对应一个可解释的情感通道：

  唤醒度(arousal)相关
    - tempo_bpm           速度：快→高唤醒
    - onset_rate          起音密度：音事件越密→越活跃
    - rms_db / loudness   能量：响→高唤醒
    - spectral_flux       谱通量：变化快→高唤醒
    - pulse_clarity       节拍规律性：越规律→越有驱动力
    - percussive_ratio    打击感占比

  效价(valence)相关
    - mode_score          大调/小调倾向（色度与模板的相关差）
    - harmonic_complexity 和声复杂度：越复杂→效价越负
    - spectral_centroid   谱质心（明亮度），与效价弱正相关
    - key_clarity         调性清晰度

特征全部来自 librosa，纯确定性，无网络依赖。
"""
from __future__ import annotations

import warnings
from typing import Any, Dict

import numpy as np

from .types import MusicEmotionError

warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

#: Krumhansl-Kessler 调性模板（大调 / 小调）。
_KK_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
_KK_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)

_PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _safe(value: float, default: float = 0.0) -> float:
    return float(value) if np.isfinite(value) else default


def _estimate_mode(chroma_mean: np.ndarray) -> Dict[str, Any]:
    """用 KK 模板做 24 个调的相关匹配，返回大/小调证据。"""
    x = chroma_mean / (chroma_mean.sum() + 1e-9)
    best_major = (-2.0, None, -1.0)
    best_minor = (-2.0, None, -1.0)

    for shift in range(12):
        rotated = np.roll(x, shift)
        r_major = float(np.corrcoef(rotated, _KK_MAJOR)[0, 1])
        r_minor = float(np.corrcoef(rotated, _KK_MINOR)[0, 1])
        if np.isfinite(r_major) and r_major > best_major[0]:
            best_major = (r_major, shift, r_minor)
        if np.isfinite(r_minor) and r_minor > best_minor[0]:
            best_minor = (r_minor, shift, r_major)

    # 色度完全退化时（纯静音窗口：chroma 全零 -> 归一化后仍是全零 -> corrcoef 全 nan），
    # 上面两处 isfinite 检查会把所有候选拒掉，best_* 停留在初始值 (…, None, …)，
    # 下一步 int(None) 直接抛 TypeError。这会**让整首歌分析失败**，
    # 而触发条件只是音频尾部有 5 秒以上静音（实测 Silver Scrapes.mp3 尾部 10 个窗口全静音）。
    # 正确行为是"判不出调"，而不是崩掉：返回 unknown + 零证据。
    if best_major[1] is None or best_minor[1] is None:
        return {
            "mode": "unknown",
            "mode_score": 0.0,
            "key": "unknown",
            "key_clarity": 0.0,
        }

    if best_major[0] >= best_minor[0]:
        key_idx = int(best_major[1])
        mode = "major"
        key_name = _PITCH_NAMES[key_idx % 12] + " major"
        clarity = best_major[0]
    else:
        key_idx = int(best_minor[1])
        mode = "minor"
        key_name = _PITCH_NAMES[key_idx % 12] + " minor"
        clarity = best_minor[0]

    # mode_score 的符号必须统一：正 = 偏大调，负 = 偏小调。
    # 修正记录：早期实现小调分支用 best_minor[0]-best_minor[2]（同一根音下的对照），
    # 而判定用 best_major[0]>=best_minor[0]（全局最优），两者口径不一致，
    # 导致**小调被算成正分**、反而抬高效价，方向完全反了。
    mode_score = float(best_major[0] - best_minor[0])

    return {
        "mode": mode,
        "mode_score": _safe(mode_score),
        "key": key_name,
        "key_clarity": _safe(clarity),
    }


def extract_features(y: np.ndarray, sr: int, segment_sec: float = 5.0,
                     hop_sec: float = 1.0) -> Dict[str, Any]:
    """提取全局特征与逐窗特征。

    Args:
        y: 单声道波形。
        sr: 采样率。
        segment_sec: 情感分段长度（秒）。
        hop_sec: 分段滑移步长（秒）。

    Returns:
        {"global": {...标量特征...}, "segments": [{"start_sec":.., "features": {...}}, ...]}
    """
    import librosa

    if y.size < sr * 0.5:
        raise MusicEmotionError(
            "AUDIO_TOO_SHORT",
            f"audio is {y.size / sr:.2f}s, need at least 0.5s",
            "provide a longer clip or raise --max-seconds",
        )

    # --- 全局基础量 ---
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    rms_db = 20.0 * np.log10(float(np.mean(rms)) + 1e-9)
    peak = float(np.abs(y).max())

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    onset_times = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=512, backtrack=False
    )
    onset_rate = float(len(onset_times)) / (len(y) / sr)

    tempo, beats = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr, hop_length=512, units="frames"
    )
    tempo_bpm = float(np.atleast_1d(tempo)[0])
    pulse_clarity = _pulse_clarity(onset_env, beats)

    cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=512)[0]
    flux = _spectral_flux(librosa.stft(y, n_fft=2048, hop_length=512))

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)
    chroma_mean = chroma.mean(axis=1)
    mode_info = _estimate_mode(chroma_mean)

    # 谐波/打击分离 -> 打击感占比。整条音频只算一次，窗口直接切片复用。
    perc_env = _percussive_envelope(y)
    energy_h = float(np.sum(y ** 2)) + 1e-9
    energy_p = float(np.sum(perc_env ** 2)) + 1e-9
    percussive_ratio = energy_p / (energy_h + energy_p)

    # 和声复杂度：色度熵（音级越分散越复杂）+ 频谱平坦度
    chroma_norm = chroma_mean / (chroma_mean.sum() + 1e-9)
    chroma_entropy = float(-np.sum(chroma_norm * np.log(chroma_norm + 1e-12)))
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=y, hop_length=512)[0]))
    harmonic_complexity = _norm(chroma_entropy, 0.0, np.log(12.0)) * 0.7 + _norm(flatness, 0.0, 0.5) * 0.3

    # --- 宏大度（grandeur）线索 ---
    # 与"攻击性"(texture) 分开度量：管弦/史诗类的力量来自规模而非粗糙度。
    # 详见 texture.py 顶部关于《英雄主义》误判的说明。
    #
    # 动态范围用分位数差而不是 max-min：单个爆音不应定义"动态"。
    _rms_db_frames = librosa.amplitude_to_db(
        librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0] + 1e-9, ref=np.max
    )
    dyn_range = float(np.percentile(_rms_db_frames, 95) - np.percentile(_rms_db_frames, 5))
    # "满响时间占比"：以本条自身的 p95 为基准，统计落在其 3dB 以内的帧比例。
    # 这是"持续用力"最直接的证据——诀别书 25.7% vs 英雄主义 40.9%。
    _p95 = float(np.percentile(_rms_db_frames, 95))
    full_loud = float((_rms_db_frames > (_p95 - 3.0)).mean() * 100.0)
    # 注意：librosa 的 spectral_contrast 内部会自行 amplitude_to_db，
    # 所以必须传**幅度谱**，不能先取对数再传（二次取对数会把值压到 0）。
    # 实测同一首：传 |S| = 27.77；传 amplitude_to_db(|S|,ref=max) = 0.0。
    _contrast_frames = librosa.feature.spectral_contrast(
        S=np.abs(librosa.stft(y, n_fft=2048, hop_length=512)),
        sr=sr,
    )
    contrast_global = float(np.mean(_contrast_frames))
    active_pitches = int((chroma_norm > 0.05).sum())

    global_feats: Dict[str, Any] = {
        "tempo_bpm": _safe(tempo_bpm),
        "onset_rate": _safe(onset_rate),
        "rms_db": _safe(rms_db),
        "peak": _safe(peak),
        "pulse_clarity": _safe(pulse_clarity),
        "spectral_centroid_hz": _safe(float(np.mean(cent))),
        "spectral_rolloff_hz": _safe(float(np.mean(rolloff))),
        "spectral_flux": _safe(float(np.mean(flux))),
        "percussive_ratio": _safe(percussive_ratio),
        "harmonic_complexity": _safe(harmonic_complexity),
        "chroma_entropy": _safe(chroma_entropy),
        "spectral_flatness": _safe(flatness),
        # 宏大度线索
        "dynamic_range_db": _safe(dyn_range),
        "full_loud_share": _safe(full_loud),
        "spectral_contrast": _safe(contrast_global),
        "active_pitches": float(active_pitches),
        "mode": mode_info["mode"],
        "mode_score": mode_info["mode_score"],
        "key": mode_info["key"],
        "key_clarity": mode_info["key_clarity"],
    }

    # --- 分段特征 ---
    # 性能说明（四次修正，均基于实测而非推测）：
    # 1) 早期每窗口调用 librosa.effects.hpss -> 改为整条算一次后切片。
    #    中间还曾因残留一行重复调用，使 HPSS 被算了两次（22s × 2）。
    # 2) onset_strength(790ms/窗) 与 beat_track(892ms/窗) 对 6s 短窗的常数开销
    #    远大于数据量，改为全局算一次后按帧号切片。
    # 3) 窗口调式一度改用 chroma_stft 提速，但 mode_score 幅度被压缩
    #    （-0.106 -> -0.022），与全局的 CQT 尺度不一致，已回退。
    # 4) 最终做法：CQT / STFT / 谱平坦度全部**整条算一次**，窗口按帧切片求均值。
    #    这样既没有逐窗重算的开销，又保证窗口与全局用的是同一套尺度。
    seg_len = int(segment_sec * sr)
    hop_len = int(hop_sec * sr)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    _, all_beats = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr, hop_length=512, units="frames"
    )
    all_beats = np.asarray(all_beats).ravel()

    mag = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    flux_frames = _spectral_flux(mag)
    flat_frames = librosa.feature.spectral_flatness(S=mag)[0]
    cent_frames = librosa.feature.spectral_centroid(S=mag, sr=sr)[0]
    rms_frames = librosa.feature.rms(S=mag, frame_length=2048)[0]

    segments = []
    for start in range(0, max(1, y.size - seg_len + 1), hop_len):
        length = min(seg_len, y.size - start)
        if length < seg_len // 2 and y.size > seg_len:
            continue
        f0 = start // 512
        f1 = min(onset_env.size, (start + length) // 512 + 1)
        segments.append({
            "start_sec": start / sr,
            "end_sec": (start + length) / sr,
            "features": _window_features(
                y[start:start + length], sr,
                perc_energy=perc_env[start:start + length],
                onset_env=onset_env[f0:f1],
                beats=all_beats[(all_beats >= f0) & (all_beats < f1)] - f0,
                chroma=chroma[:, f0:f1],
                flux=flux_frames[f0:f1],
                flatness=flat_frames[f0:f1],
                centroid=cent_frames[f0:f1],
                rms=rms_frames[f0:f1],
            ),
        })

    if not segments:  # 极短音频兜底：整条作为一个窗口
        segments.append({
            "start_sec": 0.0,
            "end_sec": y.size / sr,
            "features": _window_features(
                y, sr, perc_energy=perc_env, onset_env=onset_env, beats=all_beats,
                chroma=chroma, flux=flux_frames, flatness=flat_frames,
                centroid=cent_frames, rms=rms_frames,
            ),
        })

    return {"global": global_feats, "segments": segments}


def _window_features(
    y: np.ndarray,
    sr: int,
    perc_energy: np.ndarray | None = None,
    onset_env: np.ndarray | None = None,
    beats: np.ndarray | None = None,
    chroma: np.ndarray | None = None,
    flux: np.ndarray | None = None,
    flatness: np.ndarray | None = None,
    centroid: np.ndarray | None = None,
    rms: np.ndarray | None = None,
) -> Dict[str, float]:
    """单个时间窗的特征子集。

    所有重计算量（CQT/STFT/起音包络/HPSS）都由调用方整条算好后切片传入；
    只有当某个参数缺失时才在窗内自行计算——那是给单元测试和短音频兜底用的。
    """
    import librosa

    hop = 512

    if rms is None:
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    rms_db = 20.0 * np.log10(float(np.mean(rms)) + 1e-9)

    dur = max(len(y) / sr, 1e-6)

    if onset_env is None:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
        onset_times = librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=sr, hop_length=hop, backtrack=False
        )
        onset_rate = float(len(onset_times)) / dur
    else:
        # 起伏量：包络超过自适应阈值的次数，作为"起音事件数"的代理。
        # 与 onset_detect 的绝对计数不完全等价，但同为"事件密度"的单调量。
        onset_env = np.asarray(onset_env, dtype=np.float64)
        if onset_env.size > 4 and onset_env.max() > 0:
            thresh = np.mean(onset_env) + 0.5 * np.std(onset_env)
            above = onset_env > thresh
            rising = np.sum((~above[:-1]) & above[1:])
            onset_rate = float(max(rising, 1)) / dur
        else:
            onset_rate = 0.0

    if beats is None:
        tempo, beats_arr = librosa.beat.beat_track(
            onset_envelope=onset_env, sr=sr, hop_length=hop, units="frames"
        )
        tempo_bpm = float(np.atleast_1d(tempo)[0])
    else:
        beats_arr = np.asarray(beats).ravel()
        ioi = np.diff(beats_arr)
        ioi = ioi[ioi > 0]
        tempo_bpm = float(60.0 * sr / (np.mean(ioi) * hop)) if ioi.size >= 2 else 0.0

    if centroid is None:
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
    if flux is None:
        flux = _spectral_flux(librosa.stft(y, n_fft=2048, hop_length=hop))
    if chroma is None:
        # 与全局保持一致，用 CQT（chroma_stft 会把 mode_score 幅度压缩）
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    if flatness is None:
        flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop)[0]
    if chroma.shape[1] == 0:
        return _empty_window_features(y, sr, dur)

    chroma_mean = chroma.mean(axis=1)
    mode_info = _estimate_mode(chroma_mean)

    if perc_energy is None:
        perc_energy = _percussive_envelope(y)
    eh = float(np.sum(y ** 2)) + 1e-9
    ep = float(np.sum(np.asarray(perc_energy, dtype=np.float64) ** 2)) + 1e-9

    cn = chroma_mean / (chroma_mean.sum() + 1e-9)
    chroma_entropy = float(-np.sum(cn * np.log(cn + 1e-12)))

    return {
        "tempo_bpm": _safe(tempo_bpm),
        "onset_rate": _safe(onset_rate),
        "rms_db": _safe(rms_db),
        "peak": _safe(float(np.abs(y).max())),
        "duration_sec": len(y) / sr,
        "is_window": True,
        "pulse_clarity": _safe(_pulse_clarity(onset_env, beats_arr)),
        "spectral_centroid_hz": _safe(float(np.mean(centroid))),
        "spectral_flux": _safe(float(np.mean(flux))),
        "percussive_ratio": _safe(ep / (eh + ep)),
        "harmonic_complexity": _safe(
            _norm(chroma_entropy, 0.0, np.log(12.0)) * 0.7
            + _norm(float(np.mean(flatness)), 0.0, 0.5) * 0.3
        ),
        "mode": mode_info["mode"],
        "mode_score": mode_info["mode_score"],
        "key_clarity": mode_info["key_clarity"],
    }


def _empty_window_features(y: np.ndarray, sr: int, dur: float) -> Dict[str, float]:
    """窗口短到算不出色度帧时的占位，保证下游字段齐全、不抛 KeyError。"""
    return {
        "tempo_bpm": 0.0,
        "onset_rate": 0.0,
        "rms_db": _safe(20.0 * np.log10(float(np.sqrt(np.mean(y ** 2))) + 1e-9)),
        "peak": _safe(float(np.abs(y).max())),
        "duration_sec": len(y) / sr,
        "is_window": True,
        "pulse_clarity": 0.0,
        "spectral_centroid_hz": 0.0,
        "spectral_flux": 0.0,
        "percussive_ratio": 0.0,
        "harmonic_complexity": 0.5,
        "mode": "major",
        "mode_score": 0.0,
        "key_clarity": 0.0,
    }


def _percussive_envelope(y: np.ndarray) -> np.ndarray:
    """整条音频的打击分量（HPSS 只做一次）。

    性能修正记录：早期在 `_window_features` 里逐窗调用 hpss，
    127s 音频 × 41 窗口共耗 32s，占端到端 42s 的绝大部分。
    HPSS 是全局可分解的线性操作，按窗口切片与整条计算近似等价，
    因此改为整条算一次后切片复用。
    """
    import librosa

    _, y_p = librosa.effects.hpss(y)
    return np.asarray(y_p, dtype=np.float64)


def _spectral_flux(S: np.ndarray) -> np.ndarray:
    """谱通量：相邻帧幅度谱的正向变化之和，按总能量归一。

    注意：早期实现先对每帧做 L1 归一化再取差分，结果恒为 1e-3 量级、
    被当成常数。正确做法是在**未归一化**的幅度谱上取差分，
    再用总能量归一，得到与响度无关但保留"变化强度"的量。
    """
    mag = np.abs(S)
    diff = np.diff(mag, axis=1)
    flux = np.maximum(diff, 0.0).sum(axis=0)
    denom = mag.sum(axis=0)[1:] + 1e-9
    return flux / denom


def _pulse_clarity(onset_env: np.ndarray, beats: np.ndarray) -> float:
    """节拍规律性：拍点间期变异度与拍点起音强度的联合度量，映射到 [0,1]。

    修正记录：早期版本用 1-CV 且 ioi 取自 beat_track 的整数帧，
    对 117BPM 电子乐恒等于 0.96~1.00（顶格饱和，等于没有信息）。
    现在加入起音强度项，使"有拍但弱"与"有拍且强"能区分开。
    """
    beats = np.asarray(beats).ravel()
    if beats.size < 4:
        return 0.0
    ioi = np.diff(beats).astype(np.float64)
    ioi = ioi[ioi > 0]
    if ioi.size < 3:
        return 0.0

    cv = float(np.std(ioi) / (np.mean(ioi) + 1e-9))
    regularity = float(np.clip(1.0 - cv / 0.5, 0.0, 1.0))  # 按 0.5 的 CV 尺度归一

    env = np.asarray(onset_env, dtype=np.float64)
    if env.size < 4 or env.max() <= 0:
        return 0.0
    idx = beats[beats < env.size].astype(int)
    if idx.size < 3:
        return regularity * 0.5
    # 拍点处的起音强度相对于整体分布的显著性
    peaks = env[idx]
    baseline = float(np.percentile(env, 60))
    strength = float(np.clip(np.mean(peaks) / (baseline + 1e-9) - 1.0, 0.0, 1.0))

    return float(np.clip(0.6 * regularity + 0.4 * strength, 0.0, 1.0))


def _norm(value: float, lo: float, hi: float) -> float:
    """线性归一到 [0,1]。"""
    if hi <= lo:
        return 0.0
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))
