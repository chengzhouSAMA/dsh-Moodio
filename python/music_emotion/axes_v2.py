#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
music_axes_v2.py - score one piece of music on 12 acoustic axes in a single run.
一次运行，给一首歌的 12 个轴打分（只用 numpy + scipy，纯 CPU，不用下载模型）。

 #  Axis                         Formula (every term is first scaled to 0-1)             Basis
 1  Valence 效价                 0.60*mode + 0.25*(1-roughness) + 0.15*key_clarity       harmony / tonality
 2  Energy 能量                  0.80*loudness + 0.20*spectral_flux                      dynamic level
 3  Brightness 明亮              0.60*centroid + 0.40*high_freq_share                    spectral tilt (high side)
 4  Warmth 温润                  0.50*low_mid + 0.30*harmonicity + 0.20*(1-centroid)     spectral tilt (low side)
 5  Dissonance 刺耳              Sethares sensory roughness                              harmonic dissonance
 6  Tempo 速度                   BPM                                                     beat rate
 7  RhythmicRegularity 节奏规整  1 - CV of inter-onset intervals                         pulse stability
 8  Percussiveness 打击感        share of percussive energy (HPSS)                       sound-source component
 9  Density 音符密度             onsets per second                                       event density
10  Grandeur 规模                0.50*dynamic_range + 0.30*(1-loudness_fluct) + 0.20*bass  dynamic structure
11  BuildContrast 起伏           spread of loudness / brightness between 3-s segments    structural change
12  HarmonicComplexity 和声复杂度 entropy of the pitch-class distribution                harmonic colour

"value" is always in [0, 1].  Tempo / Density / Dissonance also give "raw" in BPM / onsets per s /
roughness.  RhythmicRegularity, Percussiveness and HarmonicComplexity are already 0-1 measures
and are NOT rescaled.  These are acoustic measurements, not human ratings.

Usage / 用法（默认整曲；--seconds N 可显式取中段 N 秒）:
  python -m music_emotion.axes_v2 song.mp3              # JSON (ASCII-safe, for a plugin/tool)
  python -m music_emotion.axes_v2 song.mp3 --seconds 60
  python -m music_emotion.axes_v2 song.mp3 --table              # readable table with Chinese names
  python -m music_emotion.axes_v2 --calibrate ./my_music --out calib.json
  python -m music_emotion.axes_v2 song.mp3 --calib calib.json   # scale relative to YOUR library
  python -m music_emotion.axes_v2 --selftest                    # synthetic sounds, sanity check

Needs numpy + scipy.  To read mp3/m4a also install ONE of: soundfile (pip install soundfile),
ffmpeg (in PATH) or librosa.  wav works with plain scipy.
"""
import argparse
import glob
import json
import math
import os
import shutil
import subprocess
import sys

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal
from scipy.ndimage import gaussian_filter1d, median_filter, uniform_filter1d

SR = 22050
N_FFT, HOP = 2048, 512            # short window: loudness, timbre, rhythm
N_LONG, HOP_LONG = 8192, SR // 2  # long window (fine pitch resolution): roughness, key; one block / 0.5 s
SEG_SECONDS = 3.0                 # segment length for BuildContrast
EPS = 1e-9

AXES = ["Valence", "Energy", "Brightness", "Warmth", "Dissonance", "Tempo",
        "RhythmicRegularity", "Percussiveness", "Density", "Grandeur", "BuildContrast",
        "HarmonicComplexity"]
ZH = {"Valence": "效价", "Energy": "能量", "Brightness": "明亮", "Warmth": "温润",
      "Dissonance": "刺耳", "Tempo": "速度", "RhythmicRegularity": "节奏规整",
      "Percussiveness": "打击感", "Density": "音符密度", "Grandeur": "规模",
      "BuildContrast": "起伏", "HarmonicComplexity": "和声复杂度"}
SHORT = {"Valence": "Val", "Energy": "Ene", "Brightness": "Bri", "Warmth": "War",
         "Dissonance": "Dis", "Tempo": "Tem", "RhythmicRegularity": "Reg",
         "Percussiveness": "Per", "Density": "Den", "Grandeur": "Gra", "BuildContrast": "Bui",
         "HarmonicComplexity": "Har"}

# (lo, hi) used to scale each raw feature to [0, 1].  Rough guesses for typical music;
# --calibrate replaces them with the 5th / 95th percentile of YOUR own library.
DEFAULT_RANGES = {
    "loudness_db": (-32.0, -8.0),       # mean frame RMS, dBFS
    "flux": (0.5, 4.0),                 # mean positive change of band energy per frame, dB
    "log_centroid": (math.log10(900.0), math.log10(4000.0)),
    "log_hf": (-2.6, -0.8),             # log10(share of energy above 3 kHz)
    "lowmid_share": (0.40, 0.95),       # energy 100-800 Hz / 80-8000 Hz
    "harmonicity": (0.30, 0.90),        # harmonic share of energy (HPSS, margin 2)
    "roughness": (0.02, 0.25),          # Sethares sensory roughness
    "mode_score": (-0.30, 0.30),        # >0 major-like, <0 minor-like
    "key_clarity": (0.40, 0.90),
    "tempo_bpm": (60.0, 180.0),
    "onset_rate": (0.5, 8.0),           # onsets per second
    "dyn_range_db": (3.0, 25.0),        # 95th - 5th percentile of 0.4-s loudness, dB
    "loud_fluct": (0.05, 1.5),          # mean |change| of loudness between frames, dB
    "bass_share": (0.10, 0.70),         # energy 40-200 Hz / 40-8000 Hz
    "seg_loud_std": (0.5, 6.0),         # std of segment loudness, dB
    "seg_bright_std": (0.01, 0.20),     # std of segment brightness (0-1 scale)
}

KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


# --------------------------------------------------------------------------- audio loading
def _resample(y, fs, sr):
    if int(fs) == sr:
        return y.astype(np.float32)
    g = math.gcd(int(fs), int(sr))
    return signal.resample_poly(y, sr // g, int(fs) // g).astype(np.float32)


def _load_ffmpeg(path, sr, seconds):
    dur = None
    if seconds > 0 and shutil.which("ffprobe"):
        try:
            out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                  "-of", "default=nw=1:nk=1", path],
                                 capture_output=True, text=True, timeout=30)
            dur = float(out.stdout.strip())
        except Exception:
            dur = None
    cmd = ["ffmpeg", "-v", "error"]
    if seconds > 0:
        ss = max(0.0, (dur - seconds) / 2.0) if dur else 0.0
        cmd.extend(["-ss", f"{ss:.2f}", "-t", f"{seconds:.2f}"])
    cmd.extend(["-i", path, "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"])
    raw = subprocess.run(cmd, capture_output=True, timeout=180).stdout
    y = np.frombuffer(raw, dtype=np.float32)
    if y.size == 0:
        raise RuntimeError("ffmpeg returned no audio")
    return y


def load_audio(path, sr=SR, seconds=0.0):
    """读取整曲；只有显式传正数 seconds 时才取中间 N 秒。"""
    errors = []
    try:  # 1) soundfile
        import soundfile as sf
        info = sf.info(path)
        n = int(seconds * info.samplerate) if seconds > 0 else -1
        start = max(0, (info.frames - n) // 2) if seconds > 0 else 0
        y, fs = sf.read(path, start=start, frames=n, dtype="float32", always_2d=True)
        return _resample(y.mean(axis=1), fs, sr)
    except Exception as e:
        errors.append(f"soundfile: {e}")
    if path.lower().endswith(".wav"):  # 2) plain scipy for wav
        try:
            from scipy.io import wavfile
            fs, y = wavfile.read(path)
            if y.dtype == np.int16:
                y = y / 32768.0
            elif y.dtype == np.int32:
                y = y / 2147483648.0
            elif y.dtype == np.uint8:
                y = (y.astype(np.float32) - 128.0) / 128.0
            y = y.astype(np.float32)
            if y.ndim > 1:
                y = y.mean(axis=1)
            n = int(seconds * fs) if seconds > 0 else len(y)
            start = max(0, (len(y) - n) // 2)
            return _resample(y[start:start + n], fs, sr)
        except Exception as e:
            errors.append(f"scipy wav: {e}")
    if shutil.which("ffmpeg"):  # 3) ffmpeg
        try:
            return _load_ffmpeg(path, sr, seconds)
        except Exception as e:
            errors.append(f"ffmpeg: {e}")
    try:  # 4) librosa
        import librosa
        if seconds > 0:
            try:
                dur = librosa.get_duration(path=path)
            except TypeError:
                dur = librosa.get_duration(filename=path)
            off = max(0.0, (dur - seconds) / 2.0)
            y, _ = librosa.load(path, sr=sr, mono=True, offset=off, duration=seconds)
        else:
            y, _ = librosa.load(path, sr=sr, mono=True)
        return y.astype(np.float32)
    except Exception as e:
        errors.append(f"librosa: {e}")
    raise RuntimeError("could not read audio (" + "; ".join(errors) +
                       "). Try: pip install soundfile  (or install ffmpeg)")


# --------------------------------------------------------------------------- low-level DSP
def stft_frames(y, n_fft, hop):
    pad = n_fft // 2
    yp = np.pad(y, (pad, pad), mode="reflect")
    return sliding_window_view(yp, n_fft)[::hop]


def sethares(f, a):
    """Sensory roughness of partials (f in Hz, a relative amplitudes), Plomp-Levelt / Sethares."""
    if len(f) < 2:
        return 0.0
    f1 = np.minimum(f[:, None], f[None, :])
    f2 = np.maximum(f[:, None], f[None, :])
    s = 0.24 / (0.021 * f1 + 19.0)
    d = np.exp(-3.5 * s * (f2 - f1)) - np.exp(-5.75 * s * (f2 - f1))
    r = 0.5 * (a[:, None] * a[None, :] * d).sum()
    return float(r / ((a ** 2).sum() + EPS))


def key_mode(chroma):
    """Krumhansl-Schmuckler: (major_corr - minor_corr, best_corr) of a 12-bin pitch-class profile."""
    c = chroma - chroma.mean()
    if c.std() < 1e-12:
        return 0.0, 0.0

    def best(profile):
        return max(np.corrcoef(c, np.roll(profile, k))[0, 1] for k in range(12))
    maj, mnr = best(KK_MAJOR), best(KK_MINOR)
    return float(maj - mnr), float(max(maj, mnr))


def norm(x, rng):
    lo, hi = rng
    return np.clip((np.asarray(x, dtype=float) - lo) / (hi - lo + EPS), 0.0, 1.0)


def _brightness(log_cent, log_hf, ranges):
    return 0.60 * norm(log_cent, ranges["log_centroid"]) + 0.40 * norm(log_hf, ranges["log_hf"])


# --------------------------------------------------------------------------- feature extraction
def extract(y):
    """Return (F, seg): raw features of the whole excerpt, and per-segment loudness / spectrum."""
    y = np.asarray(y, dtype=np.float32)
    y = y - y.mean()
    if len(y) < 3 * SR:
        raise ValueError("audio is too short (need at least 3 seconds)")
    if float(np.sqrt(np.mean(y ** 2))) < 1e-5:
        raise ValueError("audio is silent")
    dur = len(y) / SR
    fps = SR / HOP

    # ---- short window: loudness, timbre, rhythm -----------------------------------------
    fr = stft_frames(y, N_FFT, HOP)
    win = np.hanning(N_FFT).astype(np.float32)
    S = np.abs(np.fft.rfft(fr * win, axis=1)).T.astype(np.float32)            # (bins, frames)
    rms = np.sqrt(np.mean(fr.astype(np.float32) ** 2, axis=1))
    rms_db = 20 * np.log10(rms + EPS)
    active = rms_db > np.percentile(rms_db, 95) - 45
    if active.sum() < 10:
        active[:] = True
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / SR)
    P = S ** 2
    Pa = P[:, active]

    def band(lo, hi, A=Pa):
        return A[(freqs >= lo) & (freqs < hi)].sum()

    F = {}
    total = band(80, 8000) + EPS
    # loudness, dynamic range, loudness fluctuation
    F["loudness_db"] = float(rms_db[active].mean())
    st_db = 10 * np.log10(uniform_filter1d(rms ** 2, size=17, mode="nearest") + EPS)   # ~0.4 s
    F["dyn_range_db"] = float(np.percentile(st_db[active], 95) - np.percentile(st_db[active], 5))
    sm = uniform_filter1d(rms_db, size=3, mode="nearest")
    d_ok = active[1:] & active[:-1]
    F["loud_fluct"] = float(np.abs(np.diff(sm))[d_ok].mean()) if d_ok.any() else 0.0
    # spectral tilt
    cent_t = (freqs @ S) / (S.sum(axis=0) + EPS)
    F["log_centroid"] = float(np.log10(cent_t[active].mean() + 1.0))
    F["log_hf"] = float(np.log10(band(3000, 11025) / (band(80, 11025) + EPS) + 1e-4))
    F["lowmid_share"] = float(band(100, 800) / total)
    F["bass_share"] = float(band(40, 200) / (band(40, 8000) + EPS))

    # harmonic / percussive split (soft masks with margin 2 -> a residual part is left over)
    Sb = S[(freqs >= 80) & (freqs <= 8000)][:, ::2]
    H = median_filter(Sb, size=(1, 17), mode="nearest")
    Pc = median_filter(Sb, size=(17, 1), mode="nearest")
    margin = 2.0
    mh = H ** 2 / (H ** 2 + (margin * Pc) ** 2 + EPS)
    mp = Pc ** 2 / (Pc ** 2 + (margin * H) ** 2 + EPS)
    e_tot = (Sb ** 2).sum() + EPS
    F["harmonicity"] = float((mh * Sb ** 2).sum() / e_tot)
    F["percussive_share"] = float((mp * Sb ** 2).sum() / e_tot)

    # spectral flux and onsets
    edges = np.geomspace(100, 10000, 25)
    rows = [P[(freqs >= edges[k]) & (freqs < edges[k + 1])].sum(axis=0) for k in range(24)]
    B = np.stack([r for r in rows if np.any(r > 0)])
    Bdb = 10 * np.log10(B + 1e-10)
    Bdb = np.maximum(Bdb, Bdb.max() - 80)
    flux_t = np.maximum(0, np.diff(Bdb, axis=1)).mean(axis=0)
    f_ok = active[1:] & active[:-1]
    F["flux"] = float(flux_t[f_ok].mean()) if f_ok.any() else 0.0
    oenv = gaussian_filter1d(flux_t, 1.0)
    pk, _ = signal.find_peaks(oenv, prominence=0.5 * oenv.std(), distance=max(1, int(0.08 * fps)))
    F["onset_rate"] = float(len(pk) / dur)
    if len(pk) >= 4:
        ioi = np.diff(pk) / fps
        F["ioi_cv"] = float(ioi.std() / (ioi.mean() + EPS))
        F["regularity"] = float(np.clip(1.0 - F["ioi_cv"], 0.0, 1.0))
    else:                                    # too few onsets to speak of a pulse
        F["ioi_cv"], F["regularity"] = float("nan"), 0.0

    x = oenv - oenv.mean()
    ac = np.fft.irfft(np.abs(np.fft.rfft(x, 2 * len(x))) ** 2)[:len(x)]
    ac = ac / (ac[0] + EPS)
    lags = np.arange(int(fps * 60 / 200), min(int(fps * 60 / 50), len(ac) - 1) + 1)
    bpm = 60 * fps / lags
    prior = np.exp(-0.5 * np.log2(bpm / 120.0) ** 2)      # prefer 60-240 BPM around 120
    F["tempo_bpm"] = float(bpm[int(np.argmax(ac[lags] * prior))])

    # ---- long window: roughness, key / mode, pitch-class entropy -------------------------
    fr2 = stft_frames(y, N_LONG, HOP_LONG)
    S2 = np.abs(np.fft.rfft(fr2 * np.hanning(N_LONG).astype(np.float32), axis=1))
    f2 = np.fft.rfftfreq(N_LONG, 1.0 / SR)
    loud2 = 20 * np.log10(np.sqrt(np.mean(fr2.astype(np.float32) ** 2, axis=1)) + EPS)
    act2 = loud2 > loud2.max() - 45

    rsel, csel = (f2 >= 80) & (f2 <= 5000), (f2 >= 100) & (f2 <= 2000)
    rf, cf = f2[rsel], f2[csel]
    rough = np.zeros(S2.shape[0])
    chroma = np.zeros((S2.shape[0], 12))
    for i in range(S2.shape[0]):
        m = S2[i, rsel]
        if m.max() <= 0:
            continue
        p, _ = signal.find_peaks(m, height=0.03 * m.max(), distance=2)
        if len(p) >= 2:
            if len(p) > 12:
                p = p[np.argsort(m[p])[-12:]]
            rough[i] = sethares(rf[p], m[p] / m[p].max())
        mc = S2[i, csel]
        q, _ = signal.find_peaks(mc, height=0.05 * mc.max(), distance=2)
        if len(q) > 40:
            q = q[np.argsort(mc[q])[-40:]]
        pc = np.round(69 + 12 * np.log2(cf[q] / 440.0)).astype(int) % 12
        np.add.at(chroma[i], pc, np.sqrt(mc[q]))
    F["roughness"] = float(rough[act2].mean())
    pcs = chroma[act2].sum(axis=0)
    F["mode_score"], F["key_clarity"] = key_mode(pcs)
    p = pcs / (pcs.sum() + EPS)
    F["pc_entropy"] = float(-(p * np.log2(p + EPS)).sum() / math.log2(12))

    # ---- segments: loudness / brightness per 3 s ----------------------------------------
    seg_len = int(SEG_SECONDS * fps)
    if dur < 4 * SEG_SECONDS:
        seg_len = max(8, int(len(rms_db) / 4))
    seg = {"loud_db": [], "log_centroid": [], "log_hf": []}
    for a in range(0, len(rms_db) - seg_len + 1, seg_len):
        ok = active[a:a + seg_len]
        if ok.sum() < 0.5 * seg_len:
            continue
        sl = slice(a, a + seg_len)
        Ps = P[:, sl][:, ok]
        seg["loud_db"].append(float(rms_db[sl][ok].mean()))
        seg["log_centroid"].append(float(np.log10(cent_t[sl][ok].mean() + 1.0)))
        seg["log_hf"].append(float(np.log10(Ps[freqs >= 3000].sum() /
                                            (Ps[freqs >= 80].sum() + EPS) + 1e-4)))
    return F, seg


# --------------------------------------------------------------------------- scoring
def score(F, seg, ranges, with_segments=False):
    n = lambda k: float(norm(F[k], ranges[k]))
    mode_n, clarity_n, rough_n = n("mode_score"), n("key_clarity"), n("roughness")
    loud_n, flux_n, cent_n = n("loudness_db"), n("flux"), n("log_centroid")

    # BuildContrast: spread of loudness and brightness between segments
    if len(seg["loud_db"]) >= 3:
        loud_std = float(np.std(seg["loud_db"]))
        bright_seg = [float(_brightness(c, h, ranges)) for c, h in zip(seg["log_centroid"], seg["log_hf"])]
        bright_std = float(np.std(bright_seg))
    else:
        loud_std, bright_std, bright_seg = 0.0, 0.0, []
    F["seg_loud_std"], F["seg_bright_std"] = loud_std, bright_std

    values = {
        "Valence": 0.60 * mode_n + 0.25 * (1 - rough_n) + 0.15 * clarity_n,
        "Energy": 0.80 * loud_n + 0.20 * flux_n,
        "Brightness": float(_brightness(F["log_centroid"], F["log_hf"], ranges)),
        "Warmth": 0.50 * n("lowmid_share") + 0.30 * n("harmonicity") + 0.20 * (1 - cent_n),
        "Dissonance": rough_n,
        "Tempo": n("tempo_bpm"),
        "RhythmicRegularity": F["regularity"],
        "Percussiveness": F["percussive_share"],
        "Density": n("onset_rate"),
        "Grandeur": 0.50 * n("dyn_range_db") + 0.30 * (1 - n("loud_fluct")) + 0.20 * n("bass_share"),
        "BuildContrast": 0.50 * n("seg_loud_std") + 0.50 * n("seg_bright_std"),
        "HarmonicComplexity": F["pc_entropy"],
    }
    raw = {"Tempo": (F["tempo_bpm"], "BPM"), "Density": (F["onset_rate"], "onsets/s"),
           "Dissonance": (F["roughness"], "roughness")}
    axes = {}
    for a in AXES:
        axes[a] = {"zh": ZH[a], "value": round(float(np.clip(values[a], 0, 1)), 3)}
        if a in raw:
            axes[a]["raw"], axes[a]["unit"] = round(float(raw[a][0]), 3), raw[a][1]
    if with_segments:
        axes["BuildContrast"]["segments"] = {
            "loudness_db": [round(v, 1) for v in seg["loud_db"]],
            "brightness": [round(v, 3) for v in bright_seg]}
    return axes


def merged_ranges(calib):
    r = dict(DEFAULT_RANGES)
    if calib:
        r.update({k: tuple(v) for k, v in calib["ranges"].items()})
    return r


def analyze(path, seconds=0.0, calib=None, with_segments=False):
    y = load_audio(path, SR, seconds)
    F, seg = extract(y)
    axes = score(F, seg, merged_ranges(calib), with_segments)
    feats = {k: (None if (isinstance(v, float) and math.isnan(v)) else round(float(v), 4))
             for k, v in F.items()}
    return {"file": os.path.basename(path), "seconds_analyzed": round(len(y) / SR, 1),
            "calibrated": bool(calib), "axes": axes, "features": feats}


# --------------------------------------------------------------------------- calibration
AUDIO_EXT = (".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".opus")


def calibrate(folder, out, seconds, max_files):
    paths = sorted(p for p in glob.glob(os.path.join(folder, "**", "*"), recursive=True)
                   if p.lower().endswith(AUDIO_EXT))[:max_files]
    if len(paths) < 10:
        print(f"[warn] only {len(paths)} files found; percentiles and correlations will be unreliable",
              file=sys.stderr)
    data = []
    for i, p in enumerate(paths, 1):
        try:
            data.append(extract(load_audio(p, SR, seconds)))
            print(f"[{i}/{len(paths)}] {os.path.basename(p)}", file=sys.stderr)
        except Exception as e:
            print(f"[skip] {os.path.basename(p)}: {e}", file=sys.stderr)
    if len(data) < 3:
        raise RuntimeError("need at least 3 readable files to calibrate")
    ranges = dict(DEFAULT_RANGES)
    for F, seg in data:      # segment spread features need one pass with default ranges first
        score(F, seg, ranges)
    for k in DEFAULT_RANGES:
        vals = np.array([F[k] for F, _ in data if k in F and not math.isnan(F[k])])
        if len(vals) < 3:
            continue
        lo, hi = float(np.percentile(vals, 5)), float(np.percentile(vals, 95))
        if hi - lo < 1e-6:
            lo, hi = lo - 0.5 * abs(lo) - 1e-3, hi + 0.5 * abs(hi) + 1e-3
        ranges[k] = (lo, hi)
    rows = [[score(F, seg, ranges)[a]["value"] for a in AXES] for F, seg in data]
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.corrcoef(np.array(rows).T)
    flags = [(AXES[i], AXES[j], round(float(C[i, j]), 2))
             for i in range(12) for j in range(i + 1, 12) if abs(C[i, j]) > 0.7]
    calib = {"n_files": len(data), "ranges": {k: list(v) for k, v in ranges.items()},
             "corr_flags": flags}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(calib, f, indent=2)
    print(f"\nSaved {out} ({len(data)} files).  Axis pairs with |r| > 0.7 (consider merging):",
          file=sys.stderr)
    for a, b, r in flags or [("(none)", "", 0.0)]:
        print(f"  {a} ~ {b}  r = {r}", file=sys.stderr)


# --------------------------------------------------------------------------- self test
def _synth(name, dur=15.0):
    t = np.arange(int(SR * dur)) / SR
    rng = np.random.default_rng(0)

    def note(f, harm=6):
        return sum(0.6 ** (h - 1) * np.sin(2 * np.pi * f * h * t) for h in range(1, harm + 1))

    def chord(fs, vib=0.2):
        return sum(note(f) for f in fs) * (1 + vib * np.sin(2 * np.pi * 0.2 * t)) / len(fs)

    def lowpass(x, fc):
        sos = signal.butter(4, fc, btype="low", fs=SR, output="sos")
        return signal.sosfilt(sos, x)

    if name == "major pad (C E G)":
        y = 0.10 * chord([261.63, 329.63, 392.00])
    elif name == "minor pad (C Eb G)":
        y = 0.10 * chord([261.63, 311.13, 392.00])
    elif name == "cluster pad (C Db D)":
        y = 0.10 * chord([261.63, 277.18, 293.66])
    elif name == "click track 120 bpm":
        y = np.zeros_like(t)
        for k in range(int(dur * 2)):
            i = int(k * 0.5 * SR)
            y[i:i + 700] += 0.6 * rng.standard_normal(700) * np.hanning(700)
    elif name == "irregular clicks":
        y = np.zeros_like(t)
        pos = 0.3
        while pos < dur - 0.2:
            i = int(pos * SR)
            y[i:i + 700] += 0.6 * rng.standard_normal(700) * np.hanning(700)
            pos += rng.uniform(0.15, 1.1)
    elif name == "white noise (soft)":
        y = 0.05 * rng.standard_normal(len(t))
    elif name == "dark low drone":
        y = 0.25 * lowpass(chord([65.4, 98.0, 130.8]), 400)
    elif name == "bright high tones":
        y = 0.10 * chord([1046.5, 1318.5, 1568.0])
    elif name == "loud dense fast":
        y = 0.4 * chord([130.8, 196.0, 261.6, 329.6, 392.0]) * (
            0.5 + 0.5 * (np.sin(2 * np.pi * 7 * t) > 0))
        y += 0.2 * rng.standard_normal(len(t)) * (np.sin(2 * np.pi * 8 * t) > 0.6)
    elif name == "quiet->loud, dark->bright":
        ramp = np.linspace(0.02, 0.5, len(t))
        lo = lowpass(chord([130.8, 196.0, 261.6]), 600)
        hi = chord([523.3, 659.3, 784.0, 1046.5])
        y = ramp * (lo * (1 - np.linspace(0, 1, len(t))) + hi * np.linspace(0, 1, len(t)))
    elif name == "12-tone cluster":
        fs = [261.63 * 2 ** (k / 12) for k in range(12)]
        y = 0.10 * chord(fs)
    else:
        raise KeyError(name)
    return np.clip(y, -1, 1).astype(np.float32)


SELFTEST = ["major pad (C E G)", "minor pad (C Eb G)", "cluster pad (C Db D)",
            "click track 120 bpm", "irregular clicks", "white noise (soft)", "dark low drone",
            "bright high tones", "loud dense fast", "quiet->loud, dark->bright", "12-tone cluster"]


def selftest(show_raw=False):
    print(f"{'':27s}" + " ".join(f"{SHORT[a]:>5s}" for a in AXES))
    for name in SELFTEST:
        F, seg = extract(_synth(name))
        ax = score(F, seg, DEFAULT_RANGES)
        print(f"{name:27s}" + " ".join(f"{ax[a]['value']:5.2f}" for a in AXES))
        if show_raw:
            print("    " + ", ".join(f"{k}={v:.3f}" for k, v in F.items()))


# --------------------------------------------------------------------------- CLI
def print_table(res):
    print(f"{res['file']}  ({res['seconds_analyzed']} s analysed, "
          f"{'calibrated' if res['calibrated'] else 'default ranges'})")
    for i, a in enumerate(AXES, 1):
        ax = res["axes"][a]
        extra = f"   {ax['raw']} {ax['unit']}" if "raw" in ax else ""
        line = f"{i:2d}  {a:<19s} {ax['value']:5.2f}{extra}   {ax['zh']}"
        try:
            print(line)
        except UnicodeEncodeError:
            print(line.encode("ascii", "replace").decode())


def main():
    ap = argparse.ArgumentParser(description="Score music on 12 acoustic axes (CPU only).")
    ap.add_argument("audio", nargs="?", help="audio file (mp3/wav/flac/ogg/m4a)")
    ap.add_argument("--seconds", type=float, default=0.0, help="analyse the whole song by default (0); positive N means its middle N seconds")
    ap.add_argument("--calib", help="calibration json made by --calibrate")
    ap.add_argument("--calibrate", metavar="FOLDER", help="learn ranges from a folder of your own music")
    ap.add_argument("--out", default="calib.json", help="output file for --calibrate")
    ap.add_argument("--max-files", type=int, default=200)
    ap.add_argument("--table", action="store_true", help="human-readable table instead of JSON")
    ap.add_argument("--compact", action="store_true", help="one-line JSON")
    ap.add_argument("--segments", action="store_true", help="include per-segment loudness/brightness")
    ap.add_argument("--selftest", action="store_true", help="run on synthetic sounds")
    ap.add_argument("--raw", action="store_true", help="with --selftest: also print raw features")
    a = ap.parse_args()
    try:
        if a.selftest:
            selftest(a.raw)
        elif a.calibrate:
            calibrate(a.calibrate, a.out, a.seconds, a.max_files)
        elif a.audio:
            calib = None
            if a.calib:
                with open(a.calib, encoding="utf-8") as f:
                    calib = json.load(f)
            res = analyze(a.audio, a.seconds, calib, with_segments=a.segments)
            if a.table:
                print_table(res)
            else:   # ASCII-safe JSON: identical when parsed, no console-encoding surprises
                print(json.dumps(res, ensure_ascii=True, indent=None if a.compact else 2))
        else:
            ap.print_help()
    except Exception as e:  # keep the output machine-readable for a plugin/tool wrapper
        print(json.dumps({"error": str(e)}, ensure_ascii=True))
        sys.exit(1)


if __name__ == "__main__":
    main()
