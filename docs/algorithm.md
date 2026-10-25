# Algorithm notes

This document records *why* the model is shaped the way it is, including the
mistakes that were found by measuring real music. Every threshold below was set
from measured distributions, not guessed.

## 1. Pipeline

```
audio ──► decode (soundfile) ──► resample 22.05 kHz mono
      ──► features (librosa, computed once per track)
      ──► valence / arousal  (vamodel)
      ──► texture / grandeur (texture)
      ──► four-axis word     (four_axis)
```

Features are computed **once** for the whole track and then sliced per window.
An earlier version recomputed STFT/HPSS per timeline window, which cost ~379 ms
per 5 s window and made a 4-minute track take ~49 s. Sharing the analysis
reduced that without changing any value.

## 2. Valence and arousal

Both are weighted sums over normalised acoustic cues, then re-centred and
squashed to `[-1, 1]`.

**Arousal** (sum of weights = 1.0)

| cue | weight | rationale |
|---|---|---|
| `tempo_bpm` | 0.30 | faster → higher activation |
| `onset_rate` | 0.20 | more note events → more active |
| `rms_db` | 0.18 | louder → higher activation |
| `spectral_flux` | 0.16 | faster spectral change → higher activation |
| `pulse_clarity` | 0.10 | steadier pulse → more drive |
| `percussive_ratio` | 0.06 | more percussive energy |

**Valence** (sum of weights = 0.50 — this matters, see §3)

| cue | weight | rationale |
|---|---|---|
| `mode_score` | +0.42 | major → positive (largest effect in the literature) |
| `harmonic_complexity` | −0.25 | more complex harmony → less positive |
| `spectral_centroid_hz` | +0.15 | brighter → mildly more positive |
| `key_clarity` | +0.10 | clearer tonality |
| `pulse_clarity` | +0.08 | — |

`mode_score` comes from correlating the mean chroma vector against rotated
Krumhansl–Kessler major/minor profiles (24 rotations); the sign is
`best_major − best_minor`, so positive means major-leaning.

## 3. Calibration: neutral points must be fixed points

The neutral point is where the mapping outputs exactly 0. If it is wrong, every
track inherits a constant bias.

There is an arithmetic invariant: if every normalised cue equals 0.5, then
`raw = 0.5 × Σweights`. So the only defensible constants are

```
VALENCE_NEUTRAL = 0.50 × 0.5 = 0.2500
AROUSAL_NEUTRAL = 1.00 × 0.5 = 0.5000
```

### The failure this caught

The first version set these to `(0.47, 0.46)` from an intuition about
"mid-tempo, mid-brightness". But after `FEATURE_RANGES` mapping, the real cue
means are not 0.5 at all — `harmonic_complexity ≈ 0.17`, `pulse_clarity ≈ 0.22`,
`rms_db ≈ 0.89`. With the guessed constants:

- corpus valence mean was **−0.222**
- **264 of 523** tracks were labelled "melancholic"
- only **2 of 4** quadrants were ever reachable

A two-dimensional emotion model had degenerated into a line. After using the
fixed-point constants, all four quadrants appear and the synthetic-stimulus
directionality suite improved too (tempo→arousal delta went from ≈0 to +0.544).

The invariant is enforced by `tests/validate_calibration.py`, which asserts that
a probe with every cue at 0.5 yields `valence == 0`, `arousal == 0`,
`label == neutral`, and that the constants equal `Σweights × 0.5`.

### Why there are two modes

`fixed` keeps zero at "neutral evidence", so values are comparable across
collections. `corpus-relative` re-centres on the current collection's median
(`vamodel.calibrate`). A collection of bright major-key pop is genuinely
positive-leaning; whether that should read as "everything is peaceful" or be
re-centred is a question about the intended comparison, so both are exposed.
**Never mix them in one comparison.**

## 4. Texture (attack)

```
texture = 0.30·flatness + 0.26·percussive + 0.20·flux + 0.12·rms + 0.12·centroid   (each normalised)
```

`spectral_flatness` dominates because distortion and noise floor are the clearest
signal of grit. Brightness carries only 0.12: a bright harp is still gentle, so
brightness cannot decide attack on its own.

## 5. Grandeur (scale)

```
grandeur = 0.45·full_loud_share + 0.30·dynamic_range + 0.25·spectral_contrast
```

- `full_loud_share` — percentage of frames within 3 dB of the track's own 95th
  percentile loudness. This is the most direct evidence of "sustained effort",
  and it is weighted highest.
- `dynamic_range` — p95 − p5 of the frame loudness, in dB. Quantiles rather than
  max−min: one clipping spike should not define "dynamic".
- `spectral_contrast` — peak-to-valley structure, i.e. a thick harmonic body.

`active_pitches` is computed but weighted 0.00: across real tracks it clusters at
11–12 and does not discriminate.

### The failure this caught

With only `texture`, 《在虚无中永存 - 英雄主义》 was labelled **悲伤**. Its
measurements: dynamic range **34.6 dB**, **40.9 %** of the time at full loudness,
spectral contrast **27.8** — the most "grand" of everything measured. But it has
**no distortion and no high-frequency noise**, so `texture` was only 0.10 and the
model read that as *quiet*.

Adding `grandeur` gives it 0.72 (high) and the verdict becomes **悲壮**. The
comparison that separates it from 诀别书 is `full_loud_share`: 40.9 % vs 25.7 %.

### Thresholds

| axis | low | mid | high |
|---|---|---|---|
| texture | `< 0.40` | `0.40 – 0.45` | `≥ 0.45` |
| grandeur | `< 0.45` | `0.45 – 0.65` | `≥ 0.65` |

Measured texture values were `0.102 / 0.150 / 0.267 | 0.532 / 0.554` — a wide
gap between 0.27 and 0.53, so the boundary sits in the middle of the gap rather
than hugging a sample. An earlier `≥ 0.55` boundary fell exactly on
`Runaway Baby` (0.554) and pushed `Die For You` (0.532) into "mid", mislabelling
a hard-hitting track as 欢快.

## 6. Decision table

Order matters: **grandeur first, then texture.** Grandeur changes the word more
fundamentally (epic vs private); texture only decides whether grit is involved.

| valence | grandeur | texture | word |
|---|---|---|---|
| negative | high | any | 悲壮 |
| negative | mid | high | 愤懑 |
| negative | mid | mid/low | 悲伤 |
| negative | low | any | 悲伤 / 沉重 |
| positive | high | low | 壮丽 |
| positive | high | mid/high | 昂扬 |
| positive | mid | high | 热烈 |
| positive | mid | low | 平静 |
| positive | low | mid/high | 欢快 / 明快 |
| positive | low | low | 平静 |

When `|valence| ≤ 0.10` the model **refuses to state a direction** and reports
only the character of the sound (平缓 / 平实 / 强烈 / 宏伟 / 辽阔 / 尖利 / 素淡).
That is deliberate: claiming "this song is cheerful" from a valence of 0.03 would
be over-reading.

## 7. Known limitations

1. **Tempo ambiguity.** Autocorrelation can peak at both `t` and `2t`; a track
   measured at 95.7 BPM may really be 184.6. Since tempo carries 0.30 of arousal,
   a wrong octave moves the result. `_tempo_fn` uses the median over the onset
   envelope; when the top two autocorrelation peaks are close, treat tempo as
   uncertain.
2. **Clipped masters under-report arousal.** A track whose samples exceed 0 dBFS
   (`peak > 1.0`) has already lost dynamic information. The CLI sets
   `clipping_suspected` and the response carries a warning; arousal is likely
   under-estimated.
3. **Truncating a track can delete its character.** With `--max-seconds 30` the
   heroism track scored `grandeur 0.48` (mid → 平缓) because the loud section had
   not started. For epic or orchestral material, analyse the whole file.
4. **No lyrics, no culture, no listener.** This measures acoustic cues, not how a
   person feels. A requiem with a major-key arrangement reads as positive.
5. **`confidence` is input quality, not accuracy.** It drops for short clips,
   quiet audio, unclear key, or missing pulse.
6. **Neural sidecar is a speech model.** `--neural` runs `audeering/wav2small`,
   trained on speech emotion. On music its absolute values compress heavily; only
   the relative dynamics are usable, and `domain_mismatch` is always true.

## 8. Regression suites

| file | guards |
|---|---|
| `validate_calibration.py` | neutral invariant, fixed-point constants, quadrant reachability, determinism, label deadzone uses both axes |
| `test_emotion_taxonomy.py` | weak-band never over-claims a direction; flat only near origin |
| `test_texture.py` | texture separates sombre from driving; monotonic in each cue; thresholds consistent |
| `test_four_axis.py` | 悲壮 ≠ 悲伤; 热烈 ≠ 欢快; slow major stays 平静 |
| `test_batch.py` | resume ignores failures, scan is ordered, single failure is captured not raised |
| `validate_directionality.py` | synthesised stimuli move the axes in the literature's directions |

All of these run without an audio corpus, which is why CI can execute them.
