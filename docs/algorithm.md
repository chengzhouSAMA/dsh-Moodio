# Algorithm notes

This document records *why* the model is shaped the way it is, including the
mistakes that were found by measuring real music. The original four-axis
classification thresholds below were set from measured distributions; the
new `axes_v3` reliability cutoffs are provisional heuristics.

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

### Instrument candidates and combined interpretation

The default CLI/Host workflow analyzes all ten-second windows, including the
tail, with local CLAP audio/text embeddings over the 21 requested targets.
Similarity is cosine similarity, not calibrated confidence. An AST AudioSet
classifier corroborates available supervised labels. Secondary instruments
require direct classifier support as well as text/audio similarity; a high
secondary CLAP similarity alone is not treated as co-occurrence. Unknown audio,
near-tied timbres and classifier disagreement remain explicit.

Western target names have direct AST labels. Several Chinese targets only have
family-level support (for example, Zither cannot distinguish guzheng from guqin),
and yangqin has no corresponding supervised label in this verifier. The catalog
reports these distinctions. The fixed thresholds are provisional, and no
21-class accuracy benchmark is claimed. Window support is not exact audible
presence duration. Dense ensembles and synthesized instruments remain difficult.

Model setup pins revisions and verifies SHA-256 for CLAP (Apache-2.0) and AST
(BSD-3-Clause); analysis never downloads weights or sends audio to a service.
`music_analysis` combines the acoustic prior, instrument timing/verification,
melody coverage and eighteen-axis signals. The Host attaches sourced background
and lyric links after local analysis, with explicit instructions for the agent to
form a combined emotion explanation, cite evidence and discuss conflicts.
External snippets are untrusted data; lyric links do not prove vocals were
detected or that the lyric text has been read. There is no invented fused
probability, and the original acoustic measurements remain auditable.

The optional `axes_v3` diagnostics add six independent acoustic descriptors
(bass energy share, spectral flatness, onset rise, local tempo stability, local
chroma stability, and early-to-late loudness/brightness direction). All 18 are
descriptors, not trained predictions of human emotion. In `axes_v3`, unreliable
pulse/tonality evidence or too few local windows yields a per-axis `null` value
and reason. Both versions use the same twelve shared descriptors; brightness
normalization is updated in both as described below.
The legacy `HarmonicComplexity` is pitch-class entropy: a wide pitch-class
distribution is not necessarily complex chord harmony. That proxy is explicitly
identified in the `axes_v3` response. The new axes use fixed heuristic ranges
until an annotated music corpus can validate calibration and distinctiveness.

### Brightness normalization

Brightness combines log spectral centroid (weight 0.6) with log energy share
above 3 kHz (weight 0.4). The old clipped normalization turned both components
into zero below their reference lower bounds, so a measured spectrum could be
reported as zero brightness. Only these brightness components now use a smooth
logistic mapping: the reference lower/upper anchors map to 0.05/0.95, with 0.5
at their midpoint. The existing default anchors remain heuristic, and a custom
calibration still supplies its own ranges. These scores are not probabilities.

Global and segment brightness use the same mapping; consequently BuildContrast
and the brightness part of BuildDirection can also change. Other axis mappings
and the four-axis emotion calculation are unaffected. The Brightness result
includes the normalization method, raw spectral centroid/high-frequency share,
component values, weights, and unrounded score. Recompute historical results
before comparing scores across this normalization change; spectral brightness
does not by itself measure a melody's perceived emotional brightness.

### Candidate melody (enabled by default in CLI and Host)

The analyzed mono waveform is resampled to 11,025 Hz, then `librosa.pyin`
estimates F0 and voicing probability per frame in C3–C7. Consecutive stable
frames form timestamped MIDI notes (`voicing_probability` does **not** measure
pitch accuracy). Frames within half a semitone of either search boundary undergo
a second estimate with that boundary expanded by an octave. Only consistent,
voiced estimates away from the expanded boundaries survive. Verification runs in
ten-second blocks with context. `quality.boundary_validation` reports raw and
retained lower-bound note counts/fractions, confirmed/rejected frames, and ranges.
This guards boundary clipping without blindly deleting genuine C3 notes.

A gap of at least 0.35 s, eight notes or eight seconds splits display phrases.
Repetition detection independently indexes sliding windows of 4–12 notes, even
across display-phrase boundaries. It compares intervals and normalized onset
spacing, accepts transposition and uniform tempo changes, requires nonoverlapping
occurrences, and permits only one 1-semitone interval difference for 5+ notes.
Long rests stop windows; redundant subwindows are suppressed. An empty result
has `motif_search` diagnostics and does not imply no repetitions in the original.

`agent_summary.overview` separates full scan duration from detected-note coverage;
up to six chronological `sections` select representative evidence across the
whole analyzed duration and expose unsupported regions. Every contour/repetition
claim links to its own timestamps. Too few notes return `insufficient_evidence`.

This is a **monophonic candidate** tracker, not multi-instrument transcription:
it does not separate vocals or instruments, distinguish bass from lead, identify
the chorus, or certify pitch in a dense mix. CLI and Host calls analyze the whole
song including melody by default; `--no-melody` / `include_melody: false` skips
this stage. The low-level Python API retains its optional `with_melody` parameter.
No weight download or online service is needed.
Real mixed recordings with note/phrase annotations are still needed to quantify
note F1, onset timing, and motif precision before promoting this to a reliable
lead-melody detector. This output never feeds the emotion axes.

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
| `test_axes_v3.py` | 18-axis schema, no-pulse/weak-key nulls, acoustic directionality, legacy invariants |
| `test_melody.py` | pitch/time from synthesized notes, transposed motifs, false matches, abstention, CLI compatibility |
| `test_instruments.py` | target coverage, score alignment, unknown/ambiguity handling, whole-track windows, classifier crosschecks, fusion limits |

All of these run without an audio corpus, which is why CI can execute them.
