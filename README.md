# dsh-Moodio

[![CI](https://github.com/chengzhouSAMA/dsh-Moodio/actions/workflows/ci.yml/badge.svg)](https://github.com/chengzhouSAMA/dsh-Moodio/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-4d6bfe.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-4d6bfe.svg)](python/pyproject.toml)

Music emotion analysis for agents, measured instead of guessed.

Whole-song instrument candidates, melody and eighteen acoustic descriptors are
enabled by default. The Host then searches background and lyric sources and
provides `music_analysis` for an agent's combined emotion interpretation.
Acoustic features need no weights; instrument recognition uses local CLAP and
AST models installed once, with no audio upload or model API key.

[中文说明](README.zh-CN.md) · [Algorithm notes](docs/algorithm.md) · [Changelog](CHANGELOG.md) · [Releases](https://github.com/chengzhouSAMA/dsh-Moodio/releases)

## What it produces

The instrument catalog covers 21 requested targets: guzheng, erhu, pipa, dizi,
hulusi, suona, yangqin, sheng, ocarina, guqin, piano, guitar, violin, saxophone,
drum kit, cello, flute, bass guitar, ukulele, harmonica and accordion.
CLAP proposes candidates; AST corroborates supported labels. Additional
instruments require direct classifier support rather than merely a high text
similarity. Many Chinese targets have family-only or zero-shot support,
explicitly reported in `verification_scope`. Catalog coverage is not an accuracy
guarantee or a promise to recover every instrument in a dense ensemble.

```bash
python -m pip install -e "./python[instruments]"
python python/moodio.py instruments setup
python python/moodio.py instruments status
```

Setup downloads approximately 1 GB of pinned, SHA-256-verified weights (CLAP:
Apache-2.0; AST: BSD-3-Clause). An explicit `--endpoint` may select a mirror.
Analysis itself is local-only. Override the cache directory with
`MOODIO_INSTRUMENT_MODEL_DIR` / `--instrument-model-dir`; missing models produce
an explicit partial result. Use `--no-instruments` to skip recognition.

| axis | range | driven by |
|---|---|---|
| `valence` | -1..1 | mode score, harmonic complexity, brightness |
| `arousal` | -1..1 | tempo, onset rate, loudness, spectral flux |
| `texture` | 0..1 | distortion, percussion, noise floor, brightness |
| `grandeur` | 0..1 | dynamic range, sustained loudness, spectral contrast |

`texture` is attack. `grandeur` is scale. They are separate on purpose: an
orchestral piece has large grandeur and almost no texture, and if you only measure
attack you will read it as quiet.

The four axes resolve to one word. These two sit at nearly the same valence and
arousal — grandeur is the only thing that separates them, which is the whole
reason the axis exists:

| track | val | aro | tex | gran | word |
|---|---|---|---|---|---|
| 在虚无中永存 - 英雄主义 | -0.23 | -0.11 | 0.10 | 0.72 | 悲壮 |
| 邓垚 - 诀别书 | -0.16 | -0.13 | 0.15 | 0.51 | 悲伤 |
| Two Steps From Hell - Victory | +0.28 | +0.04 | 0.33 | 0.69 | 壮丽 |
| Coldplay - Yellow | +0.21 | -0.24 | 0.36 | 0.55 | 平静 |

**Twelve-axis diagnostics** (opt-in, `--axes-v2`) add Valence, Energy, Brightness,
Warmth, Dissonance, Tempo, RhythmicRegularity, Percussiveness, Density, Grandeur,
BuildContrast and HarmonicComplexity over the **whole analyzed track**. They are
0..1 measurements on a different normalization from the four axes and must not be
treated as interchangeable.

**Eighteen-axis diagnostics** (on by default; `--no-axes-v3` skips them) preserve those twelve and add
BassPresence, Noisiness, TransientSharpness, TempoStability, TonalStability and
BuildDirection. Each axis has a `status`; `value: null` means insufficient evidence,
not low intensity. BuildDirection ranges from fading (0) through flat (0.5) to
building (1). The legacy `HarmonicComplexity` measures pitch-class entropy, not
chord complexity; the extended result identifies this proxy in `measure`. These
heuristics are not validated against human ratings and never change the four-axis
emotion verdict.

Brightness now uses continuous logistic reference anchors instead of clipping
both components to zero below the reference ranges. `Brightness.evidence`
exposes the spectral centroid, energy share above 3 kHz, component scores and
weights; `normalization` identifies the mapping. Whole-track and segment scores
use the same mapping. Recompute historical brightness and dependent contrast
scores before comparing; spectral brightness is not emotional brightness.

**Candidate melody** (enabled by default in the CLI and Host, for the whole song) uses monophonic `pYIN` to produce
timestamped MIDI notes and voicing probability (not pitch accuracy) in `melody.notes`, phrase contours in
`melody.phrases`, transposition-invariant interval repetitions in `melody.motifs`,
and a grounded `melody.agent_summary`. When evidence is insufficient it abstains.
Boundary pitches are verified with wider search ranges and inconsistent readings
are removed with before/after counts in `quality.boundary_validation`. Summaries
report whole-track note coverage and chronological evidence. Motifs use sliding
interval/rhythm windows across phrase boundaries; empty results include search diagnostics.
No source separation is performed: a dense mix can cause the tracker to follow
accompaniment instead of the lead. A repeated motif is not proof of a chorus.
Use `--no-melody` / `include_melody: false` to skip it. Only set `max_seconds`
when an explicit preview is wanted; otherwise the whole song is analyzed.

## Quick start

```bash
git clone https://github.com/chengzhouSAMA/dsh-Moodio.git
cd dsh-Moodio
pip install -r python/requirements.txt

python python/moodio.py analyze --path song.mp3 --human
python python/moodio.py analyze --path song.mp3 --axes-v2 --fields emotion,emotion_4d,axes_v2
python python/moodio.py analyze --path song.mp3 --axes-v3 --fields emotion,emotion_4d,axes_v3
python python/moodio.py analyze --path song.mp3 --melody --fields clip,melody,warnings
python python/moodio.py batch  --path ./music --out mood.jsonl --workers 1
python python/moodio.py report --path mood.jsonl
```

Python 3.9+. `batch` writes one JSON object per track and appends, so re-running
resumes: finished tracks are skipped, failed ones are retried.
Use a new `--out` path when adding instruments, `--axes-v3` or melody to an existing batch:
resume does not rewrite completed records.

## The 100-track study

100 tracks sampled evenly from a 794-track library, full-song analysis, local and
deterministic. Three things came out of it:

- **62 / 100** sit inside ±0.15 arousal — two moods, one speed.
- **0.95 ± 0.02** harmonic complexity across all 100 — an axis that never moves.
- **−0.39** correlation between energy and grandeur — louder tracks measure flatter.

[![Emotion map](https://github.com/chengzhouSAMA/dsh-Moodio/releases/download/v2.2.1/01-emotion-map.png)](https://github.com/chengzhouSAMA/dsh-Moodio/releases/tag/v2.2.1)

The rest of the charts — the twelve-axis table, the fingerprint grid, the reading —
are attached to [v2.2.1](https://github.com/chengzhouSAMA/dsh-Moodio/releases/tag/v2.2.1).

## As a DSH plugin

```
plugin_manager(action: "install_bundle", target: "<abs path to dsh-Moodio>")
```

Three tools: `analyze_music_emotion`, `batch_analyze_music`,
`transcribe_music_lyrics`. The analyzer lives at `python/` inside the package,
resolved relative to `index.js`, so there is no path to configure.

Each Host tool call finishes local acoustic analysis (or lyric transcription)
and then searches online for creation background and lyrics. Source links exclude
reviews and comments; only the cleaned filename is sent, never local directory
paths.

**The lookup is mandatory, not an option.** There is no config switch and no
offline fallback: if no `web.search` provider is available, or the query throws,
the call fails with `WEB_RESEARCH_FAILED` instead of returning an analysis without
sources. The standalone Python CLI stays fully offline.

Before searching, the filename is reduced to a main title — `【Animenz】残酷天使的
行动纲领 – 新世纪福音战士 OP1 钢琴版` becomes `残酷天使的行动纲领` — and sources are
matched by title coverage rather than exact containment, so a page named after the
song is not rejected for lacking the arrangement subtitle. Every response records
the queries it sent plus `returned` / `rejected` per channel, which separates
"nothing matched the title" from "the provider had nothing".

```
analyze_music_emotion(path: "/music/track.flac", include_axes_v2: true)
analyze_music_emotion(path: "/music/track.flac", include_melody: true)
batch_analyze_music(path: "/music", workers: 1, limit: 50)
transcribe_music_lyrics(path: "/music/track.flac", model_dir: "/models/faster-whisper")
```

After analysis, batch returns `research_by_track` for each pending song. Calls are
capped at 50 tracks; pass `limit: 50` and resume for larger sets.
If any track cannot be researched, the whole call fails with `WEB_RESEARCH_FAILED`
and the completed acoustic results are returned under `value.partial`.
If some audio files fail analysis, the batch is still marked as an error but
retains its summary.

## Lyrics (opt-in)

Bring a **local** CTranslate2 Whisper directory containing `model.bin`; nothing is
downloaded automatically.

```
python -m pip install faster-whisper
python python/moodio.py lyrics --path song.mp3 --model-dir /models/faster-whisper --language en
```

Or set `MOODIO_LYRICS_MODEL_DIR`. Transcription is a guess on the mixed recording,
not verified lyrics: vocals buried in arrangement may be missed or invented.
It never changes the four-axis score.

## Caveats

This is a rule-based prior, not a trained model. The weights come from published
directional results in music psychology, not from fitting listener ratings. Use it
to rank tracks inside one run; do not quote the numbers as population averages.

Lyrics are not fed into the emotion model. A miserable lyric over bright backing
can still score as bright.

The two calibration modes are not interchangeable. `fixed` (default) puts zero at
"neutral evidence", so different collections can be compared. `corpus-relative`
re-centres on the collection median, which keeps all four quadrants reachable on a
skewed library but makes the numbers incomparable across libraries.

`confidence` is input quality, not accuracy. Short, quiet, atonal or aperiodic
audio scores low.

Tempo has an octave problem: autocorrelation often peaks at both `t` and `2t`, so a
95 BPM reading may really be 184. Tempo carries 0.30 of arousal. A `TempoStability`
near zero is the tell: the estimator is flipping between two octaves, so the tempo
value itself should not be trusted.

Clipped masters lose their dynamics before analysis. A peak of `>= 0.995` sets
`clipping_suspected`, and dynamic features such as grandeur inherit the damage.

Video rips and separated stems are often band-limited. When almost no energy sits
above 3 kHz, `Brightness` collapses toward 0 and `TransientSharpness` drops with it
— a piano piece can read darker than an orchestra. Check
`axes_v3.axes.Brightness.evidence.high_frequency_energy_share`: a real acoustic
instrument usually carries 1–5% up there, while a low-passed file carries around
0.2%. Compare a lossless source before blaming the arrangement.

Truncating with `--max-seconds` can delete a track's character: a quiet intro hides
grandeur. Analyze the whole file for orchestral material.

The calibration bug that made 264 of 523 tracks read as "melancholic" is written up
in [docs/algorithm.md](docs/algorithm.md), along with the other ones I hit.

## Layout

```
index.js              DSH host plugin
cordis.patch.yml      bundle patch
locale/{en,zh}.json   plugin metadata
python/
  moodio.py           CLI
  music_emotion/      analyzer package
    vamodel.py        valence/arousal from acoustic cues
    texture.py        texture and grandeur
    four_axis.py      four axes to an emotion word
    axes_v2.py        optional 12-axis diagnostics
    features.py       librosa extraction, shared once per track
    pipeline.py       orchestration
    batch.py          resumable directory runs
    report.py         markdown and JSON summary
    lyrics.py         optional local singing transcription
    neural.py         optional wav2small sidecar
  agent/              usage contract for agents
  tests/              self-contained suites
test/offline.test.mjs plugin contract tests, no DSH runtime needed
```

## Tests

```bash
npm run test:fast      # plugin contract + fast Python suites
npm run test:python    # all Python suites and directionality checks
npm run test:all       # everything
```

The Python suites synthesize their own stimuli or use fixed feature probes, so none
of them need an audio file. CI runs both halves on push: Python 3.9 on
`ubuntu-22.04` (no 3.9 build exists for 26.04 yet) and 3.12 on `ubuntu-latest`,
plus the Node contract tests on Node 24.

## License

MIT. The optional neural check is written against `audeering/wav2small`, whose
weights are CC-BY-NC-SA-4.0 and are not included here. See [LICENSE](LICENSE).
