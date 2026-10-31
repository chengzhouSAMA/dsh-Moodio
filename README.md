# dsh-Moodio

[![CI](https://github.com/chengzhouSAMA/dsh-Moodio/actions/workflows/ci.yml/badge.svg)](https://github.com/chengzhouSAMA/dsh-Moodio/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-4d6bfe.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-4d6bfe.svg)](python/pyproject.toml)

Music emotion analysis for agents, measured instead of guessed.

Four axes — valence, arousal, texture, grandeur — and optionally twelve acoustic
diagnostics, all computed locally from the waveform. No training, no API keys, no
downloaded weights. It comes with a DSH host plugin so an agent can call it as a
tool.

[中文说明](README.zh-CN.md) · [Algorithm notes](docs/algorithm.md) · [Changelog](CHANGELOG.md) · [Releases](https://github.com/chengzhouSAMA/dsh-Moodio/releases)

## What it produces

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

## Quick start

```bash
git clone https://github.com/chengzhouSAMA/dsh-Moodio.git
cd dsh-Moodio
pip install -r python/requirements.txt

python python/moodio.py analyze --path song.mp3 --human
python python/moodio.py analyze --path song.mp3 --axes-v2 --fields emotion,emotion_4d,axes_v2
python python/moodio.py batch  --path ./music --axes-v2 --out mood.jsonl --workers 6
python python/moodio.py report --path mood.jsonl
```

Python 3.9+. `batch` writes one JSON object per track and appends, so re-running
resumes: finished tracks are skipped, failed ones are retried.

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

Each Host tool call searches for creation background and lyrics before analyzing,
returns source links, and excludes reviews and comments. Only the cleaned filename
is sent, never local directory paths. **If the search provider fails, the plugin
returns `WEB_RESEARCH_FAILED` instead of guessing** — the standalone Python CLI
stays fully offline.

```
analyze_music_emotion(path: "/music/track.flac", include_axes_v2: true)
batch_analyze_music(path: "/music", workers: 6, limit: 50)
transcribe_music_lyrics(path: "/music/track.flac", model_dir: "/models/faster-whisper")
```

Batch returns `research_by_track` for each pending song, capped at 50 tracks per
call; pass `limit: 50` and resume for larger sets.

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
95 BPM reading may really be 184. Tempo carries 0.30 of arousal.

Clipped masters lose their dynamics before analysis. A peak of `>= 0.995` sets
`clipping_suspected`, and dynamic features such as grandeur inherit the damage.

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
