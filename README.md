# moodio

**Music emotion analysis for AI agents.** A deterministic local analyzer that turns
an audio file into an emotion you can act on — plus two tools that let a DSH agent
call it directly.

[中文说明](README.zh-CN.md) · [Algorithm notes](docs/algorithm.md) · [Changelog](CHANGELOG.md)

---

## Why another music-emotion tool

Most music emotion models collapse everything into two numbers — valence and
arousal — and then pick a word from the quadrant. That throws away the distinction
people actually care about:

| Two axes say | What it could really be |
|---|---|
| low arousal + minor key | **悲伤** (intimate sadness) *or* **悲壮** (sombre grandeur) |
| high arousal + major key | **欢快** (light, bouncy) *or* **热烈** (full-blown intensity) |

A slow piano ballad and a 34 dB-range orchestral epic can land on the same
coordinate. moodio adds two more axes so they don't:

```
valence   -1..1    emotional polarity        (mode, harmony)
arousal   -1..1    activation                (tempo, onset density)
texture    0..1    attack / aggression       (distortion, percussion, noise)
grandeur   0..1    scale / weight            (dynamic range, sustained loudness, thick texture)
```

`texture` and `grandeur` are **independent** on purpose. Orchestral power comes
from scale, not from grit: a piece can be `texture 0.10 / grandeur 0.72` — the
signature of 悲壮 — and a three-axis model that only knows "attack" will misread
it as *quiet*.

## The four axes in practice

Measured on real tracks (values are from `moodio analyze`):

| Track | valence | arousal | texture | grandeur | Verdict |
|---|---|---|---|---|---|
| 在虚无中永存 - 英雄主义 | −0.23 | −0.11 | **0.10** | **0.72** | **悲壮** |
| 邓垚 - 诀别书 | −0.16 | −0.13 | 0.15 | 0.51 | **悲伤** |
| Two Steps From Hell - Victory | +0.28 | +0.04 | 0.33 | 0.69 | **壮丽** |
| Coldplay - Yellow | +0.21 | −0.24 | 0.36 | 0.55 | **平静** |
| Bruno Mars - Runaway Baby | +0.17 | +0.47 | **0.57** | 0.48 | **热烈** |

The two negative-valence tracks differ only in grandeur, and that is exactly the
difference between an epic and an elegy.

## Install

### As a DSH plugin (gives your agent two tools)

```bash
git clone <this-repo> moodio
cd moodio
pip install -r python/requirements.txt
```

Then install it as a bundle in the DSH profile — the analyzer ships inside the
package, so no path configuration is needed:

```text
plugin_manager(action: "install_bundle", target: "<abs path to the moodio folder>")
```

This registers `analyze_music_emotion` and `batch_analyze_music`.

### As a standalone CLI

```bash
cd python
python moodio.py analyze --path "/path/to/track.flac"
```

## Usage

### One track

```bash
python moodio.py analyze --path "song.flac" --segment-sec 10 --hop-sec 10
```

```json
{
  "emotion": { "emotion": "悲壮", "valence": -0.225, "arousal": -0.114 },
  "emotion_4d": {
    "emotion": "悲壮",
    "arousal_band": "low", "texture_band": "low", "grandeur_band": "high",
    "texture": 0.1026, "grandeur": 0.722,
    "desc": "宏大而不粗糙，庄严的沉郁"
  },
  "texture_parts": { "spectral_flatness": 0.034, "percussive_ratio": 0.091 },
  "grandeur_parts": { "dynamic_range_db": 0.950, "full_loud_share": 0.519 }
}
```

Human-readable instead of JSON:

```bash
python moodio.py analyze --path "song.flac" --human
```

### A whole library

```bash
python moodio.py batch --path "~/Music" --out library.jsonl --workers 6
```

Resumable: re-running skips tracks that already succeeded and retries the
failures. Progress goes to stderr, so stdout stays parseable.

### A report

```bash
python moodio.py report --path library.jsonl --calibration fixed
```

Writes Markdown plus machine-readable JSON: distribution, quadrant shares,
confidence, arc types, and error breakdown.

### From an agent

Two tools, no shell plumbing required:

```text
analyze_music_emotion(path: "/music/track.flac")
batch_analyze_music(path: "/music", workers: 6)
```

## What it will not do

Being explicit about this is part of the tool:

- **Lyrics are not considered.** A lyrically devastating song over an upbeat
  backing reads as upbeat.
- **No human ratings are involved.** Weights come from published directions in
  music psychology, not from fitting listener scores. Direction and relative
  ranking are reliable; absolute values are not a population average. Compare
  tracks inside one run instead of quoting numbers.
- **Two calibration modes exist and must not be mixed.** `fixed` (default) makes
  zero mean "neutral evidence", so results are comparable across collections.
  `corpus-relative` re-centres on the current collection's median, which uses all
  four quadrants on a skewed library but makes values incomparable across
  libraries.
- **`confidence` measures input quality** (duration, loudness, key clarity), not
  prediction accuracy.

More detail, including the calibration failure that made 264 of 523 tracks read
as "melancholic", is in [docs/algorithm.md](docs/algorithm.md).

## Project layout

```
moodio/
├── index.js              # DSH Host plugin: registers the two tools
├── cordis.patch.yml      # bundle patch
├── locale/{en,zh}.json   # plugin display metadata
├── icon.svg
├── test/offline.test.mjs # plugin contract tests (no DSH needed)
└── python/
    ├── moodio.py         # agent-facing CLI
    ├── requirements.txt
    ├── music_emotion/    # analyzer package
    │   ├── vamodel.py         # valence/arousal from acoustic cues
    │   ├── texture.py         # texture + grandeur scores
    │   ├── four_axis.py       # four-axis -> emotion word
    │   ├── emotion_taxonomy.py# 2-axis fallback vocabulary
    │   ├── features.py        # librosa feature extraction (cached per track)
    │   ├── pipeline.py        # orchestration
    │   ├── batch.py           # resumable library runs
    │   ├── report.py          # markdown/JSON summary
    │   └── neural.py          # optional wav2small sidecar
    ├── agent/            # machine-readable usage contract (brief/rules/skills)
    └── tests/            # self-contained suites (no audio corpus required)
```

## Tests

```bash
npm test                 # plugin contract tests (25)
npm run test:python      # analyzer unit + calibration tests (34)
npm run test:all
```

The Python suites synthesise their own stimuli or use fixed feature probes, so
they run without any audio files. CI runs both halves on every push.

## Requirements

- Python ≥ 3.9 with `numpy`, `librosa`, `soundfile`
- Node ≥ 18 (plugin half only)
- Audio decoding uses libsndfile; MP3 works when libsndfile is built with mpg123
  (the common case). `m4a`/`aac` need conversion first.

## License

MIT. The optional neural cross-check is written against `audeering/wav2small`,
whose weights are CC-BY-NC-SA-4.0 and are **not** bundled here — see
[LICENSE](LICENSE) for the third-party notice.
