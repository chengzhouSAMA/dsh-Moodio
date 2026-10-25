# dsh-Moodio

Music emotion analysis, local only. Audio in, four numbers out, plus a discrete
word derived from them.

It comes in two pieces: a Python analyzer (`python/moodio.py`) and a DSH host
plugin that exposes the analyzer as two tools. No network calls, no API keys, no
model weights to fetch for the default path.

[中文说明](README.zh-CN.md) | [Algorithm notes](docs/algorithm.md) | [Changelog](CHANGELOG.md)

## Output

Four axes, all computed from librosa features:

| axis | range | driven by |
|---|---|---|
| `valence` | -1..1 | mode score, harmonic complexity, brightness |
| `arousal` | -1..1 | tempo, onset rate, loudness, spectral flux |
| `texture` | 0..1 | distortion, percussion, noise floor, brightness |
| `grandeur` | 0..1 | dynamic range, sustained loudness, spectral contrast |

`texture` is attack. `grandeur` is scale. They are separate on purpose: an
orchestral piece has large grandeur and almost no texture, and if you only
measure attack you will read it as quiet.

Spot checks from the test corpus:

| track | val | aro | tex | gran | word |
|---|---|---|---|---|---|
| 在虚无中永存 - 英雄主义 | -0.23 | -0.11 | 0.10 | 0.72 | 悲壮 |
| 邓垚 - 诀别书 | -0.16 | -0.13 | 0.15 | 0.51 | 悲伤 |
| Two Steps From Hell - Victory | +0.28 | +0.04 | 0.33 | 0.69 | 壮丽 |
| Coldplay - Yellow | +0.21 | -0.24 | 0.36 | 0.55 | 平静 |
| Bruno Mars - Runaway Baby | +0.17 | +0.47 | 0.57 | 0.48 | 热烈 |

The first two sit at nearly the same valence and arousal. Grandeur is the only
thing that separates them, and that is the whole reason the axis exists.

## Install

```
git clone https://github.com/chengzhouSAMA/dsh-Moodio.git
cd dsh-Moodio
pip install -r python/requirements.txt
```

Python 3.9 or newer. The plugin half also needs Node 18+.

As a DSH bundle (registers `analyze_music_emotion` and `batch_analyze_music`):

```
plugin_manager(action: "install_bundle", target: "<abs path to dsh-Moodio>")
```

The analyzer is inside the package at `python/`, resolved relative to `index.js`,
so there is no path to configure. Pass `projectDir` in the plugin config only if
you want it to run against a different checkout.

## Usage

Single file:

```
cd python
python moodio.py analyze --path "song.flac" --segment-sec 10 --hop-sec 10
```

```
{
  "emotion": { "emotion": "悲壮", "valence": -0.225, "arousal": -0.114 },
  "emotion_4d": {
    "emotion": "悲壮",
    "arousal_band": "low", "texture_band": "low", "grandeur_band": "high",
    "texture": 0.1026, "grandeur": 0.722
  },
  "grandeur_parts": { "dynamic_range_db": 0.950, "full_loud_share": 0.519 }
}
```

Add `--human` for a readable summary, `--fields` to trim the payload,
`--no-segments` to skip the timeline.

Whole directory:

```
python moodio.py batch --path "~/Music" --out library.jsonl --workers 6
```

Writes one JSON object per track and appends, so re-running resumes: finished
tracks are skipped, failed ones are retried. Progress goes to stderr and stdout
stays parseable.

Report from a finished run:

```
python moodio.py report --path library.jsonl --calibration fixed
```

From an agent:

```
analyze_music_emotion(path: "/music/track.flac")
batch_analyze_music(path: "/music", workers: 6)
```

## Caveats

This is a rule-based prior, not a trained model. The weights come from published
directional results in music psychology, not from fitting listener ratings. Use it
to rank tracks inside one run. Do not quote the numbers as population averages.

Lyrics are not used at all. A miserable lyric over a bright backing track reads as
bright.

The two calibration modes are not interchangeable. `fixed` (default) puts zero at
"neutral evidence", so different collections can be compared. `corpus-relative`
re-centres on the current collection's median, which keeps all four quadrants
reachable on a skewed library but makes the numbers incomparable across libraries.

`confidence` is input quality, not accuracy. Short, quiet, atonal or aperiodic
audio scores low.

Tempo detection has an octave problem. Autocorrelation often peaks at both `t` and
`2t`; a 95 BPM reading may really be 184. Tempo carries 0.30 of arousal, so a
wrong octave moves the result.

Clipped masters lose their dynamics before analysis. `peak > 1.0` sets
`clipping_suspected` and the warning says arousal is likely under-reported.

Truncating with `--max-seconds` can delete the character of a track. A quiet intro
will hide grandeur. Analyze the whole file for orchestral material.

The calibration bug that made 264 of 523 tracks read as "melancholic" is written
up in [docs/algorithm.md](docs/algorithm.md), along with the other ones I hit.

## TODO

- No human-annotated ground truth. Everything is validated against synthetic
  stimuli and internal consistency, which is weaker than it sounds.
- The neural sidecar (`--neural`) is a speech model on music. Absolute values are
  compressed; only relative movement is useful. Not wired into the four-axis word.
- `texture` and `grandeur` thresholds were calibrated on a few dozen tracks. They
  probably do not generalise to, say, free jazz or field recordings.
- Batch has no progress callback for the tool API, only stderr.
- No Windows console encoding workaround beyond `-X utf8`.

## Layout

```
index.js              DSH host plugin
cordis.patch.yml      bundle patch
locale/{en,zh}.json   plugin metadata
icon.svg
test/offline.test.mjs plugin contract tests, no DSH runtime needed
python/
  moodio.py           CLI
  music_emotion/      analyzer package
    vamodel.py        valence/arousal from acoustic cues
    texture.py        texture and grandeur scores
    four_axis.py      four axes to an emotion word
    emotion_taxonomy.py   two-axis fallback vocabulary
    features.py       librosa extraction, shared once per track
    pipeline.py       orchestration
    batch.py          resumable directory runs
    report.py         markdown and JSON summary
    neural.py         optional wav2small sidecar
  agent/              usage contract for agents (brief, rules, skills)
  tests/              self-contained suites
```

## Tests

```
npm run test:fast      # plugin + five analyzer suites (64)
npm run test:python    # same six suites CI runs (45)
npm run test:all       # everything (73)
npm test               # plugin contract only (28)
```

The Python suites synthesize their own stimuli or use fixed feature probes, so
none of them need an audio file. CI runs both halves on push.

## License

MIT. The optional neural check is written against `audeering/wav2small`, whose
weights are CC-BY-NC-SA-4.0 and are not included here. See [LICENSE](LICENSE).
