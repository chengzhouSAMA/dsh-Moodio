# dsh-Moodio

The Python acoustic analyzer works offline. The DSH Host plugin requires online
searches for song creation background and lyrics before every tool use, then
returns acoustic emotion coordinates plus source links. Song reviews/comments
are excluded from search queries and returned sources.

It comes in two pieces: a Python analyzer (`python/moodio.py`) and a DSH host
plugin exposing three tools (emotion, batch, and optional lyrics transcription).
Search failure stops Host analysis; the standalone Python acoustic CLI stays offline.

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

## Twelve-axis diagnostics (opt-in)

Pass `include_axes_v2: true` to `analyze_music_emotion` or
`batch_analyze_music`, or `--axes-v2` to the Python CLI. The supplied
`music axes v2` algorithm adds twelve independent acoustic diagnostics:
Valence, Energy, Brightness, Warmth, Dissonance, Tempo, RhythmicRegularity,
Percussiveness, Density, Grandeur, BuildContrast, and HarmonicComplexity.
It scores the **entire analyzed audio** from the already loaded waveform; the
original four-axis label remains unchanged. Only an explicit `max_seconds`
option limits both analyses to part of a track. Twelve-axis values
are 0..1 measurements with different normalization from `emotion_4d` and must
not be treated as interchangeable. Tempo/Density/Dissonance retain raw units.

## Install

```
git clone https://github.com/chengzhouSAMA/dsh-Moodio.git
cd dsh-Moodio
pip install -r python/requirements.txt
```

Python 3.9 or newer. The plugin half also needs Node 18+ and an available
Host `web.search` provider. The Host sends only the cleaned audio filename,
never local directory paths, in its two searches per song. If the web provider
fails, the plugin returns `WEB_RESEARCH_FAILED` without analyzing.

As a DSH bundle (registers `analyze_music_emotion`, `batch_analyze_music`, and
`transcribe_music_lyrics`):

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
python moodio.py analyze --path "song.flac" --axes-v2 --fields emotion,emotion_4d,axes_v2
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
analyze_music_emotion(path: "/music/track.flac", include_axes_v2: true)
batch_analyze_music(path: "/music", workers: 6)
transcribe_music_lyrics(path: "/music/track.flac", model_dir: "/models/faster-whisper")
```

Each Host tool call searches creation background and lyrics anew; it returns
source links and short background previews in `research`. It never queries song
reviews/comments. These search results are unverified leads, not official lyrics
or a human-rated emotion label. Batch returns `research_by_track` for each
pending song, capped at 50 per call; pass `limit: 50` and resume for larger sets.
The Python CLI remains offline unless you explicitly run a separate web workflow.

## Lyrics (opt-in)

Install the optional `faster-whisper` package separately; prepare a **local**
CTranslate2 Whisper model directory containing `model.bin` first (see the
[upstream model documentation](https://github.com/SYSTRAN/faster-whisper)).
No model weights are downloaded automatically; Host song searches still require
network access.

```
python -m pip install faster-whisper
python python/moodio.py lyrics --path "song.mp3" --model-dir "/models/faster-whisper" --language en
```

Alternatively set `MOODIO_LYRICS_MODEL_DIR` instead of passing `--model-dir`.
The CLI returns JSON with the guessed text and timestamps; use `--device cuda`
only if a compatible GPU runtime is installed. It transcribes the given audio
without automatically separating vocals. Sung words over dense instrumentation
may be missed or invented, so verify against the recording. The lyrics output
does **not** modify the four-axis emotion score.

## Caveats

This is a rule-based prior, not a trained model. The weights come from published
directional results in music psychology, not from fitting listener ratings. Use it
to rank tracks inside one run. Do not quote the numbers as population averages.

Lyrics are not fed into the emotion model. A miserable lyric over bright backing
can still score as bright; transcription is a separate, optional output.

The two calibration modes are not interchangeable. `fixed` (default) puts zero at
"neutral evidence", so different collections can be compared. `corpus-relative`
re-centres on the current collection's median, which keeps all four quadrants
reachable on a skewed library but makes the numbers incomparable across libraries.

`confidence` is input quality, not accuracy. Short, quiet, atonal or aperiodic
audio scores low.

Tempo detection has an octave problem. Autocorrelation often peaks at both `t` and
`2t`; a 95 BPM reading may really be 184. Tempo carries 0.30 of arousal, so a
wrong octave moves the result.

Clipped masters lose their dynamics before analysis. A peak amplitude of
`>= 0.995` sets `clipping_suspected`; the warning notes that dynamic features
may be distorted.

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
    axes_v2.py        optional 12-axis diagnostics (source script integrated)
    batch.py          resumable directory runs
    report.py         markdown and JSON summary
    neural.py         optional wav2small sidecar
    lyrics.py         optional local singing transcription
  agent/              usage contract for agents (brief, rules, skills)
  tests/              self-contained suites
```

## Tests

```
npm run test:fast      # plugin + fast Python suites, including lyrics stubs
npm run test:python    # all Python suites and directionality checks
npm run test:all       # Python + plugin
npm test               # plugin contract only
```

The Python suites synthesize their own stimuli or use fixed feature probes, so
none of them need an audio file. CI runs both halves on push.

## License

MIT. The optional neural check is written against `audeering/wav2small`, whose
weights are CC-BY-NC-SA-4.0 and are not included here. See [LICENSE](LICENSE).
