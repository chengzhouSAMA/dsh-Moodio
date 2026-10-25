# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — 2026-10-25

First public release. The analyzer, the four-axis model, and the DSH plugin are
released together because the plugin is only meaningful with the model that
motivated it.

### Added

- **Four-axis emotion model.** `valence`, `arousal`, `texture` (attack) and
  `grandeur` (scale), replacing a two-axis quadrant mapping. This is what keeps
  `悲壮` distinct from `悲伤` and `热烈` from `欢快`.
- **`grandeur` dimension** computed from sustained-loudness share, dynamic range
  and spectral contrast. Orchestral power comes from scale, not grit; without
  this axis a 34.6 dB-range epic was labelled *quiet*.
- **`texture` dimension** computed from spectral flatness, percussive ratio,
  spectral flux, loudness and brightness.
- **`batch` command** for resumable library analysis (JSONL, parallel workers,
  progress on stderr).
- **`report` command** producing Markdown plus machine-readable JSON.
- **DSH Host plugin** registering `analyze_music_emotion` and
  `batch_analyze_music`. The analyzer ships inside the package and is resolved
  from the module URL, so installation needs no path configuration.
- **Agent contract** under `python/agent/` (brief, rules, skills) with a
  documented exit-code table and structured JSON errors.
- Regression suites that need no audio corpus: 25 plugin contract tests and 34
  analyzer tests, both run in CI.

### Fixed

- **Valence was systematically negative.** `VALENCE_NEUTRAL` was set to `0.47`
  from intuition instead of the arithmetic fixed point `Σweights × 0.5 = 0.25`.
  Consequence: corpus valence mean `−0.222`, **264 of 523** tracks labelled
  "melancholic", and only **2 of 4** quadrants ever reachable. Constants are now
  derived and asserted by `tests/validate_calibration.py`.
- **`rms_db` normalisation range did not match real music.** Configured
  `(-38, -6)` mapped mastered audio to `mean 0.89`, saturating the channel.
  Re-baselined to `(-26, -4)` against measured distribution.
- **`neutral` ignored arousal.** A track with clear arousal but ambiguous valence
  fell into `neutral`, discarding a whole axis. Both dimensions must now be
  inside the dead zone.
- **`spectral_contrast` was double-log-scaled.** Passing
  `amplitude_to_db(|S|, ref=max)` produced `0.0` instead of `27.77`, silently
  zeroing 25 % of the grandeur weight. The magnitude spectrum is now passed
  directly, as librosa expects.
- **Weak valence over-claimed a direction.** A first pass at the discrete
  vocabulary labelled `V = −0.05 / A = +0.40` as "tense", producing 258 "tense"
  tracks against 115 "calm" on a pop/electronic library. A `|V| ≤ 0.10` weak band
  now reports character only.
- **Plugin `render` omissions.** The tool registry requires
  `output { schema, render }`; omitting `render` failed activation, and a
  caption-only `render` replaced the JSON so a successful analysis looked empty.
  `render` now returns the full result and the requirement is asserted in tests.
- **`include_segments` returned `[]`.** `--fields` also filters `segments`, so
  the timeline was never requested and an empty array was indistinguishable from
  "no timeline". The field is now added when the timeline is asked for.

### Known limitations

Documented in [docs/algorithm.md](docs/algorithm.md#7-known-limitations): tempo
octave ambiguity, clipped masters under-reporting arousal, truncation deleting
grandeur, no lyrics or cultural context, `confidence` measuring input quality,
and the optional neural sidecar being a speech model whose weights are licensed
separately and not bundled.
