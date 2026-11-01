# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.3.0] — 2026-11-01

### Changed

- Post-analysis background/lyrics research is now mandatory: a missing
  `web.search` provider or a failing query fails the whole call with
  `WEB_RESEARCH_FAILED` instead of returning an analysis without sources. No
  config switch, no offline fallback. Batch fails as a set and returns finished
  acoustic results under `value.partial`.
- Research responses now record the sent `queries` and per-channel `returned` /
  `rejected` counts, so an empty result set is distinguishable from a filter that
  discarded everything.

### Fixed

- Source matching no longer requires the full filename title. Subtitles and
  version tags (`– 新世纪福音战士 OP1 钢琴版`, `(Project CHE Special)`) are stripped
  to a main title and matched by character/token coverage, and an exact title hit
  no longer needs an "official/creation" marker when the provider returns no snippet.

### Fixed

- Replace brightness's lower-bound hard clipping with smooth reference scaling
  in both axes versions and segment scores. Expose raw spectral measurements,
  component contributions, and the normalization method for auditing.
- Verify pitch-boundary hits against wider local searches, preserving confirmed
  pitches and reporting rejected frames plus before/after boundary-note counts.
- Summarize full-track note coverage and chronological sections rather than only
  the first phrase. Match motifs across phrase boundaries using interval/rhythm
  windows, and explain when no reliable repetition is detected.

### Added

- Default whole-song 21-target instrument recognition with local CLAP proposals,
  AST label corroboration, timing evidence and explicit ambiguity/coverage limits.
  Added pinned model setup with checksums and a catalog/status CLI command.
- Default eighteen-axis analysis and combined `music_analysis` evidence for the
  agent, joining timbre, melody, acoustic descriptors and post-analysis web
  context. Batch summaries join context by track key; missing evidence stays explicit.
- Normalize bracketed artist names and download/version suffixes when searching
  song background and lyrics, without sending private directory paths.
- Whole-song melody enabled by default for single and batch CLI/Host analysis: timestamped
  candidate notes, phrase contours, transposed interval repetitions, and grounded
  agent summaries. Uses local pYIN without model downloads; insufficient evidence
  abstains, and dense mixes are explicitly marked as unseparated candidates.
  Use `--no-melody` / `include_melody: false` to skip this stage.
- Opt-in `axes_v3` 18-axis diagnostics for the CLI and Host single/batch tools,
  adding bass presence, noisiness, transient sharpness, tempo stability, tonal
  stability, and build direction. Existing `axes_v2` and four-axis scores stay intact.
- Per-axis reliability in `axes_v3`: missing pulse, weak tonality, or insufficient
  local evidence yields `value: null` with a reason; legacy `HarmonicComplexity`
  identifies its pitch-class-entropy proxy explicitly.

### Changed

- Host tools now run local acoustic analysis or lyric transcription first, then
  automatically search for background and lyrics. Search failures are attached
  to the result instead of suppressing completed local work; no mode switch.

## [2.2.1] — 2026-11-01

### Fixed

- CI: the Python 3.9 job now runs on `ubuntu-22.04`. `ubuntu-latest` moved to
  Ubuntu 26.04, where `actions/python-versions` publishes no 3.9 build, so
  `setup-python` failed before a single test ran.
- CI: the `axes_v2`, `lyrics` and `research` suites added in 2.2.0 are now part
  of the workflow; previously CI never ran them.

### Changed

- CI actions updated to `checkout@v7`, `setup-python@v7` and `setup-node@v7`, and
  the plugin contract job runs on Node 24 (Node 20 reached end of life).

## [2.2.0] — 2026-10-31

### Changed

- Optional twelve-axis diagnostics now process the **entire analyzed track**
  instead of a middle 60-second excerpt. The original four-axis score remains
  unchanged; `max_seconds` still limits both when explicitly requested.
- Standalone `axes_v2` reading defaults to the whole file. Passing a positive
  `--seconds` explicitly selects a middle excerpt for a quick preview.

## [2.1.0] — 2026-10-31

### Added

- Optional `axes_v2` diagnostics from the supplied 12-axis CPU script, based on the
  middle 60 seconds of the already loaded audio. The full-track four-axis
  emotion score and label remain unchanged.
- `--axes-v2` for single-file and batch CLI runs, and `include_axes_v2` for
  the corresponding Host tools; batch records store twelve axes when requested.
- Synthetic and plugin-contract tests for value ranges and legacy score stability.

## [2.0.0] — 2026-10-31

### Changed

- **Breaking:** every Host tool call now searches online for song creation background
  and lyrics before analyzing. Song reviews/comments are excluded from queries and
  cited sources. Web search failure blocks analysis rather than silently falling
  back to an unsourced emotion interpretation.
- Batch mode researches each pending track (maximum 50 per invocation), then
  checks that the processed track list still matches the researched list.
- The standalone Python acoustic CLI remains usable offline; the Host plugin is
  intentionally online-only. Online text never changes four-axis acoustic scores.

## [1.1.0] — 2026-10-31

### Added

- Optional local sung-lyrics transcription via `moodio lyrics` and the
  `transcribe_music_lyrics` Host tool. Results include timestamped ASR guesses;
  lyrics do not influence the acoustic emotion model.
- Explicit dependency/model preflight errors; Moodio never downloads models
  automatically.
- Offline contract tests for model loading, transcription, CLI and tool wiring.

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
- Regression suites that need no audio corpus: 28 plugin contract tests and 45
  analyzer tests, both run in CI.

### Fixed

- **Unportable test broke CI on Linux.** `test_rel_of_normalization` hard-coded
  `C:\music`; on POSIX that is a relative path, so `relpath` returned
  `../C:/music/...` and the assertion failed. The spellings are now derived from
  a temporary directory, and the exact result is asserted.
- **`validate_directionality.py` used `os` without importing it**, a
  `NameError` that only surfaced in CI because the local script did not run that
  suite. The suite is now part of `npm run test:python`, so local and CI run the
  same set.

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
