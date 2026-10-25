/**
 * Host half: registers music-emotion tools backed by the local Python analyzer.
 *
 * Design notes
 * ------------
 * The Python side (moodio.py) already speaks a strict agent contract: JSON on
 * stdout, structured JSON errors on stderr, and a documented exit-code table.
 * This plugin is a thin, careful wrapper — it must not re-implement analysis
 * or invent a second error vocabulary.
 *
 * Three things it deliberately does NOT do:
 *   1. Never builds a shell command string. Windows paths contain spaces and
 *      CJK characters; argv arrays sidestep quoting entirely.
 *   2. Never treats "stdout is not JSON" as success. A truncated stream looks
 *      like valid text but is not a result.
 *   3. Never lets an exception escape `execute`. A thrown body becomes an
 *      opaque harness error; callers get a structured failure instead.
 *
 * Configuration comes from the bundle patch `config` block (projectDir,
 * pythonPath, analyzer, defaultTimeoutMs, maxOutputBytes). No `Config` schema
 * is exported on purpose: the loader still passes `config` to `apply`, and
 * skipping the schema removes a failure mode that would block activation.
 *
 * Portability: the analyzer lives in ./python inside this package and is
 * resolved from the module URL, so installing the bundle needs no path setup.
 */

import { existsSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

/** Exit codes mirrored from moodio.py; kept in sync deliberately. */
const EXIT_CODES = {
  0: 'success',
  1: 'general error / batch finished with failures',
  2: 'usage error',
  10: 'auth failed',
  11: 'permission denied',
  20: 'resource not found',
  30: 'conflict / precondition failed',
}

/**
 * The analyzer ships inside this package (`python/`), so the default project
 * directory is resolved from the module location rather than hard-coded.
 * That is what makes the bundle portable: clone anywhere, install, done.
 */
const MODULE_DIR = path.dirname(fileURLToPath(import.meta.url))
const BUNDLED_PYTHON_DIR = path.join(MODULE_DIR, 'python')

const DEFAULTS = {
  // Empty means "use the bundled python/ directory". A caller may still point
  // at a different checkout, e.g. during development.
  projectDir: '',
  pythonPath: 'python',
  analyzer: 'moodio.py',
  defaultTimeoutMs: 600_000,
  maxOutputBytes: 4 * 1024 * 1024,
}

/** Fields a caller may ask `analyze` to return. Mirrors moodio's _TOP_FIELDS. */
const RESULT_FIELDS = [
  'emotion', 'emotion_4d', 'texture', 'grandeur', 'arc',
  'quality', 'features', 'modes', 'deltas', 'clip', 'segments', 'warnings',
]

export const name = 'music-emotion'

/** Required Host services. Declared so the row waits for them instead of racing. */
export const inject = ['tools', 'subprocess']

/**
 * Collect-mode readers are offset based and non-consuming, so read once from 0.
 * A lossy read must never be parsed as a result.
 */
function readCollected(reader) {
  if (!reader) return { text: '', lossy: false }
  try {
    const out = reader.readFrom(0)
    return { text: out?.text ?? '', lossy: Boolean(out?.lossy) }
  } catch {
    return { text: '', lossy: true }
  }
}

/** Extract the structured error object moodio writes to stderr, if present. */
function parseStructuredError(text) {
  const trimmed = (text ?? '').trim()
  if (!trimmed) return undefined
  const candidates = [trimmed.slice(trimmed.indexOf('{')), ...trimmed.split('\n').filter((l) => l.trim().startsWith('{'))]
  for (const candidate of candidates) {
    if (!candidate || candidate === trimmed.slice(-1)) continue
    try {
      const parsed = JSON.parse(candidate)
      if (parsed && typeof parsed === 'object' && parsed.error === true) return parsed
    } catch {
      /* try the next candidate */
    }
  }
  return undefined
}

/**
 * Run the analyzer and normalize the outcome into
 *   { ok: true, data } | { ok: false, code, message, suggestion, exitCode? }
 * Never throws.
 */
async function runAnalyzer(ctx, cfg, argv, { timeoutMs, signal } = {}) {
  const script = path.join(cfg.projectDir, cfg.analyzer)
  if (!existsSync(script)) {
    return {
      ok: false,
      code: 'ANALYZER_NOT_FOUND',
      message: `analyzer script not found: ${script}`,
      suggestion:
        'The bundled python/ directory is missing. Reinstall the bundle, or set "projectDir" in the plugin row config to a folder containing moodio.py.',
    }
  }

  const limit = Number(timeoutMs ?? cfg.defaultTimeoutMs) || DEFAULTS.defaultTimeoutMs
  const controller = new AbortController()
  let timedOut = false
  const onAbort = () => controller.abort()
  if (signal) {
    if (signal.aborted) onAbort()
    else signal.addEventListener('abort', onAbort, { once: true })
  }
  const timer = setTimeout(() => {
    timedOut = true
    controller.abort()
  }, limit)

  const cleanup = () => {
    clearTimeout(timer)
    if (signal) signal.removeEventListener?.('abort', onAbort)
  }

  let handle
  try {
    handle = ctx.subprocess.spawn({
      argv: [cfg.pythonPath, '-X', 'utf8', script, ...argv],
      cwd: cfg.projectDir,
      stdio: {
        stdin: 'ignore',
        stdout: { maxBytes: cfg.maxOutputBytes ?? DEFAULTS.maxOutputBytes },
        stderr: { maxBytes: cfg.maxOutputBytes ?? DEFAULTS.maxOutputBytes },
      },
      graceMs: 10_000,
      signal: controller.signal,
    })
  } catch (error) {
    cleanup()
    return {
      ok: false,
      code: 'SPAWN_FAILED',
      message: `failed to start python: ${error instanceof Error ? error.message : String(error)}`,
      suggestion: 'Check "pythonPath" in the plugin row config; it must be on PATH or absolute.',
    }
  }

  let exitCode = null
  try {
    const outcome = await handle.done
    exitCode = outcome?.exitCode ?? null
  } catch (error) {
    cleanup()
    return timedOut
      ? {
          ok: false,
          code: 'TIMEOUT',
          message: `analysis exceeded ${Math.round(limit / 1000)}s and was terminated`,
          suggestion:
            'Raise "defaultTimeoutMs", or pass max_seconds to analyze only the first part of a long file.',
        }
      : {
          ok: false,
          code: 'PROCESS_FAILED',
          message: `python exited abnormally: ${error instanceof Error ? error.message : String(error)}`,
          suggestion: 'Run the same command manually to see the full traceback.',
        }
  } finally {
    cleanup()
  }

  const stdout = readCollected(handle.collected?.stdout)
  const stderr = readCollected(handle.collected?.stderr)

  if (timedOut) {
    return {
      ok: false,
      code: 'TIMEOUT',
      message: `analysis exceeded ${Math.round(limit / 1000)}s and was terminated`,
      suggestion:
        'Raise "defaultTimeoutMs", or pass max_seconds to analyze only the first part of a long file.',
    }
  }

  if (exitCode !== 0) {
    const structured = parseStructuredError(stderr.text)
    return {
      ok: false,
      code: structured?.code ?? `EXIT_${exitCode}`,
      message: structured?.message ?? (stderr.text.trim() || `analyzer exited with code ${exitCode}`),
      suggestion:
        structured?.suggestion ?? `exit code ${exitCode}: ${EXIT_CODES[exitCode] ?? 'unknown'}`,
      exitCode,
    }
  }

  if (stdout.lossy) {
    return {
      ok: false,
      code: 'OUTPUT_TRUNCATED',
      message:
        'analyzer output exceeded the capture limit and was truncated; refusing to parse a partial document',
      suggestion:
        'A very long file with small timeline windows produces large output. Raise segment_sec, set include_segments false, or raise maxOutputBytes.',
    }
  }

  const text = stdout.text.trim()
  if (!text) {
    return {
      ok: false,
      code: 'EMPTY_OUTPUT',
      message: 'analyzer exited 0 but produced no output',
      suggestion: 'Run the same command manually; something may have redirected stdout.',
    }
  }

  try {
    return { ok: true, data: JSON.parse(text) }
  } catch {
    return {
      ok: false,
      code: 'INVALID_JSON',
      message: 'analyzer stdout was not valid JSON',
      suggestion: 'Non-JSON text reached stdout. Re-run manually to locate it.',
    }
  }
}

/** Build a compact, decision-ready summary instead of dumping the whole report. */
function summarizeAnalysis(data, { includeSegments, includeEvidence }) {
  const emo = data.emotion ?? {}
  const four = data.emotion_4d ?? {}
  const quality = data.quality ?? {}
  const out = {
    emotion: four.emotion ?? emo.emotion,
    emotion_id: four.emotion_id ?? emo.emotion_id,
    directional: four.directional,
    description: four.desc ?? emo.desc,
    axes: {
      valence: emo.valence,
      arousal: emo.arousal,
      arousal_band: four.arousal_band,
      texture: four.texture,
      texture_band: four.texture_band,
      grandeur: four.grandeur,
      grandeur_band: four.grandeur_band,
    },
    arc: data.arc,
    confidence: quality.confidence,
    clip: data.clip,
    warnings: data.warnings ?? [],
  }
  if (four.texture_parts) out.texture_parts = four.texture_parts
  else if (data.texture?.parts) out.texture_parts = data.texture.parts
  if (four.grandeur_parts) out.grandeur_parts = four.grandeur_parts
  else if (data.grandeur?.parts) out.grandeur_parts = data.grandeur.parts

  // Surface the axis bands even when the analyzer omitted emotion_4d (older runs).
  if (!four.texture_band && data.texture?.band) out.axes.texture_band = data.texture.band
  if (!four.grandeur_band && data.grandeur?.band) out.axes.grandeur_band = data.grandeur.band
  if (includeSegments && Array.isArray(data.segments)) {
    out.segments = data.segments.map((s) => ({
      start_sec: s.start_sec,
      end_sec: s.end_sec,
      valence: s.valence,
      arousal: s.arousal,
      label: s.label,
    }))
  } else if (data.segments && !Array.isArray(data.segments)) {
    out.segments = data.segments
  } else if (!includeSegments) {
    // Say the timeline was omitted rather than leaving a missing key: a caller
    // must be able to tell "not requested" from "analyzer produced nothing".
    out.segments = { omitted: true, hint: 'pass include_segments: true for the per-window timeline' }
  } else {
    out.segments = []
  }
  if (includeEvidence) {
    if (data.deltas) out.deltas = data.deltas
    if (data.features) out.features = data.features
    if (data.modes) out.modes = data.modes
  }
  return out
}

function fail(value) {
  const hint = value.suggestion ? `\nHint: ${value.suggestion}` : ''
  return {
    isError: true,
    value,
    content: [
      {
        type: 'text',
        text: `Music emotion analysis failed [${value.code}]: ${value.message}${hint}`,
      },
    ],
  }
}

export function apply(ctx, config = {}) {
  const cfg = { ...DEFAULTS, ...(config ?? {}) }
  // Empty / missing projectDir falls back to the analyzer bundled in this
  // package, so a plain `install_bundle` works with no path configuration.
  if (!cfg.projectDir) cfg.projectDir = BUNDLED_PYTHON_DIR
  // Resolve once: an absolute path keeps cwd handling unambiguous even when
  // the caller passes a relative directory.
  cfg.projectDir = path.resolve(cfg.projectDir)

  ctx.effect(() => {
    const disposers = []

    disposers.push(
      ctx.tools.register({
        name: 'analyze_music_emotion',
        description:
          'Analyze one audio file and return its emotion. Reports four axes: valence (negative..positive), ' +
          'arousal (calm..energetic), texture (attack: distortion and percussion, high for rock/electronic), ' +
          'and grandeur (scale: dynamic range plus sustained loudness, high for orchestral/epic). ' +
          'The emotion word derives from all four, which keeps sombre-and-grand (悲壮) distinct from ' +
          'intimate sadness (悲伤), and triumphant (昂扬) distinct from cheerful (欢快). ' +
          'Use this instead of guessing emotion from a title or artist: it reads the actual audio. ' +
          'Limits: lyrics and semantics are NOT considered; values are relative coordinates from an ' +
          'acoustic prior, not human rating averages, so compare tracks within one run rather than ' +
          'quoting absolute numbers. When analysing epic or orchestral tracks, do not truncate with ' +
          'max_seconds: a quiet intro can hide the grandeur that defines the emotion.',
        parameters: {
          type: 'object',
          properties: {
            path: {
              type: 'string',
              description: 'Absolute path to an audio file (mp3/flac/wav/ogg/aiff).',
            },
            segment_sec: {
              type: 'number',
              description:
                'Timeline window length in seconds (default 10). Raise to 20-30 for long tracks to reduce output size.',
            },
            hop_sec: {
              type: 'number',
              description: 'Timeline step in seconds (default 10).',
            },
            max_seconds: {
              type: 'number',
              description: 'Analyze only the first N seconds (0 = whole file, default 0).',
            },
            include_segments: {
              type: 'boolean',
              description: 'Include the per-window emotion timeline (default false).',
            },
            include_evidence: {
              type: 'boolean',
              description:
                'Include raw acoustic features and the per-axis contributing parts (default false).',
            },
          },
          required: ['path'],
          additionalProperties: false,
        },
        output: {
          schema: { type: 'object', additionalProperties: true },
          // `render` is MANDATORY: the registry rejects a tool that omits it
          // ("must declare output { schema, render, presentationMeta? }").
          // It is also the only thing the model reads, so it must carry the
          // full result — an earlier version returned a one-line caption and
          // the caller saw only that caption for a successful run.
          render: (_args, value) => {
            if (value && typeof value === 'object') {
              return [{ type: 'text', text: JSON.stringify(value, null, 2) }]
            }
            return [{ type: 'text', text: 'music emotion analysis returned no value' }]
          },
        },
        async execute(args, exec) {
          const input = args ?? {}
          if (typeof input.path !== 'string' || !input.path.trim()) {
            return fail({
              code: 'INVALID_ARGUMENT',
              message: 'path is required',
              suggestion: 'Pass an absolute path to an audio file.',
            })
          }

          const argv = ['analyze', '--path', input.path]
          if (Number.isFinite(input.segment_sec)) argv.push('--segment-sec', String(input.segment_sec))
          if (Number.isFinite(input.hop_sec)) argv.push('--hop-sec', String(input.hop_sec))
          if (Number.isFinite(input.max_seconds)) argv.push('--max-seconds', String(input.max_seconds))

          const fields = input.include_evidence
            ? RESULT_FIELDS
            : ['emotion', 'emotion_4d', 'texture', 'grandeur', 'arc', 'quality', 'clip', 'deltas']
          // `--fields` also filters `segments`; asking for the timeline without
          // requesting the field makes the analyzer omit it and the caller gets
          // an empty array that looks like "no timeline".
          if (input.include_segments && !fields.includes('segments')) fields.push('segments')
          argv.push('--fields', fields.join(','))
          if (!input.include_segments) argv.push('--no-segments')

          const result = await runAnalyzer(ctx, cfg, argv, { signal: exec?.signal })
          if (!result.ok) return fail(result)

          const value = summarizeAnalysis(result.data, {
            includeSegments: Boolean(input.include_segments),
            includeEvidence: Boolean(input.include_evidence),
          })
          return {
            isError: false,
            value,
            content: [{ type: 'text', text: JSON.stringify(value, null, 2) }],
          }
        },
      }),
    )

    disposers.push(
      ctx.tools.register({
        name: 'batch_analyze_music',
        description:
          'Analyze every audio file under a directory, appending one JSON record per track to a JSONL file. ' +
          'Resumable: re-running skips tracks that already succeeded and retries the failed ones, so a long ' +
          'run survives interruption. Progress is not returned; the returned value is the final summary. ' +
          'Use this instead of calling analyze_music_emotion in a loop.',
        parameters: {
          type: 'object',
          properties: {
            path: {
              type: 'string',
              description: 'Absolute path to the directory to scan recursively.',
            },
            out: {
              type: 'string',
              description: 'JSONL output path. Defaults to moodio_batch.jsonl in the current directory.',
            },
            workers: {
              type: 'integer',
              description: 'Parallel worker processes (default 4). Raise only with spare cores.',
            },
            limit: {
              type: 'integer',
              description: 'Process only the first N pending tracks (0 = all). Use for a trial run.',
            },
            segment_sec: { type: 'number', description: 'Window length in seconds (default 10).' },
            hop_sec: { type: 'number', description: 'Window step in seconds (default 10).' },
            max_seconds: {
              type: 'number',
              description: 'Analyze only the first N seconds per track (0 = all).',
            },
            timeout_ms: {
              type: 'number',
              description: 'Override the batch timeout for this call. Large libraries need a big value.',
            },
            no_resume: {
              type: 'boolean',
              description:
                'Discard previous results and start over. Destructive: the output file is deleted first.',
            },
          },
          required: ['path'],
          additionalProperties: false,
        },
        output: {
          schema: { type: 'object', additionalProperties: true },
          render: (_args, value) => {
            if (value && typeof value === 'object') {
              return [{ type: 'text', text: JSON.stringify(value, null, 2) }]
            }
            return [{ type: 'text', text: 'batch music analysis returned no value' }]
          },
        },
        async execute(args, exec) {
          const input = args ?? {}
          if (typeof input.path !== 'string' || !input.path.trim()) {
            return fail({
              code: 'INVALID_ARGUMENT',
              message: 'path is required',
              suggestion: 'Pass an absolute path to a directory containing audio files.',
            })
          }

          const argv = ['batch', '--path', input.path, '--quiet']
          if (typeof input.out === 'string' && input.out.trim()) argv.push('--out', input.out)
          if (Number.isFinite(input.workers)) argv.push('--workers', String(input.workers))
          if (Number.isFinite(input.limit)) argv.push('--limit', String(input.limit))
          if (Number.isFinite(input.segment_sec)) argv.push('--segment-sec', String(input.segment_sec))
          if (Number.isFinite(input.hop_sec)) argv.push('--hop-sec', String(input.hop_sec))
          if (Number.isFinite(input.max_seconds)) argv.push('--max-seconds', String(input.max_seconds))
          if (input.no_resume === true) argv.push('--no-resume')

          const result = await runAnalyzer(ctx, cfg, argv, {
            timeoutMs: Number.isFinite(input.timeout_ms) ? input.timeout_ms : cfg.defaultTimeoutMs,
            signal: exec?.signal,
          })
          if (!result.ok) return fail(result)

          // Drop the CLI's agent-context tail; it is noise inside a tool result.
          const value = { ...result.data }
          delete value.skills
          delete value.issue

          return {
            isError: false,
            value,
            content: [{ type: 'text', text: JSON.stringify(value, null, 2) }],
          }
        },
      }),
    )

    return () => {
      for (const dispose of disposers) {
        try {
          dispose()
        } catch {
          /* a failing disposer must not block the rest */
        }
      }
    }
  }, 'music-emotion tools')
}
