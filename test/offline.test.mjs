/**
 * Offline contract tests for the music-emotion plugin.
 *
 * No DSH runtime involved: a stub `ctx` supplies the two services and fakes a
 * subprocess. This lets us exercise every failure branch — truncated output,
 * non-zero exit with a structured error, timeout, spawn failure, missing
 * script — before installing anything.
 *
 * Run: node test/offline.test.mjs
 */

import assert from 'node:assert/strict'
import { existsSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

const here = path.dirname(fileURLToPath(import.meta.url))
const pkgRoot = path.dirname(here)

// Windows: dynamic import() needs a file:// URL; a bare drive path is rejected.
const { apply, inject, name } = await import(pathToFileURL(path.join(pkgRoot, 'index.js')).href)

let passed = 0
let failed = 0

function check(label, fn) {
  try {
    fn()
    passed += 1
    console.log(`  [PASS] ${label}`)
  } catch (error) {
    failed += 1
    console.log(`  [FAIL] ${label}\n         ${error.message}`)
  }
}

async function checkAsync(label, fn) {
  try {
    await fn()
    passed += 1
    console.log(`  [PASS] ${label}`)
  } catch (error) {
    failed += 1
    console.log(`  [FAIL] ${label}\n         ${error.message}`)
  }
}

/** Minimal stand-in for a SubprocessHandle. */
function fakeHandle({ stdout = '', stderr = '', exitCode = 0, lossyStdout = false, rejectDone } = {}) {
  return {
    collected: {
      stdout: { readFrom: () => ({ text: stdout, nextOffset: stdout.length, lossy: lossyStdout }) },
      stderr: { readFrom: () => ({ text: stderr, nextOffset: stderr.length, lossy: false }) },
    },
    done: rejectDone ? Promise.reject(rejectDone) : Promise.resolve({ exitCode, signal: null }),
    terminate() {},
    waitForExit: async () => true,
  }
}

/**
 * Build a stub ctx. `spawnImpl` decides what each run returns, and `onSpawn`
 * records the argv so we can assert on how the command was assembled.
 */
function makeCtx({ spawnImpl, config = {}, onSpawn, searchImpl, listImpl } = {}) {
  const registered = new Map()
  const disposers = []
  const ctx = {
    web: {
      search(request, signal) {
        return searchImpl ? searchImpl(request, signal) : Promise.resolve({ sources: [] })
      },
    },
    tools: {
      register(definition) {
        assert.ok(definition.name, 'tool must have a name')
        assert.ok(definition.description, 'tool must have a description')
        assert.ok(definition.parameters, 'tool must have parameters')
        assert.ok(definition.output?.schema, 'tool must declare output.schema')
        // The live registry rejects a row without render:
        //   TypeError: tool "x" must declare output { schema, render, presentationMeta? }
        assert.equal(
          typeof definition.output.render,
          'function',
          `tool ${definition.name} must declare output.render (registry requirement)`,
        )
        assert.equal(typeof definition.execute, 'function')
        assert.ok(!registered.has(definition.name), `duplicate tool ${definition.name}`)
        registered.set(definition.name, definition)
        return () => registered.delete(definition.name)
      },
    },
    subprocess: {
      spawn(spec) {
        if (onSpawn) onSpawn(spec)
        if (spec.argv.includes('--list-only')) {
          return listImpl ? listImpl(spec) : fakeHandle({ stdout: JSON.stringify({ pending: ['song.mp3'] }) })
        }
        if (spawnImpl) return spawnImpl(spec)
        return fakeHandle()
      },
    },
    effect(fn) {
      const dispose = fn()
      disposers.push(dispose)
      return () => disposers.forEach((d) => d?.())
    },
  }
  apply(ctx, config)
  return { ctx, registered, disposers }
}

/** Create a temp "project" so the existsSync(script) guard passes. */
function makeProject() {
  const dir = mkdtempSync(path.join(tmpdir(), 'moodio-plugin-'))
  writeFileSync(path.join(dir, 'moodio.py'), '# stub\n', 'utf8')
  return dir
}

const project = makeProject()
const baseConfig = { projectDir: project }

console.log('='.repeat(62))
console.log('music-emotion plugin · offline contract tests')
console.log('='.repeat(62))

// ---------------------------------------------------------------- metadata
console.log('\n[t1] module surface')
check('exports name', () => assert.equal(name, 'music-emotion'))
check('injects tools, subprocess and web', () => {
  assert.ok(Array.isArray(inject))
  assert.ok(inject.includes('tools'))
  assert.ok(inject.includes('subprocess'))
  assert.ok(inject.includes('web'))
})
check('apply is a function', () => assert.equal(typeof apply, 'function'))

// ---------------------------------------------------------------- registration
console.log('\n[t2] tool registration')
{
  const { registered } = makeCtx({ config: baseConfig })
  check('registers three tools', () => {
    assert.ok(registered.has('analyze_music_emotion'))
    assert.ok(registered.has('batch_analyze_music'))
    assert.ok(registered.has('transcribe_music_lyrics'))
  })
  check('lyrics require an audio path', () => {
    const params = registered.get('transcribe_music_lyrics').parameters
    assert.deepEqual(params.required, ['path'])
    assert.equal(params.additionalProperties, false)
  })
  check('analyze requires path', () => {
    const params = registered.get('analyze_music_emotion').parameters
    assert.deepEqual(params.required, ['path'])
    assert.equal(params.additionalProperties, false)
  })
  check('batch requires path', () => {
    const params = registered.get('batch_analyze_music').parameters
    assert.deepEqual(params.required, ['path'])
  })
}

// ---------------------------------------------------------------- argv shape
console.log('\n[t3] argv assembly (never a shell string)')
await checkAsync('passes argv array with CJK path intact', async () => {
  let seen
  const { registered } = makeCtx({
    config: baseConfig,
    onSpawn: (spec) => {
      seen = spec
    },
    spawnImpl: () =>
      fakeHandle({
        stdout: JSON.stringify({ emotion: { emotion: '悲伤', valence: -0.2, arousal: -0.1 }, emotion_4d: { emotion: '悲壮' } }),
      }),
  })
  const target = 'C:\\歌曲\\邓垚 - 诀别书.mp3'
  await registered.get('analyze_music_emotion').execute({ path: target }, { signal: new AbortController().signal })
  assert.ok(Array.isArray(seen.argv), 'argv must be an array')
  assert.equal(seen.argv[0], 'python')
  assert.ok(seen.argv.includes(target), 'path must be passed verbatim as one argv element')
  assert.ok(seen.argv.includes('--no-segments'), 'no-segments default')
})

await checkAsync('单曲在运行 Python 前每次联网检索两类资料', async () => {
  const events = []
  const { registered } = makeCtx({
    config: baseConfig,
    searchImpl: async (request) => {
      events.push(request.query)
      return { sources: request.query.includes('lyrics')
        ? [{ url: 'https://lyrics.example/die-for-you', title: 'Die For You Lyrics' }]
        : [{ url: 'https://playvalorant.com/music/die-for-you', title: 'Die For You Official Song' }] }
    },
    onSpawn: () => { events.push('python') },
    spawnImpl: () => fakeHandle({ stdout: JSON.stringify({ emotion: {}, emotion_4d: {} }) }),
  })
  const result = await registered.get('analyze_music_emotion').execute({ path: 'D:\\私人\\Grabbitz - Die For You.mp3' }, {})
  assert.equal(result.isError, false)
  assert.equal(events.length, 3)
  assert.equal(events.at(-1), 'python')
  assert.ok(result.value.research?.background?.found)
  assert.ok(result.value.research?.lyrics?.found)
  assert.ok(events.slice(0, 2).every((q) => !q.includes('私人') && !/review|comment|评论/.test(q)))
})

await checkAsync('lyrics forwards local model and audio path as separate argv', async () => {
  let seen
  const searches = []
  const { registered } = makeCtx({
    config: baseConfig,
    searchImpl: async (request) => { searches.push(request.query); return { sources: [] } },
    onSpawn: (spec) => { seen = spec },
    spawnImpl: () => fakeHandle({ stdout: JSON.stringify({ text: 'heard words', language: 'en', segments: [] }) }),
  })
  const track = 'C:\\歌曲\\a.mp3'
  const model = 'D:\\models\\whisper-tiny'
  const result = await registered.get('transcribe_music_lyrics').execute({ path: track, model_dir: model, language: 'en' }, {})
  assert.equal(result.isError, false)
  assert.equal(result.value.text, 'heard words')
  assert.equal(seen.argv[seen.argv.indexOf('--path') + 1], track)
  assert.equal(seen.argv[seen.argv.indexOf('--model-dir') + 1], model)
  assert.equal(seen.argv[seen.argv.indexOf('--language') + 1], 'en')
  assert.equal(searches.length, 2)
  assert.ok(result.value.research?.lyrics)
})

console.log('\n[t4] field selection')
await checkAsync('default fields omit raw features', async () => {
  let seen
  const { registered } = makeCtx({
    config: baseConfig,
    onSpawn: (spec) => {
      seen = spec
    },
    spawnImpl: () => fakeHandle({ stdout: JSON.stringify({ emotion: {}, emotion_4d: {} }) }),
  })
  await registered.get('analyze_music_emotion').execute({ path: 'x.mp3' }, {})
  const fields = seen.argv[seen.argv.indexOf('--fields') + 1]
  assert.ok(!fields.includes('features'), 'raw features should be opt-in')
  assert.ok(fields.includes('warnings'), 'audio warnings must not be hidden')
  assert.ok(fields.includes('emotion_4d'), 'four-axis result must be requested')
})

await checkAsync('include_evidence adds raw feature fields', async () => {
  let seen
  const { registered } = makeCtx({
    config: baseConfig,
    onSpawn: (spec) => {
      seen = spec
    },
    spawnImpl: () => fakeHandle({ stdout: JSON.stringify({ emotion: {}, emotion_4d: {} }) }),
  })
  await registered.get('analyze_music_emotion').execute({ path: 'x.mp3', include_evidence: true }, {})
  const fields = seen.argv[seen.argv.indexOf('--fields') + 1]
  assert.ok(fields.includes('features'))
  assert.ok(fields.includes('modes'))
})

await checkAsync('十二轴仅在显式请求时随四轴一起返回', async () => {
  let seen
  const { registered } = makeCtx({
    config: baseConfig,
    onSpawn: (spec) => { seen = spec },
    spawnImpl: () => fakeHandle({ stdout: JSON.stringify({
      emotion: { valence: -0.1, arousal: 0.2 },
      emotion_4d: { emotion: '强烈', texture: 0.6, grandeur: 0.4 },
      axes_v2: { axes: { Energy: { value: 0.7 } }, sampling: 'middle_excerpt' },
    }) }),
  })
  const result = await registered.get('analyze_music_emotion').execute({ path: 'x.mp3', include_axes_v2: true }, {})
  assert.equal(result.value.emotion, '强烈')
  assert.equal(result.value.axes_v2.axes.Energy.value, 0.7)
  assert.ok(seen.argv.includes('--axes-v2'))
  assert.ok(seen.argv[seen.argv.indexOf('--fields') + 1].includes('axes_v2'))
})

await checkAsync('include_segments requests the segments field', async () => {
  // Regression: --fields filters segments too. Asking for the timeline without
  // requesting the field yields an empty array that looks like "no timeline".
  let seen
  const { registered } = makeCtx({
    config: baseConfig,
    onSpawn: (spec) => {
      seen = spec
    },
    spawnImpl: () => fakeHandle({ stdout: JSON.stringify({ emotion: {}, emotion_4d: {} }) }),
  })
  await registered
    .get('analyze_music_emotion')
    .execute({ path: 'x.mp3', include_segments: true }, {})
  const fields = seen.argv[seen.argv.indexOf('--fields') + 1]
  assert.ok(fields.includes('segments'), 'segments must be in --fields when the timeline is requested')
  assert.ok(!seen.argv.includes('--no-segments'), '--no-segments must not be passed')
})

// ---------------------------------------------------------------- failures
console.log('\n[t5] failure normalization')

await checkAsync('missing path argument is rejected without spawning', async () => {
  let spawned = false
  const { registered } = makeCtx({
    config: baseConfig,
    spawnImpl: () => {
      spawned = true
      return fakeHandle()
    },
  })
  const result = await registered.get('analyze_music_emotion').execute({}, {})
  assert.equal(result.isError, true)
  assert.equal(result.value.code, 'INVALID_ARGUMENT')
  assert.equal(spawned, false)
})

await checkAsync('联网失败时不启动本地声学分析', async () => {
  let spawned = false
  const { registered } = makeCtx({
    config: baseConfig,
    searchImpl: async () => { throw new Error('network down') },
    spawnImpl: () => { spawned = true; return fakeHandle() },
  })
  const result = await registered.get('analyze_music_emotion').execute({ path: 'song.mp3' }, {})
  assert.equal(result.isError, true)
  assert.equal(result.value.code, 'WEB_RESEARCH_FAILED')
  assert.equal(spawned, false)
})

await checkAsync('non-zero exit with structured stderr is surfaced', async () => {
  const payload = { error: true, code: 'FILE_NOT_FOUND', message: 'no such file', suggestion: 'check path' }
  const { registered } = makeCtx({
    config: baseConfig,
    spawnImpl: () => fakeHandle({ exitCode: 20, stdout: '', stderr: JSON.stringify(payload) }),
  })
  const result = await registered.get('analyze_music_emotion').execute({ path: 'nope.mp3' }, {})
  assert.equal(result.isError, true)
  assert.equal(result.value.code, 'FILE_NOT_FOUND')
  assert.equal(result.value.suggestion, 'check path')
  assert.equal(result.value.exitCode, 20)
})

await checkAsync('structured error survives an earlier noisy stderr line', async () => {
  const payload = { error: true, code: 'BAD_AUDIO', message: 'cannot decode' }
  const { registered } = makeCtx({
    config: baseConfig,
    spawnImpl: () => fakeHandle({ exitCode: 1, stderr: `decoder warning\n${JSON.stringify(payload)}\n` }),
  })
  const result = await registered.get('analyze_music_emotion').execute({ path: 'x.mp3' }, {})
  assert.equal(result.value.code, 'BAD_AUDIO')
})

await checkAsync('structured error survives a log prefix on the same line', async () => {
  const payload = { error: true, code: 'BAD_AUDIO', message: 'cannot decode' }
  const { registered } = makeCtx({
    config: baseConfig,
    spawnImpl: () => fakeHandle({ exitCode: 1, stderr: `decoder warning ${JSON.stringify(payload)}\n` }),
  })
  const result = await registered.get('analyze_music_emotion').execute({ path: 'x.mp3' }, {})
  assert.equal(result.value.code, 'BAD_AUDIO')
})

await checkAsync('non-zero exit without structured stderr still fails', async () => {
  const { registered } = makeCtx({
    config: baseConfig,
    spawnImpl: () => fakeHandle({ exitCode: 1, stdout: '', stderr: 'Traceback: boom' }),
  })
  const result = await registered.get('analyze_music_emotion').execute({ path: 'x.mp3' }, {})
  assert.equal(result.isError, true)
  assert.match(result.value.code, /EXIT_1|Traceback/)
})

await checkAsync('truncated stdout is refused, not parsed', async () => {
  const half = '{"emotion": {"emotion": "悲'
  const { registered } = makeCtx({
    config: baseConfig,
    spawnImpl: () => fakeHandle({ stdout: half, lossyStdout: true }),
  })
  const result = await registered.get('analyze_music_emotion').execute({ path: 'x.mp3' }, {})
  assert.equal(result.isError, true)
  assert.equal(result.value.code, 'OUTPUT_TRUNCATED')
})

await checkAsync('invalid JSON fails with INVALID_JSON', async () => {
  const { registered } = makeCtx({
    config: baseConfig,
    spawnImpl: () => fakeHandle({ stdout: 'not json at all' }),
  })
  const result = await registered.get('analyze_music_emotion').execute({ path: 'x.mp3' }, {})
  assert.equal(result.isError, true)
  assert.equal(result.value.code, 'INVALID_JSON')
})

await checkAsync('empty stdout fails with EMPTY_OUTPUT', async () => {
  const { registered } = makeCtx({
    config: baseConfig,
    spawnImpl: () => fakeHandle({ stdout: '   \n' }),
  })
  const result = await registered.get('analyze_music_emotion').execute({ path: 'x.mp3' }, {})
  assert.equal(result.isError, true)
  assert.equal(result.value.code, 'EMPTY_OUTPUT')
})

await checkAsync('spawn throwing is normalized to SPAWN_FAILED', async () => {
  const { registered } = makeCtx({
    config: baseConfig,
    spawnImpl: () => {
      throw new Error('ENOENT python')
    },
  })
  const result = await registered.get('analyze_music_emotion').execute({ path: 'x.mp3' }, {})
  assert.equal(result.isError, true)
  assert.equal(result.value.code, 'SPAWN_FAILED')
})

await checkAsync('missing analyzer script fails before spawning', async () => {
  const emptyDir = mkdtempSync(path.join(tmpdir(), 'moodio-empty-'))
  let spawned = false
  const { registered } = makeCtx({
    config: { projectDir: emptyDir },
    spawnImpl: () => {
      spawned = true
      return fakeHandle()
    },
  })
  const result = await registered.get('analyze_music_emotion').execute({ path: 'x.mp3' }, {})
  assert.equal(result.isError, true)
  assert.equal(result.value.code, 'ANALYZER_NOT_FOUND')
  assert.equal(spawned, false)
})

await checkAsync('done rejecting is normalized to PROCESS_FAILED', async () => {
  const { registered } = makeCtx({
    config: baseConfig,
    spawnImpl: () => fakeHandle({ rejectDone: new Error('killed') }),
  })
  const result = await registered.get('analyze_music_emotion').execute({ path: 'x.mp3' }, {})
  assert.equal(result.isError, true)
  assert.equal(result.value.code, 'PROCESS_FAILED')
})

// ---------------------------------------------------------------- success
console.log('\n[t6] success path and summary shape')
await checkAsync('summarizes four axes, not the raw dump', async () => {
  const payload = {
    emotion: { emotion: '悲伤', emotion_id: 'sad', valence: -0.225, arousal: -0.114, desc: '低沉' },
    emotion_4d: {
      emotion: '悲壮',
      emotion_id: 'somber_majestic',
      desc: '宏大而不粗糙',
      directional: true,
      arousal_band: 'low',
      texture_band: 'low',
      grandeur_band: 'high',
      texture: 0.1026,
      grandeur: 0.722,
    },
    texture: { texture: 0.1026, parts: { a: 1 } },
    grandeur: { grandeur: 0.722, parts: { b: 2 } },
    arc: 'wave',
    quality: { confidence: 1.0 },
    clip: { duration_sec: 127.12 },
    features: { huge: 'blob' },
    segments: [{ start_sec: 0, end_sec: 10, valence: -0.1, arousal: -0.2, label: 'melancholic' }],
  }
  const { registered } = makeCtx({
    config: baseConfig,
    spawnImpl: () => fakeHandle({ stdout: JSON.stringify(payload) }),
  })
  const result = await registered.get('analyze_music_emotion').execute({ path: 'x.mp3' }, {})
  assert.equal(result.isError, false)
  assert.equal(result.value.emotion, '悲壮', 'four-axis word must win over the 2D word')
  assert.equal(result.value.axes.grandeur_band, 'high')
  // Real analyzer output carries the parts on the top-level texture/grandeur
  // blocks; summarizeAnalysis must fall back to them when emotion_4d omits them.
  assert.deepEqual(result.value.texture_parts, { a: 1 })
  assert.deepEqual(result.value.grandeur_parts, { b: 2 })
  assert.ok(!('features' in result.value), 'raw features must not leak by default')
  assert.ok(result.value.segments.omitted ?? Array.isArray(result.value.segments))
  assert.ok(Array.isArray(result.content) && result.content[0].type === 'text')
})

await checkAsync('surfaces grandeur_parts when the analyzer provides them', async () => {
  const payload = {
    emotion: { emotion: '悲壮', valence: -0.225, arousal: -0.114 },
    emotion_4d: {
      emotion: '悲壮',
      texture: 0.1026,
      grandeur: 0.722,
      texture_parts: { spectral_flatness: 0.03 },
      grandeur_parts: { full_loud_share: 0.52, dynamic_range_db: 0.95 },
    },
  }
  const { registered } = makeCtx({
    config: baseConfig,
    spawnImpl: () => fakeHandle({ stdout: JSON.stringify(payload) }),
  })
  const result = await registered.get('analyze_music_emotion').execute({ path: 'x.mp3' }, {})
  assert.equal(result.isError, false)
  assert.equal(result.value.grandeur_parts.full_loud_share, 0.52)
  assert.equal(result.value.texture_parts.spectral_flatness, 0.03)
})

await checkAsync('批量逐首检索后才执行分析，并固定检索过的曲目集合', async () => {
  const events = []
  const { registered } = makeCtx({
    config: baseConfig,
    listImpl: () => fakeHandle({ stdout: JSON.stringify({ pending: ['one.mp3', 'sub/two.flac'] }) }),
    onSpawn: (spec) => events.push(spec.argv.includes('--list-only') ? 'list' : 'analyze'),
    searchImpl: async (request) => { events.push(request.query); return { sources: [] } },
    spawnImpl: () => fakeHandle({ stdout: JSON.stringify({ ok: 2, failed: 0 }) }),
  })
  const result = await registered.get('batch_analyze_music').execute({ path: 'D:\\private\\music', limit: 2 }, {})
  assert.equal(result.isError, false)
  assert.equal(events.length, 6)
  assert.equal(events[0], 'list')
  assert.equal(events.at(-1), 'analyze')
  assert.ok(events.slice(1, -1).every((event) => !event.includes('private')))
  assert.equal(result.value.research_by_track.length, 2)
})

await checkAsync('批量联网失败时不分析任何曲目', async () => {
  let analyzed = false
  const { registered } = makeCtx({
    config: baseConfig,
    searchImpl: async () => { throw new Error('offline') },
    spawnImpl: () => { analyzed = true; return fakeHandle() },
  })
  const result = await registered.get('batch_analyze_music').execute({ path: 'D:\\music' }, {})
  assert.equal(result.isError, true)
  assert.equal(result.value.code, 'WEB_RESEARCH_FAILED')
  assert.equal(analyzed, false)
})

await checkAsync('batch strips the CLI agent-context tail', async () => {
  const payload = {
    scanned: 794, processed: 782, ok: 782, failed: 0,
    skills: [{ name: 'x' }], issue: 'report with ...',
  }
  const { registered } = makeCtx({
    config: baseConfig,
    spawnImpl: () => fakeHandle({ stdout: JSON.stringify(payload) }),
  })
  const result = await registered.get('batch_analyze_music').execute({ path: 'dir' }, {})
  assert.equal(result.isError, false)
  assert.equal(result.value.ok, 782)
  assert.ok(!('skills' in result.value), 'skills tail must be removed')
  assert.ok(!('issue' in result.value), 'issue tail must be removed')
})

await checkAsync('批量十二轴开关仅在显式请求时传入 CLI', async () => {
  let seen
  const { registered } = makeCtx({
    config: baseConfig,
    onSpawn: (spec) => { if (!spec.argv.includes('--list-only')) seen = spec },
    spawnImpl: () => fakeHandle({ stdout: JSON.stringify({ ok: 1, failed: 0 }) }),
  })
  await registered.get('batch_analyze_music').execute({ path: 'dir', include_axes_v2: true }, {})
  assert.ok(seen.argv.includes('--axes-v2'))
})

await checkAsync('batch always passes --quiet so stdout stays pure', async () => {
  let seen
  const { registered } = makeCtx({
    config: baseConfig,
    onSpawn: (spec) => {
      seen = spec
    },
    spawnImpl: () => fakeHandle({ stdout: JSON.stringify({ ok: 1, failed: 0 }) }),
  })
  await registered.get('batch_analyze_music').execute({ path: 'dir' }, {})
  assert.ok(seen.argv.includes('--quiet'))
})

await checkAsync('batch output defaults to the host cwd, including relative paths', async () => {
  let seen
  const { registered } = makeCtx({
    config: baseConfig,
    onSpawn: (spec) => { seen = spec },
    spawnImpl: () => fakeHandle({ stdout: JSON.stringify({ ok: 1, failed: 0 }) }),
  })
  const tool = registered.get('batch_analyze_music')
  await tool.execute({ path: 'dir' }, {})
  assert.equal(seen.argv[seen.argv.indexOf('--out') + 1], path.resolve(process.cwd(), 'moodio_batch.jsonl'))
  await tool.execute({ path: 'dir', out: 'results/library.jsonl' }, {})
  assert.equal(seen.argv[seen.argv.indexOf('--out') + 1], path.resolve(process.cwd(), 'results/library.jsonl'))
})

console.log('\n[t6b] render output carries the whole result')
await checkAsync('render emits full JSON, not a caption', async () => {
  const payload = { emotion: {}, emotion_4d: { emotion: '悲壮' }, arc: 'wave' }
  const { registered } = makeCtx({
    config: baseConfig,
    spawnImpl: () => fakeHandle({ stdout: JSON.stringify(payload) }),
  })
  const tool = registered.get('analyze_music_emotion')
  const result = await tool.execute({ path: 'x.mp3' }, {})
  const blocks = tool.output.render({ path: 'x.mp3' }, result.value)
  const text = blocks.map((b) => b.text).join('\n')
  // A caption-only render once hid a successful analysis; the JSON must survive.
  assert.ok(text.includes('"emotion": "悲壮"'), 'rendered text must contain the result values')
  assert.ok(text.length > 40, 'rendered text must not be a one-line caption')
})

// ---------------------------------------------------------------- portability
console.log('\n[t7] bundled analyzer resolution')
await checkAsync('no projectDir config falls back to the bundled python/ dir', async () => {
  // What makes the bundle installable anywhere: the analyzer ships in ./python
  // and is resolved from the module URL, so nothing needs path configuration.
  let seen
  const { registered } = makeCtx({
    config: {},
    onSpawn: (spec) => {
      seen = spec
    },
    spawnImpl: () => fakeHandle({ stdout: JSON.stringify({ emotion: {}, emotion_4d: {} }) }),
  })
  await registered.get('analyze_music_emotion').execute({ path: 'x.mp3' }, {})
  assert.equal(path.basename(seen.cwd), 'python', `cwd should be <pkg>/python, got ${seen.cwd}`)
  assert.ok(
    seen.argv.some((a) => typeof a === 'string' && a.endsWith('moodio.py')),
    'analyzer path must end with moodio.py',
  )
  assert.ok(
    !seen.cwd.includes('英雄主义'),
    'must not depend on the author machine path',
  )
})

await checkAsync('an explicit projectDir still wins', async () => {
  const custom = makeProject()
  let seen
  const { registered } = makeCtx({
    config: { projectDir: custom },
    onSpawn: (spec) => {
      seen = spec
    },
    spawnImpl: () => fakeHandle({ stdout: JSON.stringify({ emotion: {}, emotion_4d: {} }) }),
  })
  await registered.get('analyze_music_emotion').execute({ path: 'x.mp3' }, {})
  assert.equal(path.resolve(seen.cwd), path.resolve(custom))
})

check('no absolute author path is hard-coded in index.js', () => {
  const source = readFileSync(path.join(pkgRoot, 'index.js'), 'utf8')
  assert.ok(!source.includes('C:\\Users\\'), 'index.js must not embed a machine-specific path')
})

// ---------------------------------------------------------------- teardown
console.log('\n[t8] disposal')
check('effect returns a disposer that unregisters tools', () => {
  const { registered, disposers } = makeCtx({ config: baseConfig })
  assert.equal(registered.size, 3)
  assert.equal(disposers.length, 1)
  disposers[0]()
  assert.equal(registered.size, 0)
})

console.log('\n' + '='.repeat(62))
console.log(`结果：${passed}/${passed + failed} 项通过`)
console.log('='.repeat(62))
process.exit(failed === 0 ? 0 : 1)
