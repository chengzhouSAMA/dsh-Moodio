import test from 'node:test'
import assert from 'node:assert/strict'
import { combineMusicEvidence } from '../interpretation.js'

test('联合解释保留本地证据并附来源，不把检索摘要当作声学分数', () => {
  const data = { music_analysis: { status: 'evidence_ready', primary_acoustic_emotion: '悲壮', summary: '音频证据', constraints: [] } }
  const result = combineMusicEvidence(data, { song: 'song', background: { sources: [{ url: 'https://artist.example/song', preview: 'Official background.' }] }, lyrics: { sources: [] } })
  assert.equal(result.primary_acoustic_emotion, '悲壮')
  assert.equal(result.context.status, 'sources_available')
  assert.equal(result.context.background_sources.length, 1)
  assert.equal(data.music_analysis.context, undefined)
  assert.ok(result.analysis_requirements.some((text) => text.includes('冲突')))
})

test('联网失败保留音频判断并显式标为部分证据', () => {
  const result = combineMusicEvidence({ music_analysis: { status: 'evidence_ready', primary_acoustic_emotion: '平静' } }, { status: 'failed', error: { code: 'WEB_RESEARCH_FAILED' } })
  assert.equal(result.status, 'partial')
  assert.equal(result.primary_acoustic_emotion, '平静')
  assert.equal(result.context.status, 'unavailable')
  assert.deepEqual(result.context.background_sources, [])
})
