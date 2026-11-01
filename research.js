import path from 'node:path'

const BLOCKED_SOURCE = /(?:^|\.)(?:reddit\.com|rateyourmusic\.com|albumoftheyear\.org|musicboard\.app|trustpilot\.com)$/i
const BLOCKED_TEXT = /(?:\breviews?\b|\bcomments?\b|\bcommentary\b|\breactions?\b|\bforum\b|\bratings?\b|评论|乐评|测评|评价)/i
const LYRICS_MARKER = /(?:\blyrics?\b|歌词|歌詞|\blrc\b)/i
const BACKGROUND_MARKER = /(?:\bofficial\b|\breleases?\b|\breveals?\b|\banthem\b|\bmaking\b|\bproduction\b|\bproduced\b|\binterview\b|\bproject\b|\bcreation\b|\bcreated\b|\bchampions\b|创作|制作|发布|幕后|テーマソング|主题曲|片头曲|片尾曲)/i
const KARAOKE_MARKER = /(?:\bkaraoke\b|\bktv\b|노래방|カラオケ)/i

// Version / arrangement tags that describe a file, not a song. They must never
// reach the query or the title match: a source page called "残酷天使的行动纲领"
// can never contain "– 新世纪福音战士 OP1 钢琴版".
const SEPARATOR = /\s*(?:[–—−‒]|\s-\s|[|｜/／×]|[：:])\s*/u
const TRAILING_TAG = /(?:\s*[（(\[【][^）)\]】]{0,40}[）)\]】]|\s*(?:钢琴版|钢琴|伴奏|纯音乐|翻奏|改编|翻唱|完整版|高音质|无损|音频|动态歌词|卡拉ok|demo|remix|cover|piano|instrumental|karaoke|mv|op\s*\d+|ed\s*\d+|tv\s*size|full\s*size|version))$/iu
const CJK = /[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]/u

function normalizeWords(text) {
  return String(text).normalize('NFKC').toLowerCase().replace(/[^\p{L}\p{N}]+/gu, ' ').trim()
}

function songName(audioPath) {
  // 路径只用于提取文件名；任何私人目录都不发送给搜索服务。
  const base = path.win32.basename(path.posix.basename(String(audioPath)))
  return base.replace(/\.[^.]+$/, '').replace(/(?:[. _-]+(?:音频|audio))$/iu, '')
    .replace(/[_]+/g, ' ').replace(/\s+/g, ' ').trim()
}

/** Main title only: drop "– 新世纪福音战士 OP1 钢琴版" and similar file suffixes. */
export function coreTitle(title) {
  const head = String(title ?? '').split(SEPARATOR)[0] || ''
  let value = (head || String(title ?? '')).trim()
  for (let i = 0; i < 4; i += 1) {
    const next = value.replace(TRAILING_TAG, '').trim()
    if (!next || next === value) break
    value = next
  }
  return value || String(title ?? '').trim()
}

function songIdentity(song) {
  const bracket = song.match(/^(?:【([^】]+)】|\[([^\]]+)\])\s*(.+)$/u)
  const artist = bracket ? bracket[1] ?? bracket[2] : song.includes(' - ') ? song.split(' - ').slice(0, -1).join(' - ') : ''
  const named = bracket ? bracket[3] : song.includes(' - ') ? song.split(' - ').at(-1) : song
  const title = coreTitle(named)
  return { title: title || named, artist_hint: artist, source: 'filename_hint' }
}

/**
 * How much of `title` appears in the source locator, 0..1.
 * Exact containment is 1; otherwise CJK uses character bigrams and Latin uses
 * word tokens, so "残酷天使的行动纲领" still matches a page that only carries
 * that name without the arrangement subtitle.
 */
export function titleCoverage(locator, title) {
  const hay = normalizeWords(locator)
  const needle = normalizeWords(title)
  if (!hay || !needle) return 0
  if (hay.includes(needle)) return 1
  if (CJK.test(needle)) {
    const chars = needle.replace(/\s+/g, '')
    if (chars.length < 2) return hay.includes(chars) ? 1 : 0
    const grams = []
    for (let i = 0; i < chars.length - 1; i += 1) grams.push(chars.slice(i, i + 2))
    const hits = grams.filter((gram) => hay.includes(gram)).length
    return hits / grams.length
  }
  const tokens = needle.split(' ').filter(Boolean)
  if (!tokens.length) return 0
  return tokens.filter((token) => hay.includes(token)).length / tokens.length
}

const MIN_TITLE_COVERAGE = 0.6

function usableSource(source, kind, title) {
  if (!source || typeof source.url !== 'string') return false
  try {
    const parsed = new URL(source.url)
    if (!['https:', 'http:'].includes(parsed.protocol)) return false
    if (BLOCKED_SOURCE.test(parsed.hostname)) return false
    const locator = `${parsed.hostname} ${decodeURIComponent(parsed.pathname)} ${parsed.search} ${parsed.hash} ${source.title ?? ''}`
    if (BLOCKED_TEXT.test(`${locator} ${source.snippet ?? ''}`)) return false
    const coverage = titleCoverage(locator, title)
    if (coverage < MIN_TITLE_COVERAGE) return false
    if (kind === 'background') {
      if (KARAOKE_MARKER.test(locator)) return false
      // A page named exactly after the song is background material even when the
      // provider returns no snippet to scan; a partial match still needs a marker.
      if (coverage < 0.9 && !BACKGROUND_MARKER.test(`${locator} ${source.snippet ?? ''}`)) return false
    }
    if (kind === 'lyrics' && !LYRICS_MARKER.test(locator)) return false
    return true
  } catch {
    return false
  }
}

function selectedSources(response, kind, title) {
  if (!response || !Array.isArray(response.sources)) throw new Error('web search returned no source list')
  const kept = []
  let rejected = 0
  for (const source of response.sources) {
    if (usableSource(source, kind, title)) kept.push(source)
    else rejected += 1
  }
  return {
    sources: kept.slice(0, 4).map((source) => ({
      url: source.url,
      title: typeof source.title === 'string' ? source.title.slice(0, 150) : '',
      ...(kind === 'background' && source.snippet
        ? { preview: String(source.snippet).replace(/\s+/g, ' ').slice(0, 180) } : {}),
    })),
    returned: response.sources.length,
    rejected,
  }
}

export async function researchSong(web, audioPath, signal) {
  if (typeof web?.search !== 'function') throw new Error('web.search is required for background and lyric lookup')
  const song = songName(audioPath)
  if (!song) throw new Error('cannot search without a song filename')
  const identity = songIdentity(song)
  const subject = [identity.artist_hint, identity.title].filter(Boolean).join(' ')
  const backgroundQuery = `${subject} official song creation background artist interview`
  const lyricsQuery = `${identity.title} lyrics official song text`
  const [background, lyrics] = await Promise.all([
    web.search({ query: backgroundQuery, maxResults: 8 }, signal),
    web.search({ query: lyricsQuery, maxResults: 8 }, signal),
  ])
  const title = identity.title
  const backgroundPick = selectedSources(background, 'background', title)
  const lyricPick = selectedSources(lyrics, 'lyrics', title)
  return {
    song,
    identity,
    searched_at: new Date().toISOString(),
    queries: { background: backgroundQuery, lyrics: lyricsQuery },
    background: {
      found: backgroundPick.sources.length > 0,
      sources: backgroundPick.sources,
      returned: backgroundPick.returned,
      rejected: backgroundPick.rejected,
    },
    lyrics: {
      found: lyricPick.sources.length > 0,
      sources: lyricPick.sources,
      returned: lyricPick.returned,
      rejected: lyricPick.rejected,
    },
    note: 'Background previews are unverified source excerpts; lyric links are not audio transcription. Reviews/comments are excluded. Acoustic scores never use online text.',
  }
}
