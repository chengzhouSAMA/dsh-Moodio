import path from 'node:path'

const BLOCKED_SOURCE = /(?:^|\.)(?:reddit\.com|rateyourmusic\.com|albumoftheyear\.org|musicboard\.app|trustpilot\.com)$/i
const BLOCKED_TEXT = /(?:\breviews?\b|\bcomments?\b|\bcommentary\b|\breactions?\b|\bforum\b|\bratings?\b|评论|乐评|测评|评价)/i
const LYRICS_MARKER = /(?:\blyrics?\b|歌词|歌詞|\blrc\b)/i
const BACKGROUND_MARKER = /(?:\bofficial\b|\breleases?\b|\breveals?\b|\banthem\b|\bmaking\b|\bproduction\b|\bproduced\b|\binterview\b|\bproject\b|\bcreation\b|\bcreated\b|\bchampions\b|创作|制作|发布|幕后|テーマソング)/i
const KARAOKE_MARKER = /(?:\bkaraoke\b|\bktv\b|노래방|カラオケ)/i

function normalizeWords(text) {
  return String(text).normalize('NFKC').toLowerCase().replace(/[^\p{L}\p{N}]+/gu, ' ').trim()
}

function songName(audioPath) {
  // 路径只用于提取文件名；任何私人目录都不发送给搜索服务。
  const base = path.win32.basename(path.posix.basename(String(audioPath)))
  return base.replace(/\.[^.]+$/, '').replace(/[_]+/g, ' ').replace(/\s+/g, ' ').trim()
}

function usableSource(source, kind, title) {
  if (!source || typeof source.url !== 'string') return false
  try {
    const parsed = new URL(source.url)
    if (!['https:', 'http:'].includes(parsed.protocol)) return false
    if (BLOCKED_SOURCE.test(parsed.hostname)) return false
    const locator = `${parsed.hostname} ${decodeURIComponent(parsed.pathname)} ${parsed.search} ${parsed.hash} ${source.title ?? ''}`
    if (BLOCKED_TEXT.test(`${locator} ${source.snippet ?? ''}`)) return false
    if (!normalizeWords(locator).includes(normalizeWords(title))) return false
    if (kind === 'background' &&
        (KARAOKE_MARKER.test(locator) || !BACKGROUND_MARKER.test(`${locator} ${source.snippet ?? ''}`))) return false
    if (kind === 'lyrics' && !LYRICS_MARKER.test(locator)) return false
    return true
  } catch {
    return false
  }
}

function selectedSources(response, kind, title) {
  if (!response || !Array.isArray(response.sources)) throw new Error('web search returned no source list')
  return response.sources.filter((source) => usableSource(source, kind, title)).slice(0, 4).map((source) => ({
    url: source.url,
    title: typeof source.title === 'string' ? source.title.slice(0, 150) : '',
    ...(kind === 'background' && source.snippet
      ? { preview: String(source.snippet).replace(/\s+/g, ' ').slice(0, 180) } : {}),
  }))
}

export async function researchSong(web, audioPath, signal) {
  if (typeof web?.search !== 'function') throw new Error('web.search is required before music analysis')
  const song = songName(audioPath)
  if (!song) throw new Error('cannot search without a song filename')
  const [background, lyrics] = await Promise.all([
    web.search({ query: `${song} official song creation background artist interview`, maxResults: 8 }, signal),
    web.search({ query: `${song} lyrics official song text`, maxResults: 8 }, signal),
  ])
  const title = song.includes(' - ') ? song.split(' - ').at(-1) : song
  const backgroundSources = selectedSources(background, 'background', title)
  const lyricSources = selectedSources(lyrics, 'lyrics', title)
  return {
    song,
    searched_at: new Date().toISOString(),
    background: { found: backgroundSources.length > 0, sources: backgroundSources },
    lyrics: { found: lyricSources.length > 0, sources: lyricSources },
    note: 'Background previews are unverified source excerpts; lyric links are not audio transcription. Reviews/comments are excluded. Acoustic scores never use online text.',
  }
}
