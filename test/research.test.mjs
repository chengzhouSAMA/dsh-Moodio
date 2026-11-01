import test from 'node:test'
import assert from 'node:assert/strict'
import { researchSong, coreTitle, titleCoverage } from '../research.js'

test('每次分别检索创作背景和歌词，不发送本地目录或搜索评论', async () => {
  const requests = []
  const web = {
    async search(request) {
      requests.push(request)
      if (request.query.toLowerCase().includes('lyrics')) {
        return { sources: [
          { url: 'https://music.example/lyrics', title: 'Die For You Lyrics', snippet: 'copyrighted lyrics go here' },
          { url: 'https://news.example/song', title: 'Die For You song unveiled' },
          { url: 'https://reddit.com/r/music/comments/123', title: 'Song comments' },
        ] }
      }
      return { sources: [
        { url: 'https://playvalorant.com/news/die-for-you', title: 'Die For You: VALORANT Champions song', snippet: 'Created for the tournament.' },
        { url: 'https://noraebang.app/songs/die-for-you', title: 'Die For You karaoke 노래방' },
        { url: 'https://music.example/die-for-you#comments', title: 'Making of Die For You' },
        { url: 'https://music.example/artist', title: 'Artist biography' },
        { url: 'https://music.example/feature', title: 'Making of the song', snippet: 'Read fan comments about the song.' },
        { url: 'https://reviews.example/review', title: 'Music review' },
      ] }
    },
  }
  const track = String.raw`C:\Users\Someone\Private\VALORANT_Grabbitz - Die For You.mp3`
  const result = await researchSong(web, track)
  assert.equal(requests.length, 2)
  assert.ok(requests.some((request) => /background|creation|making.of|创作/i.test(request.query)))
  assert.ok(requests.some((request) => /lyrics/i.test(request.query)))
  assert.ok(requests.every(({ query }) => !query.includes('Private') && !query.includes('C:') && !/comment|review|reddit|评论|乐评/i.test(query)))
  assert.equal(result.background.sources.length, 1)
  assert.equal(result.lyrics.sources.length, 1)
  assert.ok(!JSON.stringify(result.lyrics).includes('copyrighted lyrics'))
})

test('检索函数对缺失联网能力或搜索失败显式报错', async () => {
  await assert.rejects(researchSong({}, 'song.mp3'), /联网|web|search/i)
  await assert.rejects(researchSong({ search: async () => { throw new Error('offline') } }, 'song.mp3'), /offline/)
})

test('每次调用都重新检索而不是复用旧结果', async () => {
  let count = 0
  const web = { search: async () => { count += 1; return { sources: [] } } }
  await researchSong(web, 'song.mp3')
  await researchSong(web, 'song.mp3')
  assert.equal(count, 4)
})

test('从下载文件名提取曲名，避免版本标签阻断来源匹配', async () => {
  const queries = []
  const web = { search: async ({ query }) => {
    queries.push(query)
    return { sources: [{ url: 'https://music.example/official', title: '鸟之诗 official piano arrangement', snippet: 'Official release.' }] }
  } }
  const result = await researchSong(web, 'C:/private/【Animenz】鸟之诗 (Project CHE Special).音频.mp3')
  assert.equal(result.identity.title, '鸟之诗')
  assert.equal(result.identity.artist_hint, 'Animenz')
  assert.ok(queries.every((query) => !query.includes('音频') && !query.includes('private')))
  assert.equal(result.background.sources.length, 1)
})

test('把副标题与版本标签从标题剥掉，来源只按主标题匹配', async () => {
  const queries = []
  const web = { search: async ({ query }) => {
    queries.push(query)
    if (/lyrics/i.test(query)) return { sources: [] }
    return { sources: [
      { url: 'https://zh.wikipedia.org/wiki/残酷天使的行动纲领', title: '残酷天使的行动纲领 - 维基百科', snippet: '《新世纪福音战士》主题曲，佐藤英敏作曲、及川眠子作词。' },
      { url: 'https://www.tan8.com/yuepu-69868.html', title: '残酷天使的行动纲领 karaoke カラオケ', snippet: '伴奏下载' },
    ] }
  } }
  const result = await researchSong(web, 'C:/x/【Animenz】残酷天使的行动纲领 – 新世纪福音战士 OP1 钢琴版.音频.mp3')
  assert.equal(result.identity.title, '残酷天使的行动纲领')
  assert.ok(queries.every((query) => !query.includes('钢琴版') && !query.includes('OP1')))
  assert.equal(result.background.returned, 2)
  assert.ok(result.background.rejected >= 1, '版本标签页不应全部命中')
  assert.ok(result.background.sources.length >= 1, '维基百科条目应该能匹配上')
  assert.ok(result.background.sources.some((source) => source.url.includes('wikipedia')))
})

test('标题匹配用覆盖率而不是整串包含', () => {
  assert.equal(coreTitle('残酷天使的行动纲领 – 新世纪福音战士 OP1 钢琴版'), '残酷天使的行动纲领')
  assert.equal(coreTitle('鸟之诗 (Project CHE Special)'), '鸟之诗')
  assert.ok(titleCoverage('zh.wikipedia.org 残酷天使的行动纲领 维基百科', '残酷天使的行动纲领') >= 0.6)
  assert.ok(titleCoverage('https://x/song-title', 'song title') >= 0.6)
  assert.ok(titleCoverage('https://x/unrelated', 'song title') < 0.6)
})
