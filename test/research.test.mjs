import test from 'node:test'
import assert from 'node:assert/strict'
import { researchSong } from '../research.js'

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

test('缺失联网能力或搜索失败时拒绝继续', async () => {
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
