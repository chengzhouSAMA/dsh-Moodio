# dsh-Moodio

[![CI](https://github.com/chengzhouSAMA/dsh-Moodio/actions/workflows/ci.yml/badge.svg)](https://github.com/chengzhouSAMA/dsh-Moodio/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-4d6bfe.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-4d6bfe.svg)](python/pyproject.toml)

给 agent 用的音乐情绪分析：量出来，不猜。

四根轴——效价、唤醒、织体、规模——外加可选的十二轴声学诊断，全部在本地从波形算出。
不训练、不要 API key、不下载模型权重。附一个 DSH Host 插件，agent 可以直接当工具调用。

[English](README.md) · [算法说明](docs/algorithm.md) · [更新日志](CHANGELOG.md) · [发布页](https://github.com/chengzhouSAMA/dsh-Moodio/releases)

## 输出什么

| 轴 | 范围 | 主要由什么驱动 |
|---|---|---|
| `valence` | -1..1 | 调式得分、和声复杂度、亮度 |
| `arousal` | -1..1 | 速度、起音密度、响度、谱通量 |
| `texture` | 0..1 | 失真、打击感、噪声底、亮度 |
| `grandeur` | 0..1 | 动态范围、持续强奏、频谱对比度 |

`texture` 是攻击性，`grandeur` 是规模。两者刻意分开：管弦乐 grandeur 很高、
texture 几乎为零，如果只量攻击性，它会被读成「安静」。

四轴最后落成一个词。下面两首的效价和唤醒几乎一样，只有 grandeur 分得开——
这一维就是为这个加的：

| 曲目 | val | aro | tex | gran | 判定 |
|---|---|---|---|---|---|
| 在虚无中永存 - 英雄主义 | -0.23 | -0.11 | 0.10 | 0.72 | 悲壮 |
| 邓垚 - 诀别书 | -0.16 | -0.13 | 0.15 | 0.51 | 悲伤 |
| Two Steps From Hell - Victory | +0.28 | +0.04 | 0.33 | 0.69 | 壮丽 |
| Coldplay - Yellow | +0.21 | -0.24 | 0.36 | 0.55 | 平静 |

**十二轴诊断**（可选，`--axes-v2`）另外给出效价、能量、明亮、温润、刺耳、速度、
节奏规整、打击感、音符密度、规模、起伏、和声复杂度，**分析整段音频**。它们是独立
归一化的 0..1 声学量，和四轴数值不可互换；速度、密度、刺耳仍保留原始单位。

## 快速开始

```bash
git clone https://github.com/chengzhouSAMA/dsh-Moodio.git
cd dsh-Moodio
pip install -r python/requirements.txt

python python/moodio.py analyze --path song.mp3 --human
python python/moodio.py analyze --path song.mp3 --axes-v2 --fields emotion,emotion_4d,axes_v2
python python/moodio.py batch  --path ./music --axes-v2 --out mood.jsonl --workers 6
python python/moodio.py report --path mood.jsonl
```

需要 Python 3.9+。`batch` 每首写一行 JSON 并追加，重跑即续跑：完成的跳过，失败的
重试，stdout 始终可解析。

## 100 首实测

从 794 首曲库等距抽 100 首，整首分析，本地、确定性。三件事：

- **62 / 100** 的唤醒落在 ±0.15 内——两种情绪，一个速度。
- **和声复杂度 0.95 ± 0.02**——一根几乎不动的轴。
- **能量与规模相关 −0.39**——越响的歌，测出来的动态越小。

[![情绪地图](https://github.com/chengzhouSAMA/dsh-Moodio/releases/download/v2.2.1/01-emotion-map.png)](https://github.com/chengzhouSAMA/dsh-Moodio/releases/tag/v2.2.1)

其余图表——十二轴统计表、100 张指纹、读后记——都挂在
[v2.2.1 发布页](https://github.com/chengzhouSAMA/dsh-Moodio/releases/tag/v2.2.1)。

## 作为 DSH 插件

```
plugin_manager(action: "install_bundle", target: "<dsh-Moodio 的绝对路径>")
```

三个工具：`analyze_music_emotion`、`batch_analyze_music`、`transcribe_music_lyrics`。
分析器就在包内 `python/`，相对 `index.js` 解析，不需要配置路径。

每次工具调用都会先搜该曲的创作背景和歌词，返回来源链接，排除评论与乐评。只发送
清理后的文件名，不发送本地目录。**搜索失败时返回 `WEB_RESEARCH_FAILED`，不会瞎猜**；
独立的 Python CLI 则完全离线。

```
analyze_music_emotion(path: "/music/track.flac", include_axes_v2: true)
batch_analyze_music(path: "/music", workers: 6, limit: 50)
transcribe_music_lyrics(path: "/music/track.flac", model_dir: "/models/faster-whisper")
```

批量调用会为每首待处理曲目返回 `research_by_track`，单次上限 50 首；更大的集合传
`limit: 50` 分批续跑。

## 歌词转写（可选）

需要**本地** CTranslate2 Whisper 模型目录（含 `model.bin`），不会自动下载权重。

```
python -m pip install faster-whisper
python python/moodio.py lyrics --path song.mp3 --model-dir /models/faster-whisper --language zh
```

也可以设 `MOODIO_LYRICS_MODEL_DIR` 代替 `--model-dir`。转写是对混合录音的猜测，不是
核对过的歌词：人声被编曲埋住时可能漏词或编词。它**不改变**四轴情绪分。

## 已知问题

这是规则先验，不是训练出来的模型。权重来自音乐心理学已发表的**方向性**结论，不是拟合
听众评分。请在同一批结果内部排序，不要把数值当成人类评分的总体均值。

歌词不进入情绪模型。悲伤的词配明亮的伴奏，仍然会被判成明亮。

两种标定模式不可混用。`fixed`（默认）把零点放在「中性证据」，不同曲库之间可比；
`corpus-relative` 按当前曲库中位数重新居中，偏斜曲库也能保住四个象限，但跨库不可比。

`confidence` 是输入质量，不是准确率。短、轻、无调性或非周期性的音频分数低。

速度有八度问题：自相关常在 `t` 与 `2t` 同时出峰，读到 95 BPM 可能其实是 184。速度
占唤醒 0.30 的权重，错一个八度结果就偏。

削波的母带在分析前就已经丢了动态。峰值 `>= 0.995` 会置 `clipping_suspected`，规模这类
动态指标会一并受害。

`--max-seconds` 截断可能删掉一首歌的性格：安静的前奏会藏住 grandeur。管弦乐请分析整首。

把 523 首里的 264 首读成「忧郁」的那个标定 bug，连同其他几个，写在
[docs/algorithm.md](docs/algorithm.md)。

## 目录

```
index.js              DSH Host 插件
cordis.patch.yml      bundle patch
locale/{en,zh}.json   插件元数据
python/
  moodio.py           CLI
  music_emotion/      分析器包
    vamodel.py        由声学线索算效价/唤醒
    texture.py        织体与规模
    four_axis.py      四轴落到一个情绪词
    axes_v2.py        可选十二轴诊断
    features.py       librosa 特征，每首只算一次
    pipeline.py       编排
    batch.py          可续跑的目录批处理
    report.py         Markdown / JSON 汇总
    lyrics.py         可选本地歌词转写
    neural.py         可选 wav2small 旁路
  agent/              给 agent 的使用契约
  tests/              自带刺激信号的测试
test/offline.test.mjs 插件契约测试，不需要 DSH 运行时
```

## 测试

```bash
npm run test:fast      # 插件契约 + 快速 Python 套件
npm run test:python    # 全部 Python 套件与方向性检查
npm run test:all       # 全部
```

Python 套件自己合成刺激信号或用固定特征探针，都不需要音频文件。CI 在每次 push 时跑
两半：Python 3.9 在 `ubuntu-22.04`（26.04 还没有 3.9 构建）、3.12 在 `ubuntu-latest`，
Node 契约测试在 Node 24。

## 许可

MIT。可选神经旁路基于 `audeering/wav2small`，其权重为 CC-BY-NC-SA-4.0，不随仓库分发。
见 [LICENSE](LICENSE)。
