# dsh-Moodio

[![CI](https://github.com/chengzhouSAMA/dsh-Moodio/actions/workflows/ci.yml/badge.svg)](https://github.com/chengzhouSAMA/dsh-Moodio/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-4d6bfe.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-4d6bfe.svg)](python/pyproject.toml)

给 agent 用的音乐情绪分析：量出来，不猜。

默认分析整首的乐器候选、旋律和十八轴，再结合创作背景与歌词来源，供 agent 综合解释音乐情感。
声学计算不需要权重；乐器识别使用一次下载、本地运行的 CLAP + AST 模型，无需音频上传或模型 API key。

[English](README.md) · [算法说明](docs/algorithm.md) · [更新日志](CHANGELOG.md) · [发布页](https://github.com/chengzhouSAMA/dsh-Moodio/releases)

## 输出什么

`music_analysis` 汇总乐器、旋律、十八轴、背景/歌词来源及局限；综合情感结论由 agent
依据这些证据形成，四轴词是声学起点，不是不可修正的最终结论。

### 乐器识别

目标目录共 21 类：古筝、二胡、琵琶、笛子、葫芦丝、唢呐、扬琴、笙、陶笛、古琴；
钢琴、吉他、小提琴、萨克斯、架子鼓、大提琴、长笛、贝斯、尤克里里、口琴、手风琴。

每十秒分析一个窗口，连续覆盖整首并保留末段。CLAP 给出候选相似度，AST 为已有监督
类别提供交叉证据；次高相似度不直接当作共同存在的乐器。中国乐器中不少类别仅有
家族或零样本支持，`verification_scope` 会明确标记。未知声音、相近音色和模型分歧
不会伪装为确定结果。**目录覆盖不等于 21 类已经逐类验证准确率**，也不保证辨认所有合奏声部。

安装一次约 1 GB 的模型（CLAP: Apache-2.0；AST: BSD-3-Clause）：

```bash
python -m pip install -e "./python[instruments]"
python python/moodio.py instruments setup
python python/moodio.py instruments status
```

模型下载固定版本并核对 SHA-256；可显式传 `--endpoint https://hf-mirror.com` 使用镜像。
默认缓存位于 `~/.cache/moodio/clap-htsat-unfused`，可用 `MOODIO_INSTRUMENT_MODEL_DIR`
或 `--instrument-model-dir` 指向已有模型。正式分析只读本地权重，不自动下载。
缺模型时返回 `instruments.status: unavailable`，其余分析继续。需要跳过时使用
`--no-instruments` 或 `include_instruments: false`。

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

**十八轴诊断**（默认开启；`--no-axes-v3` 可跳过）保留上述十二轴，增加低频存在、噪声感、瞬态锐度、
速度稳定、调性稳定和铺陈方向。每轴附带 `status`；证据不足时 `value: null`，而不是
虚构一个低分。铺陈方向中 0 为减弱、0.5 为大体持平、1 为增强。原有 `HarmonicComplexity`
实际计算的是**音级分布熵**，并不等于和弦复杂度；十八轴输出会显式标记其 `measure`。
这些扩展数值仍是规则先验，未经过人工标注曲库的准确率验证，不参与四轴情绪判定。

明亮度的两个分量使用连续映射，低于经验参考下限时不再直接截成零。
`Brightness.evidence` 返回谱质心、3 kHz 以上能量占比及各分量；`normalization`
标记算法。整曲和分段亮度一致使用该映射，历史亮度及相关起伏分数需重新计算后比较。
这里度量的是声学频谱亮度，不等同于旋律或情绪上的“明亮”。

**旋律候选**（CLI 和插件默认开启，默认分析整首）使用单声部 `pYIN` 从已分析的波形估计
`melody.notes`（音高 MIDI、起止秒数、有声概率），再组织 `melody.phrases`（走势）、
`melody.motifs`（允许整体移调的重复音程序列）和带时间证据的
`melody.agent_summary`。证据不足时 `status: insufficient_evidence`，不推断乐句。
命中音高搜索边界的片段会扩大音域复核，剔除不一致读数；`quality.boundary_validation`
返回复核前后数量。摘要按整首时长报告音符覆盖率及分段证据；重复检测使用跨分句滑窗，
同时比较音程和相对节奏。`motifs` 为空时由 `motif_search` 解释，不能据此断言原曲没有重复。
音源未分离：复杂混音的候选音符可能来自伴奏或贝斯，`candidate` **不代表**
已确认主旋律；重复动机也不等于副歌。需要跳过旋律时传 `--no-melody` 或
`include_melody: false`。仅在明确需要预览时设置 `max_seconds`；不设置则读取整首。

## 快速开始

```bash
git clone https://github.com/chengzhouSAMA/dsh-Moodio.git
cd dsh-Moodio
pip install -r python/requirements.txt

python python/moodio.py analyze --path song.mp3 --human
python python/moodio.py analyze --path song.mp3 --axes-v2 --fields emotion,emotion_4d,axes_v2
python python/moodio.py analyze --path song.mp3 --axes-v3 --fields emotion,emotion_4d,axes_v3
python python/moodio.py analyze --path song.mp3 --melody --fields clip,melody,warnings
python python/moodio.py batch  --path ./music --out mood.jsonl --workers 1
python python/moodio.py report --path mood.jsonl
```

需要 Python 3.9+。`batch` 每首写一行 JSON 并追加，重跑即续跑：完成的跳过，失败的
重试，stdout 始终可解析。已有十二轴批量结果要补跑十八轴时，请使用**新的** `--out` 路径；
断点续跑不会修改已完成记录；旧批量结果若需补充乐器、旋律或更新亮度，也请换新的 `--out`。

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

插件先完成本地声学分析（或歌词转写），再联网检索创作背景和歌词，返回来源链接，
排除评论与乐评。只发送清理后的文件名，不发送本地目录。

**检索是硬性环节，不是可选开关。** 没有配置项能关掉它，也没有离线降级：拿不到
`web.search`，或者查询抛错，整次调用直接返回 `WEB_RESEARCH_FAILED`，不会给出没有来源
的分析结果。独立的 Python CLI 仍然完全离线。

检索前会先把文件名还原成主标题——`【Animenz】残酷天使的行动纲领 – 新世纪福音战士
OP1 钢琴版` 变成 `残酷天使的行动纲领`——并按标题覆盖率匹配来源，而不是要求整串包含，
所以只写着歌名的页面不会被「缺了改编副标题」判掉。每次返回都会带上实际发出的
`queries`，以及各通道的 `returned` / `rejected`，用来区分「没有任何结果」和「都被过滤了」。

```
analyze_music_emotion(path: "/music/track.flac", include_axes_v2: true)
analyze_music_emotion(path: "/music/track.flac", include_axes_v3: true)
analyze_music_emotion(path: "/music/track.flac", include_melody: true)
batch_analyze_music(path: "/music", workers: 1, limit: 50)
transcribe_music_lyrics(path: "/music/track.flac", model_dir: "/models/faster-whisper")
```

批量完成声学分析后会为每首待处理曲目返回 `research_by_track`。单次上限 50 首；更大的
集合传 `limit: 50` 分批续跑。任何一首检索失败，整批返回 `WEB_RESEARCH_FAILED`，
已完成的声学结果放在 `value.partial` 里保留。若部分音频本身分析失败，批量结果仍带
失败数并标为错误。

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
占唤醒 0.30 的权重，错一个八度结果就偏。`TempoStability` 接近 0 就是征兆：估计器正在
两个八度之间来回跳，这时候速度值本身不该被信。

削波的母带在分析前就已经丢了动态。峰值 `>= 0.995` 会置 `clipping_suspected`，规模这类
动态指标会一并受害。

视频提取的音轨和分离出来的分轨常常被削掉高频。3 kHz 以上几乎没有能量时，明亮会掉到接近
0，瞬态锐度也跟着塌——一首钢琴曲可能比管弦乐还「暗」。看
`axes_v3.axes.Brightness.evidence.high_frequency_energy_share`：真实原声乐器一般在
1–5%，被低通过的文件只有 0.2% 左右。怀疑编曲之前，先换一版无损源对照。

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
                      十八轴共用特征提取，启用 --axes-v3 后追加六轴
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
