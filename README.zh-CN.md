# dsh-Moodio

本地音乐情绪分析。输入音频，输出四个数值，再由这四个数给出一个情绪词。

两部分：Python 分析器（`python/moodio.py`），以及一个把分析器暴露成两个工具的
DSH host 插件。默认路径不联网、不要 API key、不用下模型权重。

[English](README.md) | [算法说明](docs/algorithm.md) | [更新日志](CHANGELOG.md)

## 输出

四个轴，全部由 librosa 特征算出来：

| 轴 | 范围 | 主要由什么驱动 |
|---|---|---|
| `valence` | -1..1 | 调式得分、和声复杂度、亮度 |
| `arousal` | -1..1 | 速度、起音密度、响度、谱通量 |
| `texture` | 0..1 | 失真、打击感、噪声底、亮度 |
| `grandeur` | 0..1 | 动态范围、持续强奏、频谱对比度 |

`texture` 是攻击性，`grandeur` 是规模。两者刻意分开：管弦乐 grandeur 很高、
texture 几乎为零，如果只量攻击性，它会被读成"安静"。

测试语料上的抽样：

| 曲目 | val | aro | tex | gran | 判定 |
|---|---|---|---|---|---|
| 在虚无中永存 - 英雄主义 | -0.23 | -0.11 | 0.10 | 0.72 | 悲壮 |
| 邓垚 - 诀别书 | -0.16 | -0.13 | 0.15 | 0.51 | 悲伤 |
| Two Steps From Hell - Victory | +0.28 | +0.04 | 0.33 | 0.69 | 壮丽 |
| Coldplay - Yellow | +0.21 | -0.24 | 0.36 | 0.55 | 平静 |
| Bruno Mars - Runaway Baby | +0.17 | +0.47 | 0.57 | 0.48 | 热烈 |

前两首的效价和唤醒度几乎一样，只有 grandeur 分得开。这一维就是为这个加的。

## 安装

```
git clone https://github.com/chengzhouSAMA/dsh-Moodio.git
cd dsh-Moodio
pip install -r python/requirements.txt
```

需要 Python 3.9+。插件部分额外需要 Node 18+。

作为 DSH bundle 安装（注册 `analyze_music_emotion` 和 `batch_analyze_music`）：

```
plugin_manager(action: "install_bundle", target: "<dsh-Moodio 的绝对路径>")
```

分析器就在包里 `python/` 下，按 `index.js` 的位置解析，不需要配路径。只有想让它
跑另一份 checkout 时才需要在插件配置里传 `projectDir`。

## 用法

单曲：

```
cd python
python moodio.py analyze --path "song.flac" --segment-sec 10 --hop-sec 10
```

```
{
  "emotion": { "emotion": "悲壮", "valence": -0.225, "arousal": -0.114 },
  "emotion_4d": {
    "emotion": "悲壮",
    "arousal_band": "low", "texture_band": "low", "grandeur_band": "high",
    "texture": 0.1026, "grandeur": 0.722
  },
  "grandeur_parts": { "dynamic_range_db": 0.950, "full_loud_share": 0.519 }
}
```

`--human` 给可读摘要，`--fields` 裁字段，`--no-segments` 跳过时间线。

整目录：

```
python moodio.py batch --path "~/Music" --out library.jsonl --workers 6
```

每首一行 JSON，追加写入，所以重跑即续跑：已完成的跳过，失败的会重试。进度走
stderr，stdout 始终可解析。

跑完之后出报告：

```
python moodio.py report --path library.jsonl --calibration fixed
```

从 agent 调用：

```
analyze_music_emotion(path: "/music/track.flac")
batch_analyze_music(path: "/music", workers: 6)
```

## 限制

这是规则先验，不是训练出来的模型。权重来自音乐心理学文献里的方向性结论，不是
拟合听者评分。它适合在同一批结果里排序。不要把数值当成人群平均来引用。

完全不用歌词。歌词很惨但伴奏明亮的歌，会被判成明亮。

两套标定模式不能混用。`fixed`（默认）把零点放在"无证据即中性"，因此不同曲库
之间可比。`corpus-relative` 改用当前曲库中位数作零点，偏斜的库上四个象限都能
用到，但跨库不可比。

`confidence` 是输入质量，不是准确率。短、轻、无调性、无节拍的音频分数会低。

速度检测有倍频问题。自相关常在 `t` 和 `2t` 同时出峰，95 BPM 可能实际是 184。
速度占 arousal 权重的 0.30，错一个倍频结果就偏了。

削波的母带在分析前就已经丢了动态。`peak > 1.0` 会置 `clipping_suspected`，
warning 里会说明 arousal 可能被低估。

用 `--max-seconds` 截断可能把曲子的性格截掉。安静的前奏会藏住 grandeur。
管弦类素材建议整曲分析。

那个让 523 首里 264 首被判"低沉"的标定 bug 写在
[docs/algorithm.md](docs/algorithm.md) 里，和踩过的其它坑放在一起。

## TODO

- 没有人工标注真值。现在只对合成刺激和内部一致性做了验证，这比听起来要弱。
- 神经侧写（`--neural`）是语音模型用在音乐上。绝对值被压得很扁，只有相对起伏
  能用。还没接进四轴判定。
- `texture` 和 `grandeur` 的阈值是在几十首曲子上定的。换成自由爵士或实地录音
  大概率不成立。
- batch 对工具 API 没有进度回调，只有 stderr。
- Windows 控制台除 `-X utf8` 外没做额外编码处理。

## 目录

```
index.js              DSH host 插件
cordis.patch.yml      bundle patch
locale/{en,zh}.json   插件元数据
icon.svg
test/offline.test.mjs 插件契约测试，不依赖 DSH 运行时
python/
  moodio.py           CLI
  music_emotion/      分析器包
    vamodel.py        由声学线索算 valence/arousal
    texture.py        texture 与 grandeur
    four_axis.py      四轴 -> 情绪词
    emotion_taxonomy.py   两轴回退词表
    features.py       librosa 特征提取，整曲只算一次
    pipeline.py       流程编排
    batch.py          可续跑的整目录处理
    report.py         markdown 与 JSON 汇总
    neural.py         可选 wav2small 侧写
  agent/              agent 用法契约（brief、rules、skills）
  tests/              自包含测试套件
```

## 测试

```
npm run test:fast      # 插件 + 五套分析器（64）
npm run test:python    # 与 CI 相同的六套（45）
npm run test:all       # 全部（73）
npm test               # 只跑插件契约（28）
```

Python 测试自己合成刺激或用固定特征探针，都不需要音频文件。CI 每次 push 跑两半。

## 许可证

MIT。可选的神经侧写针对 `audeering/wav2small` 实现，其权重为 CC-BY-NC-SA-4.0，
仓库里没有包含。见 [LICENSE](LICENSE)。
