# moodio

**给 AI agent 用的音乐情感分析。** 一个确定性的本地分析器，把音频变成可据以行动的情绪结论，
外加两个让 DSH agent 直接调用的工具。

[English](README.md) · [算法说明](docs/algorithm.md) · [更新日志](CHANGELOG.md)

---

## 为什么需要另一个音乐情绪工具

多数模型把情绪压成两个数——效价（valence）和唤醒度（arousal）——再按象限取一个词。
这会丢掉人真正关心的区别：

| 二维坐标说 | 实际可能是 |
|---|---|
| 低唤醒 + 小调 | **悲伤**（内敛的忧伤）*或* **悲壮**（庄严的宏大） |
| 高唤醒 + 大调 | **欢快**（轻快跳脱）*或* **热烈**（高能外放） |

一首慢速钢琴抒情和一首动态 34 dB 的管弦史诗可能落在同一个点上。moodio 增加两个维度把它们分开：

```
valence   -1..1   情绪正负        （调式、和声）
arousal   -1..1   激活程度        （速度、起音密度）
texture    0..1   攻击性/粗糙度    （失真、打击感、噪声）
grandeur   0..1   规模感/重量      （动态范围、持续强奏、厚织体）
```

`texture` 与 `grandeur` 是**刻意独立**的。管弦的力量来自规模而非粗糙度：
一首曲子可以是 `texture 0.10 / grandeur 0.72`——这正是「悲壮」的特征——
而只知道「攻击性」的三维模型会把它误读成「安静」。

## 四轴的实际表现

真实曲目实测（数值来自 `moodio analyze`）：

| 曲目 | valence | arousal | texture | grandeur | 结论 |
|---|---|---|---|---|---|
| 在虚无中永存 - 英雄主义 | −0.23 | −0.11 | **0.10** | **0.72** | **悲壮** |
| 邓垚 - 诀别书 | −0.16 | −0.13 | 0.15 | 0.51 | **悲伤** |
| Two Steps From Hell - Victory | +0.28 | +0.04 | 0.33 | 0.69 | **壮丽** |
| Coldplay - Yellow | +0.21 | −0.24 | 0.36 | 0.55 | **平静** |
| Bruno Mars - Runaway Baby | +0.17 | +0.47 | **0.57** | 0.48 | **热烈** |

两首负效价的曲子只差在 grandeur 上，而这恰好就是史诗与挽歌的差别。

## 安装

### 作为 DSH 插件（给你的 agent 两个工具）

```bash
git clone https://github.com/chengzhouSAMA/dsh-Moodio.git
cd dsh-Moodio
pip install -r python/requirements.txt
```

然后在 DSH profile 里作为 bundle 安装——分析器随包分发，无需配置路径：

```text
plugin_manager(action: "install_bundle", target: "<moodio 目录的绝对路径>")
```

会注册 `analyze_music_emotion` 与 `batch_analyze_music`。

### 作为独立 CLI

```bash
cd python
python moodio.py analyze --path "/path/to/track.flac"
```

## 用法

### 单曲

```bash
python moodio.py analyze --path "song.flac" --segment-sec 10 --hop-sec 10
python moodio.py analyze --path "song.flac" --human      # 人类可读
```

### 整库

```bash
python moodio.py batch --path "~/Music" --out library.jsonl --workers 6
```

断点续跑：重跑会跳过已成功的曲目、重试失败的。进度走 stderr，stdout 始终可解析。

### 报告

```bash
python moodio.py report --path library.jsonl --calibration fixed
```

产出 Markdown 与机器可读 JSON：分布、象限占比、置信度、弧型、失败明细。

### 从 agent 调用

```text
analyze_music_emotion(path: "/music/track.flac")
batch_analyze_music(path: "/music", workers: 6)
```

## 它不做什么

明确边界是工具的一部分：

- **不读歌词。** 歌词悲恸但伴奏轻快的歌会被判为轻快。
- **没有人工评分参与。** 权重来自音乐心理学文献中的方向性结论，不是拟合听众评分。
  方向与相对排序可靠；绝对值不是人群平均。请在同一批结果内比较，不要直接引用数字。
- **两套标定模式不能混用。** `fixed`（默认）让零点表示「无证据即中性」，因此跨曲库可比；
  `corpus-relative` 改用本库中位数作零点，在偏斜的曲库上能用满四象限，但跨库不可比。
- **`confidence` 衡量输入质量**（时长、响度、调性清晰度），不是预测准确率。

更深入的说明——包括那次让 523 首里 264 首被判「低沉」的标定故障——
见 [docs/algorithm.md](docs/algorithm.md)。

## 目录结构

```
dsh-Moodio/
├── index.js              # DSH Host 插件：注册两个工具
├── cordis.patch.yml      # bundle patch
├── locale/{en,zh}.json   # 插件显示元数据
├── icon.svg
├── test/offline.test.mjs # 插件契约测试（不依赖 DSH 运行时）
└── python/
    ├── moodio.py         # 面向 agent 的 CLI
    ├── requirements.txt
    ├── music_emotion/    # 分析器包
    ├── agent/            # 机器可读用法契约（brief/rules/skills）
    └── tests/            # 自包含测试（不需要音频语料）
```

## 测试

```bash
npm test                 # 插件契约测试（28 项）
npm run test:python      # 分析器单元 + 标定测试（36 项）
npm run test:all
```

Python 测试自行合成刺激或使用固定特征探针，**不需要任何音频文件**即可运行。
CI 在每次推送时同时跑两半。

## 环境要求

- Python ≥ 3.9，依赖 `numpy`、`librosa`、`soundfile`
- Node ≥ 18（仅插件部分）
- 音频解码走 libsndfile；libsndfile 带 mpg123 时支持 MP3（常见情况）。
  `m4a`/`aac` 需先转换。

## 许可证

MIT。可选的神经侧写针对 `audeering/wav2small` 实现，其权重为 CC-BY-NC-SA-4.0，
**本仓库不包含**该权重——详见 [LICENSE](LICENSE) 的第三方声明。
