---
name: brief
description: moodio 是什么、能给 agent 什么
---

moodio 默认分析整首的 21 类乐器候选、旋律和十八轴，以四轴情绪先验为起点。Host 工具随后自动检索背景和歌词来源，并返回 `music_analysis` 供 agent 综合解释音乐情感、引用时间戳及来源、说明证据冲突。不能仅复述四轴词，也不能用乐器名称套固定情绪。独立 CLI 保留离线能力；模型或网络缺失时显式标记，保留可用证据，不编造识别结果或来源。原始声学数值不被网络文本改写。

乐器使用本地 CLAP 候选匹配和 AST 类别互证，需一次安装模型。相似度不是准确率，21 类目录不保证全部准确识别；中国乐器中不少类别只有家族或零样本支持。结合 `instruments.segments`、`verification_scope` 和歧义标记判断，详见 `interpreting-instruments`。

单曲和批量工具默认听完整首并分析旋律，无需额外开关：返回带时间戳的候选音符、乐句走向、重复动机与 agent 证据。不要自行用 `max_seconds` 截取开头；只有用户明确要求预览时才限制时长。只在 `melody.status: candidate` 时解释；单声部音高估计未经分轨，不保证混音中的候选就是主旋律，也不能由重复推断副歌。需要跳过旋律时传 `include_melody: false`（CLI: `--no-melody`）。
