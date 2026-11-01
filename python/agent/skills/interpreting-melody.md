---
name: interpreting-melody
description: 根据带时间戳的候选音符、乐句与重复动机回答旋律问题
---

## 调用

用户问旋律如何起伏、哪里重复、哪段像副歌时，用
`moodio analyze --path "<file>" --fields clip,melody,warnings`；
DSH 插件使用 `analyze_music_emotion(path: "...")`，默认启用整首旋律分析。
不要自行截取前 30 秒；仅在用户明确要求预览时传 `max_seconds`。
插件完成本地分析后仍会自动联网检索背景和歌词。

## 解释顺序

1. 先看 `melody.status`。只有 `candidate` 才能解读 `notes`、`phrases` 和 `motifs`；
   `insufficient_evidence` 时说明原因，不推断走向或重复。
2. 先报告 `melody.agent_summary.overview.note_coverage`，扫描整首不等于整首都有可靠音符。
   用 `sections` 覆盖前、中、后段，缺少证据的区间明确留空，不用开头一句概括整首。
   引用 `evidence` 的时间戳；`contour` 只描述对应的代表乐句。
3. 始终称“候选旋律”；单声部 pYIN 未分离混音，可能跟踪伴奏或贝斯。
   `notes[].voicing_probability` 只表示有声概率，不是音高准确率，
   更不能证明它属于主旋律。
4. 检查 `quality.boundary_validation` 中复核前后的边界音符数量与剔除帧数。
   边界音高只在扩大搜索音域后仍一致时保留；通过复核也不证明它属于主旋律。
5. `motifs` 来自跨分句的音符滑窗，匹配音程和相对节奏；为空时阅读 `motif_search` 的原因，
   不得把“未检出”说成“歌曲没有重复”。重复动机不是自动识别副歌；没有段落/结构证据时不要把它称为副歌。
   在线资料只能解释作品背景，不能用歌词或搜索结果替代听觉证据。
