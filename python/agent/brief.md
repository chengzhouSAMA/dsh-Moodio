---
name: brief
description: moodio 是什么、能给 agent 什么
---

moodio 把音频文件转成**情绪坐标**（valence、arousal、texture、grandeur），附逐段时间线、情感弧线与声学证据。可选的 `include_axes_v2` 还会附上整段已分析音频的十二轴声学诊断；它是独立量纲，不覆盖原四轴。独立 Python CLI 仍可离线运行；DSH 插件每次调用先联网检索创作背景和歌词，并排除歌曲评论，搜索失败则不分析。联网来源只供解释，不改变声学评分。可选的本地 `faster-whisper` 模型可猜测唱词；未安装或缺权重时不伪称已识别歌词。
