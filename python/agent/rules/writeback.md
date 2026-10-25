---
name: writeback
description: 发现问题时如何反馈
---

当输出与预期不符（数值明显不合理、方向反了、字段缺失、耗时异常）时：

```
python moodio.py issue create --type bad-output --message "描述现象与输入文件"
```

`--type` 取值：`bug` / `requirement` / `suggestion` / `bad-output`。

查看：`python moodio.py issue list`
查看单条：`python moodio.py issue show --id <n>`

反馈里请带上：

- 输入文件（路径 + 时长）
- 你期望的方向与实测方向
- 是否用了 `--max-seconds` / `--no-neural`

不要提交无法复现的"感觉不对"。能定位到具体特征（如 `spectral_centroid_hz` 顶格）最有价值。
