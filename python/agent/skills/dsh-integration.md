---
name: dsh-integration
description: 把 moodio 接进 DSH 作为工具或 skill
---

两种接法，按需要选：

## A. 作为 DSH Skill（最轻，无需写插件）

1. 在 `~/.dsh/skills/` 下建目录 `music-emotion/`。
2. 放一个 `SKILL.md`，frontmatter 写 `name` / `description`，正文写调用约定，
   内容可直接引用本项目 `agent/rules/*.md`。
3. 正文里给出绝对路径命令，例如：

   ```
   python C:\Users\35968\Desktop\英雄主义\moodio.py analyze --path "<file>"
   ```

4. 重启会话后，用户提到"分析音乐情绪"时该 skill 会被加载。

适用：只要 agent 会读文件、能跑命令即可。**不涉及 DSH 插件 API。**

## B. 作为 DSH Host 插件（注册成真正的 Tool）

需要按 Cordis 插件规范写一个包，用 `ctx.tool(...)` 之类的方式注册
`analyze_music_emotion`，参数为音频路径与分段粒度，内部调用本项目的
`music_emotion.analyze_file` 并把 `report.to_dict()` 作为工具返回值。

这样做的收益：工具 schema 进入 agent 的可见工具列表，不需要 agent 先查 skill 文档。
代价：需要处理插件打包、Python 子进程调用、超时与错误映射。

具体插件 API 请先用 `cordis_inspect_list` / `cordis_inspect_query` 查当前 profile 的
Tool 与 Service 契约，再动手写——不要凭记忆猜方法名。
