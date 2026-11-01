---
name: workflow
description: 调用 moodio 的标准流程
---

## 单曲

1. 默认读取整首并分析旋律，不自动截取开头；仅在用户明确要求预览时设置 `--max-seconds`：

   ```
   moodio analyze --path "<file>" --no-segments
   ```

2. 正式分析。长音频用 `--segment-sec 6 --hop-sec 3` 控制 token 量
   （默认 5/1 会在 5 分钟的歌上产生约 300 个分段）：

   ```
   moodio analyze --path "<file>" --segment-sec 6 --hop-sec 3
   ```

3. 读结果时**先看 quality.confidence 和 warnings**，再看 emotion：

   - `confidence < 0.6`：结论只当参考，不要据此做不可逆决策。
   - `clipping_suspected`：动态特征失真，arousal 可能被低估。
   - `n_segments` 太少：弧线分析没有意义。

4. 解释结论时用 `modes.acoustic` 里的 top-3 贡献特征，不要编造理由。
   例：`arousal` 主要来自 `tempo_bpm` 与 `onset_rate`，说明是"速度快 + 音符密"。

5. 需要交叉验证时**显式**加 `--neural`（默认**不开**，因为要 import torch）。
   注意 `neural.domain_mismatch` 恒为 true，它的绝对值不可用，
   只看 `neural.agreement.arousal_r` 是否与声学基线同向。
   实测本项目两首电子乐 `arousal_r = 0.066`（弱/无关）：这类素材上它基本没有信息量，
   不要因为"多接了一个模型"就默认更可信。

6. 只关心某一维度时用 `--fields emotion,arc` 省 token。

7. 问音符走向或重复时用 `--melody --fields clip,melody,warnings`；按
   `melody.agent_summary.evidence` 引用秒数。`status: insufficient_evidence`
   时拒绝推断；混音结果只是候选，不能证明已找到主旋律或副歌。

## 整库

整曲分析耗时取决于时长、CPU 和模型加载；整库用 `batch` 复用 worker 中的乐器模型：

```
moodio batch --path "<dir>" --out results.jsonl --workers 1 --segment-sec 10 --hop-sec 10
```

- 逐条**追加**写入 JSONL，进程被杀也不丢已完成的部分。
- 重跑同一条命令即续跑：已成功的跳过，失败的**会重试**。
- `--limit N` 只处理 N 首（试跑用）；`--no-resume` 清空重来。
- 进度走 **stderr**，stdout 只有最终汇总 JSON，可以安全重定向。
- 默认乐器模型会在每个 worker 中单独加载，默认使用 1 个 worker；确认内存充足后再增加。
- 用 `music_analysis` 联合解释乐器、旋律与十八轴；Host 自动附上联网语境。旧 JSONL 续跑不会补充新字段，需使用新的输出路径。

再汇总成报告：

```
moodio report --path results.jsonl --calibration fixed
```

产出同名 `.md` + `.json`。`--calibration corpus-relative` 表示把 0 点
设为本库中位数（跨库不可比），`fixed`（默认）才是跨库可比的绝对坐标。

## 退出码与错误

退出码：`0` 成功；`1` 一般错误（`batch` 有失败项时也是 1）；`2` 参数错误；
`10` 鉴权失败；`11` 权限不足；`20` 文件/资源不存在；`30` 冲突/前置条件不满足。

错误一律是 stderr 上的 JSON：`{"error":true,"code":...,"message":...,"suggestion":...}`，
stdout 保持为空，因此可以直接管道给 `jq`。
