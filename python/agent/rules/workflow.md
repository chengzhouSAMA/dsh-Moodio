---
name: workflow
description: 调用 moodio 的标准流程
---

## 单曲

1. 先用 `--max-seconds 30` 快速试跑，确认路径与格式没问题：

   ```
   moodio analyze --path "<file>" --max-seconds 30 --no-segments
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

## 整库

单曲 5 分钟的歌约 30~50 秒，整库不要循环调用 `analyze`——用 `batch`：

```
moodio batch --path "<dir>" --out results.jsonl --workers 6 --segment-sec 10 --hop-sec 10
```

- 逐条**追加**写入 JSONL，进程被杀也不丢已完成的部分。
- 重跑同一条命令即续跑：已成功的跳过，失败的**会重试**。
- `--limit N` 只处理 N 首（试跑用）；`--no-resume` 清空重来。
- 进度走 **stderr**，stdout 只有最终汇总 JSON，可以安全重定向。

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
