# -*- coding: utf-8 -*-
"""神经侧写：audeering/wav2small 的 V-A-D 三维回归，作为独立的交叉验证信号。

为什么用它、又为什么只当"交叉信号"：
    - 优点：15K 参数、CPU 上每 10s 窗口约 75ms，无需 MERT 这类 330M 骨干。
    - 缺点：它是**语音**情感模型（MSP-Podcast 训练），迁移到音乐上会出现
      严重的分布压缩（实测音乐样本 arousal 只在 0.56~0.71 之间抖动），
      绝对值没有意义，但**窗口之间的相对起伏仍有信息量**。
    因此输出的是"标准化后的相对趋势"，并显式标注 domain_mismatch。

实现说明（为什么不依赖 transformers）：
    官方示例用 Wav2Vec2PreTrainedModel + PretrainedConfig，只是为了复用权重加载。
    但这里是 15K 参数的全卷积网络、权重直接来自 safetensors，
    引入 transformers 只会让 import 多花约 6 秒，且带来版本耦合。
    所以用纯 nn.Module 复刻结构，load_state_dict 直接加载。

降级约定：
    模型文件缺失或推理失败时**不抛错**，返回 {"available": false, ...}，
    让主流程继续走可解释的声学基线。
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np

__all__ = ["NeuralProfiler", "model_available", "DEFAULT_MODEL_DIR"]

DEFAULT_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_research", "wav2small"
)

#: wav2small 要求 16kHz 单声道输入。
NEURAL_SR = 16000
#: 每个推理窗口长度（秒）。全卷积结构支持变长，10s 是精度/开销的折中。
WINDOW_SEC = 10


def model_available(model_dir: str = DEFAULT_MODEL_DIR) -> bool:
    return os.path.isfile(os.path.join(model_dir, "config.json")) and os.path.isfile(
        os.path.join(model_dir, "model.safetensors")
    )


def _build_model():
    """构造与 wav2small 检查点一致的模型（纯 nn.Module）。"""
    import librosa
    import torch
    from torch import nn

    def _prenorm(x, attention_mask=None):
        if attention_mask is not None:
            n = attention_mask.sum(1, keepdim=True)
            x -= x.sum(1, keepdim=True) / n
            var = (x * x).sum(1, keepdim=True) / n
        else:
            x -= x.mean(1, keepdim=True)
            var = (x * x).mean(1, keepdim=True)
        return x / torch.sqrt(var + 1e-7)

    class Spectrogram(nn.Module):
        """用固定权重的 Conv1d 实现前 33 点 DFT。"""

        def __init__(self, n_fft=64, n_time=64, hop_length=32):
            super().__init__()
            fft_window = librosa.filters.get_window("hann", n_time, fftbins=True)
            fft_window = librosa.util.pad_center(fft_window, size=n_time)
            out_channels = n_fft // 2 + 1
            (x, y) = np.meshgrid(np.arange(n_time), np.arange(n_fft))
            omega = np.exp(-2 * np.pi * 1j / n_time)
            dft = np.power(omega, x * y) * fft_window[None, :]
            dft = dft[0:out_channels, :][:, None, :]
            self.conv_real = nn.Conv1d(1, out_channels, n_fft, stride=hop_length, bias=False)
            self.conv_imag = nn.Conv1d(1, out_channels, n_fft, stride=hop_length, bias=False)
            # 必须用 copy_：直接给 .data 赋值在 torch 2.x 会报 incompatible tensor type
            # （set_data 对非 contiguous 源张量失败）。
            with torch.no_grad():
                self.conv_real.weight.copy_(torch.tensor(np.real(dft), dtype=torch.float32))
                self.conv_imag.weight.copy_(torch.tensor(np.imag(dft), dtype=torch.float32))
            for p in self.parameters():
                p.requires_grad = False

        def forward(self, x):
            x = x[:, None, :]
            return self.conv_real(x) ** 2 + self.conv_imag(x) ** 2

    class LogmelFilterBank(nn.Module):
        def __init__(self, sr=16000, n_fft=64, n_mels=26, fmin=0.0):
            super().__init__()
            w2 = librosa.filters.mel(
                sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=sr // 2
            ).T
            self.register_buffer("melW", torch.Tensor(w2))
            self.register_buffer("amin", torch.Tensor([1e-10]))

        def forward(self, x):
            x = torch.matmul(x[:, None, :, :].transpose(2, 3), self.melW)
            x = torch.where(x > self.amin, x, self.amin)
            return 10 * torch.log10(x)

    class Conv(nn.Module):
        def __init__(self, c_in, c_out, k=3, stride=1, padding=1):
            super().__init__()
            self.conv = nn.Conv2d(c_in, c_out, k, stride=stride, padding=padding, bias=False)
            self.norm = nn.BatchNorm2d(c_out)

        def forward(self, x):
            return torch.relu_(self.norm(self.conv(x)))

    class Vgg7(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1, self.l2, self.l3 = Conv(1, 13), Conv(13, 13), Conv(13, 13)
            self.maxpool_A = nn.MaxPool2d(3, stride=2, padding=1)
            self.l4, self.l5 = Conv(13, 13), Conv(13, 13)
            self.l6, self.l7 = Conv(13, 13), Conv(13, 13)
            self.lin = nn.Conv2d(13, 13, 1, bias=False)
            self.sof = nn.Conv2d(13, 13, 1, bias=False)
            self.spectrogram_extractor = Spectrogram()
            self.logmel_extractor = LogmelFilterBank()

        def forward(self, x, attention_mask=None):
            x = _prenorm(x, attention_mask=attention_mask)
            x = self.logmel_extractor(self.spectrogram_extractor(x))
            x = self.l1(x); x = self.l2(x); x = self.l3(x)
            x = self.maxpool_A(x)
            x = self.l4(x); x = self.l5(x); x = self.l6(x); x = self.l7(x)
            # 原实现：lin(x) * softmax(sof(x), dim=mels)。
            # 注意 lin 是**带权重的** 1x1 卷积（只有 bias 不存在），不能省略，
            # 否则数值会偏（实测 arousal 均值从 0.6296 掉到 0.549）。
            x = self.lin(x) * self.sof(x).softmax(2)
            x = x.sum(2)
            x = torch.cat([x, torch.bmm(x, x.transpose(1, 2))], 2)
            return x.reshape(-1, 338)

    class Wav2Small(nn.Module):
        def __init__(self):
            super().__init__()
            self.vgg7 = Vgg7()
            self.adv = nn.Linear(338, 3)  # 0=arousal, 1=dominance, 2=valence

        def forward(self, x, attention_mask=None):
            return self.adv(self.vgg7(x, attention_mask=attention_mask))

    # 注意：必须返回实例而不是类。早期写成 `return Wav2Small`，
    # 导致 load_state_dict 变成未绑定方法调用、报 "missing 1 required positional argument"。
    return Wav2Small()


class NeuralProfiler:
    """按窗口跑 wav2small，输出相对趋势（标准化后）。"""

    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR, window_sec: float = WINDOW_SEC) -> None:
        self.model_dir = model_dir
        self.window_sec = window_sec
        self._model = None
        self._torch = None

    def _load(self):
        if self._model is not None:
            return self._model
        import torch
        from safetensors.torch import load_file

        model = _build_model()
        state = load_file(os.path.join(self.model_dir, "model.safetensors"))
        if not isinstance(state, dict):
            raise RuntimeError(f"safetensors returned {type(state).__name__}, expected dict")
        missing, _ = model.load_state_dict(state, strict=False)
        # lin/sof 的 bias 在检查点里不存在（官方实现里这两个卷积无 bias），属预期。
        real_missing = [k for k in missing if not k.endswith(("lin.bias", "sof.bias"))]
        if real_missing:
            raise RuntimeError(f"checkpoint missing weights: {real_missing[:5]}")

        self._torch = torch
        self._model = model.eval()
        return self._model

    def profile(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """对整条音频做逐窗推理。

        Returns:
            available / model / domain_mismatch / raw_mean / dynamic_range /
            series / relative(z 分)
        """
        if not model_available(self.model_dir):
            return {
                "available": False,
                "reason": "model files not found",
                "model_dir": self.model_dir,
                "hint": "download config.json + model.safetensors from audeering/wav2small",
            }

        try:
            import librosa

            model = self._load()
            torch = self._torch

            if sr != NEURAL_SR:
                y16 = librosa.resample(y, orig_sr=sr, target_sr=NEURAL_SR, res_type="soxr_hq")
            else:
                y16 = y

            win = int(self.window_sec * NEURAL_SR)
            hop = max(win // 2, 1)  # 50% 重叠，抑制窗口边界抖动
            if y16.size < win:
                y16 = np.pad(y16, (0, win - y16.size))

            rows = []
            for start in range(0, y16.size - win + 1, hop):
                seg = torch.from_numpy(np.ascontiguousarray(y16[start:start + win]))[None, :]
                with torch.no_grad():
                    probs = torch.sigmoid(model(seg))[0]
                rows.append((
                    start / NEURAL_SR,
                    float(probs[0]),  # arousal
                    float(probs[1]),  # dominance
                    float(probs[2]),  # valence
                ))
        except Exception as exc:  # noqa: BLE001 - 侧写失败不应打断主流程
            return {
                "available": False,
                "reason": f"inference failed: {type(exc).__name__}: {exc}",
                "model_dir": self.model_dir,
            }

        arr = np.array(rows, dtype=np.float64)
        a, v = arr[:, 1], arr[:, 3]

        def _z(x: np.ndarray) -> List[float]:
            sd = float(x.std())
            if sd < 1e-6:
                return [0.0] * len(x)
            return [round(float(t), 3) for t in (x - x.mean()) / sd]

        return {
            "available": True,
            "model": "audeering/wav2small",
            "params": 15030,
            "domain_mismatch": True,
            "domain_note": (
                "trained on speech emotion (MSP-Podcast); absolute values compress on music, "
                "only relative dynamics are usable"
            ),
            "raw_mean": {
                "arousal": round(float(a.mean()), 4),
                "dominance": round(float(arr[:, 2].mean()), 4),
                "valence": round(float(v.mean()), 4),
            },
            "dynamic_range": {
                "arousal": round(float(a.max() - a.min()), 4),
                "valence": round(float(v.max() - v.min()), 4),
            },
            "series": [
                {
                    "t": round(float(arr[i, 0]), 2),
                    "arousal": round(float(arr[i, 1]), 4),
                    "dominance": round(float(arr[i, 2]), 4),
                    "valence": round(float(arr[i, 3]), 4),
                }
                for i in range(len(arr))
            ],
            "relative": {"arousal_z": _z(a), "valence_z": _z(v)},
        }
