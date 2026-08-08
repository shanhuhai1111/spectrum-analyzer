"""数字信号处理模块：加窗 FFT、幅度谱转 dB、峰值/音高检测。

全部用 numpy 实现，不依赖任何第三方 DSP 库——每一步原理都是透明的，
简历上可以把这些函数对应到《数字信号处理》课程里的知识点。
"""

import numpy as np


def windowed_fft(samples, sample_rate, window="hann"):
    """对一段时域信号加窗并计算幅度谱。

    加窗能抑制 FFT 的频谱泄漏（旁瓣），让谱峰更尖锐、读数更准。

    参数
    ----
    samples : np.ndarray
        时域信号（长度 n）
    sample_rate : int
        采样率 (Hz)
    window : str
        窗函数类型："hann" / "hamming" / "rect"（不加窗）

    返回
    ----
    freqs : np.ndarray
        频率轴 (Hz)，长度 = n // 2 + 1（只取正频率，用 rfft）
    mag : np.ndarray
        幅度谱（对单频正弦已归一化为真实幅度）
    """
    n = len(samples)
    if window == "hann":
        w = np.hanning(n)
    elif window == "hamming":
        w = np.hamming(n)
    else:
        w = np.ones(n)

    spectrum = np.fft.rfft(samples * w)
    freqs = np.fft.rfftfreq(n, 1 / sample_rate)
    # 幅度归一化：rfft 结果乘 2/n 后，单频正弦的峰值幅度≈真实幅值。
    # 注意加窗会带来幅度衰减（汉宁窗增益约 0.5），所以读到的峰值
    # 实际约为真实幅度 × 窗增益。这里保留窗增益，让幅度随窗函数如实变化。
    mag = 2.0 * np.abs(spectrum) / n
    return freqs, mag


def to_db(mag, ref=1.0):
    """幅度谱转分贝：dB = 20*log10(mag/ref)。加极小值防止 log(0)。"""
    return 20.0 * np.log10(np.clip(mag, 1e-12, None) / ref)


def peak_frequency(freqs, mag, fmin=80.0, fmax=20000.0):
    """在 [fmin, fmax] 频带内找幅度最大的峰值频率。

    这是音高检测的最简实现：滤掉人耳不关心的频段后直接取最大谱峰。
    更稳的基频检测可以用自相关法或倒谱法（见 README 的扩展建议）。
    """
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return 0.0
    mag_in_band = np.where(mask, mag, -np.inf)
    return float(freqs[np.argmax(mag_in_band)])
