"""实时绘图模块：用 matplotlib 同时显示时域波形与频域 FFT 频谱。

和信号源解耦——只要传进来的 source 实现了 start() / read_latest(n) / stop()
这三个接口即可（AudioStream 和 DemoSynth 都可以）。
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from dsp import to_db, peak_frequency, windowed_fft


class SpectrumAnalyzer:
    """一个 matplotlib 双子图窗口：上 = 时域波形，下 = FFT 频谱（带峰值标记）。"""

    def __init__(self, source, sample_rate=44100, block_size=1024,
                 plot_blocks=8, fmin=80.0, fmax=20000.0):
        self.source = source
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.plot_blocks = plot_blocks       # 每帧拼接多少个音频块
        self.fmin = fmin
        self.fmax = fmax
        self.wave_len = block_size * plot_blocks
        self.anim = None

        # ---- 绘图窗口 ----
        self.fig, (self.ax_wave, self.ax_spec) = plt.subplots(2, 1, figsize=(11, 7))
        self.fig.canvas.manager.set_window_title("实时频谱分析仪 / 简易示波器")

        # 上：时域波形
        t = np.arange(self.wave_len) / sample_rate
        self.wave_line, = self.ax_wave.plot(t, np.zeros(self.wave_len), lw=1, color="tab:blue")
        self.ax_wave.set_xlim(0, t[-1])
        self.ax_wave.set_ylim(-1.0, 1.0)
        self.ax_wave.grid(True, alpha=0.3)
        self.ax_wave.set_xlabel("时间 (s)")
        self.ax_wave.set_ylabel("幅度")
        self.ax_wave.set_title("时域波形")

        # 下：FFT 频谱
        self.freqs = np.fft.rfftfreq(self.wave_len, 1 / sample_rate)
        self.spec_line, = self.ax_spec.plot(
            self.freqs, np.zeros(self.freqs.size), lw=1, color="tab:orange")
        self.peak_marker, = self.ax_spec.plot([], [], "rv", ms=10)  # 红色三角标峰值
        self.ax_spec.set_xlim(0, sample_rate / 2)
        self.ax_spec.set_ylim(-90, 10)
        self.ax_spec.grid(True, alpha=0.3)
        self.ax_spec.set_xlabel("频率 (Hz)")
        self.ax_spec.set_ylabel("幅度 (dB)")
        self.ax_spec.set_title("FFT 频谱")

        self.fig.tight_layout()

    # ------------------------------------------------------------ 实时刷新
    def _update(self, _frame):
        samples = self.source.read_latest(self.plot_blocks)

        # 时域：直接更新波形
        self.wave_line.set_ydata(samples)

        # 频域：加窗 FFT -> dB，并标出检测到的峰值频率
        freqs, mag = windowed_fft(samples, self.sample_rate)
        db = to_db(mag)
        self.spec_line.set_ydata(db)

        fpeak = peak_frequency(freqs, mag, self.fmin, self.fmax)
        idx = np.searchsorted(freqs, fpeak)
        peak_db = db[idx] if 0 <= idx < db.size else -90.0
        self.peak_marker.set_data([fpeak], [peak_db])
        self.ax_spec.set_title(f"FFT 频谱 | 峰值频率: {fpeak:8.1f} Hz")

        return self.wave_line, self.spec_line, self.peak_marker

    def run(self, interval=30):
        """开始采集并进入 GUI 主循环。interval 为刷新间隔（毫秒）。"""
        self.source.start()
        self.anim = FuncAnimation(self.fig, self._update, interval=interval, blit=False)
        try:
            plt.show()
        finally:
            self.anim.event_source.stop()
            self.source.stop()
