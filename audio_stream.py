"""音频采集模块：把电脑麦克风/声卡当作一块 ADC，实时采集音频流。

为什么用 sounddevice：它是对 PortAudio 的轻量封装，回调式采集，
底层音频线程在后台跑，主线程（GUI 线程）随时可以取走最近的数据，
不会互相阻塞，延迟低。
"""

import collections

import numpy as np
import sounddevice as sd


class AudioStream:
    """基于 sounddevice(PortAudio) 的实时输入流。

    用法：
        src = AudioStream()
        src.start()                      # 底层采集线程开始往环形缓冲写
        samples = src.read_latest(8)     # 取出最近 8 个数据块拼成一段波形
        src.stop()
    """

    def __init__(self, sample_rate=44100, block_size=1024, channels=1, device=None):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.channels = channels
        self.device = device
        self._buffer = collections.deque(maxlen=32)  # 环形缓冲，只保留最近 32 块
        self._stream = None

    def start(self):
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            channels=self.channels,
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"[audio] {status}", flush=True)
        self._buffer.append(indata[:, 0].copy())  # 只取第一个声道

    def read_latest(self, num_blocks):
        """取出最近 num_blocks 个数据块，拼接成一段连续波形。"""
        blocks = list(self._buffer)[-num_blocks:]
        if blocks:
            return np.concatenate(blocks)
        return np.zeros(num_blocks * self.block_size, dtype=np.float32)

    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


class DemoSynth:
    """演示信号源：合成 440Hz + 1000Hz 正弦波并叠加一点噪声。

    实现了和 AudioStream 一样的 start()/read_latest()/stop() 接口，
    在没有麦克风（或不想开麦）的机器上也能看到完整效果。
    440Hz 正好是钢琴的 A4 音，方便你验证音高检测。
    """

    def __init__(self, sample_rate=44100, block_size=1024, device=None):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.device = device
        self._sample = 0  # 全局采样序号，保证正弦波形连续

    def start(self):
        pass

    def read_latest(self, num_blocks):
        n = num_blocks * self.block_size
        t = self._sample + np.arange(n) / self.sample_rate
        signal = (
            0.5 * np.sin(2 * np.pi * 440.0 * t)      # 440 Hz 主音
            + 0.3 * np.sin(2 * np.pi * 1000.0 * t)   # 1000 Hz 泛音
            + 0.05 * np.random.randn(n)              # 模拟环境噪声
        )
        self._sample += n
        return signal.astype(np.float32)

    def stop(self):
        pass
