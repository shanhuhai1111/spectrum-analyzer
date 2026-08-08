"""实时频谱分析仪 / 简易示波器 —— 程序入口。

用法：
    python main.py                 # 用系统默认麦克风
    python main.py --device 2      # 指定输入设备（不带参数先运行会打印设备列表）
    python main.py --demo          # 不接麦克风，用合成信号演示
"""

import argparse

import sounddevice as sd

from audio_stream import AudioStream, DemoSynth
from viz import SpectrumAnalyzer


def parse_args():
    parser = argparse.ArgumentParser(
        description="实时频谱分析仪 / 简易示波器（纯软件，用声卡当 ADC）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", type=int, default=None,
                        help="输入设备索引；不指定时用系统默认输入设备")
    parser.add_argument("--rate", type=int, default=44100, help="采样率 (Hz)")
    parser.add_argument("--block", type=int, default=1024, help="每个音频块的采样点数")
    parser.add_argument("--blocks", type=int, default=8,
                        help="每帧拼接的音频块数，越大频率分辨率越高、刷新越慢")
    parser.add_argument("--demo", action="store_true",
                        help="演示模式：用合成正弦波代替麦克风")
    parser.add_argument("--fmin", type=float, default=80.0, help="峰值检测最低频率 (Hz)")
    parser.add_argument("--fmax", type=float, default=20000.0, help="峰值检测最高频率 (Hz)")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.demo:
        # 打印可用设备，方便用户用 --device 手动指定
        print("可用输入设备（注意带 input 标记的）：")
        print(sd.query_devices(), flush=True)
        print(f"当前默认输入设备索引: {sd.default.device[0]}\n")
        print("若想换设备，关掉窗口后 用  --device <索引>  重新运行\n")

    source = DemoSynth(args.rate, args.block) if args.demo else AudioStream(
        args.rate, args.block, device=args.device)

    app = SpectrumAnalyzer(source, sample_rate=args.rate, block_size=args.block,
                           plot_blocks=args.blocks, fmin=args.fmin, fmax=args.fmax)
    print("频谱分析仪已启动，按窗口右上角 × 或 Ctrl+C 退出。")
    app.run()


if __name__ == "__main__":
    main()
