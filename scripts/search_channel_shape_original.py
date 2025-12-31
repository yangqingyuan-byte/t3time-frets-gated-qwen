import subprocess
import os

# 配置
SEEDS = list(range(2020, 2041))
CHANNELS = [32, 64, 96, 128, 256]
DATA_PATH = "ETTh1"
EPOCHS = 100
EMBED_VERSION = "qwen3_0.6b"

print(f"开始对 T3Time_Wavelet_Gated_Shape_Qwen 进行 Channel 寻优...")
print(f"种子范围: 2020-2040, 通道范围: {CHANNELS}")

for channel in CHANNELS:
    for seed in SEEDS:
        print(f"\n[运行中] Channel={channel}, Seed={seed}")
        cmd = [
            "python", "train_wavelet_gated_shape_qwen.py",
            "--data_path", DATA_PATH,
            "--channel", str(channel),
            "--seed", str(seed),
            "--epochs", str(EPOCHS),
            "--batch_size", "16",
            "--learning_rate", "1e-4",
            "--embed_version", EMBED_VERSION,
            "--shape_lambda", "0.1"
        ]
        try:
            subprocess.run(cmd, check=True)
        except Exception as e:
            print(f"运行时出错 (Channel {channel}, Seed {seed}): {e}")

