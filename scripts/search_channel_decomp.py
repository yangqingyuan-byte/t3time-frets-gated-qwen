import subprocess
import os

SEEDS = list(range(2020, 2041))
CHANNELS = [32, 64, 96, 128, 256]
DATA_PATH = "ETTh1"
EPOCHS = 100

for channel in CHANNELS:
    for seed in SEEDS:
        print(f"\n>>> 正在运行: Channel={channel}, Seed={seed}")
        cmd = [
            "python", "train_wavelet_decomp_gated_qwen.py",
            "--data_path", DATA_PATH,
            "--channel", str(channel),
            "--seed", str(seed),
            "--epochs", str(EPOCHS),
            "--batch_size", "16",
            "--learning_rate", "1e-4"
        ]
        try:
            subprocess.run(cmd, check=True)
        except Exception as e:
            print(f"Error at Channel {channel}, Seed {seed}: {e}")

