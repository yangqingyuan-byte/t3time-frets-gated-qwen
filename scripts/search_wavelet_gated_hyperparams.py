import subprocess
import os

# 配置搜索空间
SEEDS = list(range(2020, 2041))  # 2020 到 2040
LRS = [1e-4, 5e-5]
DROPOUTS = [0.2, 0.3]
CHANNELS = [32, 64]

# 固定参数
DATA_PATH = "ETTh1"
SEQ_LEN = 96
PRED_LEN = 96
BATCH_SIZE = 16
WAVELET = "db4"
EMBED_VERSION = "qwen3_0.6b"

def run_experiment(seed, lr, dropout, channel):
    cmd = [
        "python", "train_wavelet_gated_qwen.py",
        "--data_path", DATA_PATH,
        "--seq_len", str(SEQ_LEN),
        "--pred_len", str(PRED_LEN),
        "--batch_size", str(BATCH_SIZE),
        "--learning_rate", str(lr),
        "--dropout_n", str(dropout),
        "--channel", str(channel),
        "--seed", str(seed),
        "--wavelet", WAVELET,
        "--embed_version", EMBED_VERSION,
        "--use_cross_attention",
        "--epochs", "100",
        "--es_patience", "15"
    ]
    
    print(f"\n>>> 正在运行: Seed={seed}, LR={lr}, Dropout={dropout}, Channel={channel}")
    try:
        # 运行训练脚本
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"!!! 实验失败: {e}")

def main():
    # 确保在根目录运行
    if not os.path.exists("train_wavelet_gated_qwen.py"):
        print("错误: 请在项目根目录下运行此脚本")
        return

    count = 0
    total = len(SEEDS) * len(LRS) * len(DROPOUTS) * len(CHANNELS)
    
    for seed in SEEDS:
        for lr in LRS:
            for dropout in DROPOUTS:
                for channel in CHANNELS:
                    count += 1
                    print(f"\n进度: {count}/{total}")
                    run_experiment(seed, lr, dropout, channel)

if __name__ == "__main__":
    main()

