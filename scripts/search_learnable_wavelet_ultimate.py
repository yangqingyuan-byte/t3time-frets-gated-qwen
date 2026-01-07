import subprocess
import os
import time

# 设置使用的显卡
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# --- 实验配置 ---
MODEL_SCRIPT = "train_learnable_wavelet_gated_shape_qwen.py"
SEEDS = list(range(2020, 2041))           # 2020 到 2040
CHANNELS = [256]        # 全通道覆盖
DATA_PATH = "ETTh1"
SEQ_LEN = 96
PRED_LEN = 96
EPOCHS = 200                             # 按照最高标准设定为 200
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EMBED_VERSION = "qwen3_0.6b"
SHAPE_LAMBDA = 0.1
LEVELS = 3                               # 小波分解层级

def run_cmd(cmd):
    print(f"\n[执行中] {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"!!! 实验失败: {e}")

def main():
    if not os.path.exists(MODEL_SCRIPT):
        print(f"错误: 找不到训练脚本 {MODEL_SCRIPT}")
        return

    total_experiments = len(CHANNELS) * len(SEEDS)
    count = 0
    print(f"🚀 开始可学习小波版模型 (Learnable Wavelet) 终极参数寻优")
    print(f"目标模型: T3Time_Learnable_Wavelet_Gated_Shape_Qwen")
    print(f"预计总实验数: {total_experiments}")
    print("=" * 60)

    for channel in CHANNELS:
        for seed in SEEDS:
            count += 1
            print(f"\n>>> 进度: {count}/{total_experiments} | Channel: {channel} | Seed: {seed}")
            
            cmd = [
                "python", MODEL_SCRIPT,
                "--data_path", DATA_PATH,
                "--seq_len", str(SEQ_LEN),
                "--pred_len", str(PRED_LEN),
                "--channel", str(channel),
                "--seed", str(seed),
                "--epochs", str(EPOCHS),
                "--batch_size", str(BATCH_SIZE),
                "--learning_rate", str(LEARNING_RATE),
                "--embed_version", EMBED_VERSION,
                "--shape_lambda", str(SHAPE_LAMBDA),
                "--levels", str(LEVELS)
            ]
            
            run_cmd(cmd)

    print("\n" + "=" * 60)
    print("✅ 可学习小波版所有寻优实验已完成！")

if __name__ == "__main__":
    main()

