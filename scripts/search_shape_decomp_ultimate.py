import subprocess
import os
import time

# --- 实验配置 ---
MODEL_SCRIPT = "train_wavelet_gated_shape_decomp_qwen.py"
SEEDS = list(range(2020, 2041))  # 2020 到 2040
CHANNELS = [32,64,96,128,256]             # 既然 64 效果好，我们重点看 64 并尝试 128
DATA_PATH = "ETTh1"
SEQ_LEN = 96
PRED_LEN = 96
EPOCHS = 200                     # 按照最强组合的要求设定为 150
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EMBED_VERSION = "qwen3_0.6b"

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
    print(f"🚀 开始终极参数寻优计划")
    print(f"目标模型: T3Time_Wavelet_Gated_Shape_Decomp_Qwen")
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
                "--shape_lambda", "0.1"
            ]
            
            run_cmd(cmd)

    print("\n" + "=" * 60)
    print("✅ 所有实验已完成！请运行分析脚本查看最佳结果。")

if __name__ == "__main__":
    main()

