import os
import subprocess
from datetime import datetime

# 配置区域
GPU_ID = "1"
MODEL_TYPE = "T3Time_Learnable_Wavelet_Packet"
DATA_PATH = "ETTh1"
PRED_LENS = [96, 192, 336, 720]
SEEDS = list(range(2020, 2041)) # 2020 到 2040

# 固定超参数 (基于之前的最佳经验)
CHANNEL = 256
BATCH_SIZE = 16
LR = 0.0001
DROPOUT = 0.4
EPOCHS = 200
PATIENCE = 20
WP_LEVEL = 2

def run_experiment():
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
    
    total_runs = len(PRED_LENS) * len(SEEDS)
    current_run = 0

    print(f"开始 {MODEL_TYPE} 自动化寻优实验...")
    print(f"总计任务数: {total_runs}")

    for pred_len in PRED_LENS:
        for seed in SEEDS:
            current_run += 1
            print(f"\n[{current_run}/{total_runs}] 正在运行: pred_len={pred_len}, seed={seed}")
            
            cmd = [
                "python", "train_learnable_wavelet_packet_gated_qwen.py",
                "--data_path", DATA_PATH,
                "--seq_len", "96",
                "--pred_len", str(pred_len),
                "--channel", str(CHANNEL),
                "--batch_size", str(BATCH_SIZE),
                "--learning_rate", str(LR),
                "--dropout_n", str(DROPOUT),
                "--wp_level", str(WP_LEVEL),
                "--epochs", str(EPOCHS),
                "--patience", str(PATIENCE),
                "--seed", str(seed),
                "--embed_version", "qwen3_0.6b"
            ]
            
            try:
                # 运行并实时打印输出
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"实验失败: pred_len={pred_len}, seed={seed}. 错误: {e}")
                continue

    print("\n所有实验任务已完成！结果已记录至 experiment_results.log")

if __name__ == "__main__":
    run_experiment()
