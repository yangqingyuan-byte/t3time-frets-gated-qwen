import subprocess
import os
import time

# 显卡设置 (固定使用显卡 1)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# --- 实验配置 ---
MODEL_SCRIPT = "train_fft_qwen_lite.py"
DATA_PATH = "ETTh1"
SEQ_LEN = 96
PRED_LEN = 96
EPOCHS = 150
ES_PATIENCE = 20
EMBED_VERSION = "qwen3_0.6b"

# --- 寻优空间 ---
# 种子范围: 2020-2040
SEEDS = list(range(2020, 2041)) 
# 通道数
CHANNELS = [64, 128, 256, 512]
# 学习率
LEARNING_RATES = [1e-4, 5e-4]
# 批大小
BATCH_SIZES = [128, 256]
# 丢弃率
DROPOUTS = [0.1, 0.2,0.3,0.4]

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

    # 计算总实验数
    total_experiments = len(SEEDS) * len(CHANNELS) * len(LEARNING_RATES) * len(BATCH_SIZES) * len(DROPOUTS)
    count = 0
    
    print(f"🚀 开始 T3Time_FFT_Qwen_Lite 参数寻优")
    print(f"预计总实验数: {total_experiments}")
    print("=" * 60)

    for channel in CHANNELS:
        for lr in LEARNING_RATES:
            for batch_size in BATCH_SIZES:
                for dropout in DROPOUTS:
                    for seed in SEEDS:
                        count += 1
                        print(f"\n>>> 进度: {count}/{total_experiments}")
                        print(f"    Channel: {channel}, LR: {lr}, Batch: {batch_size}, Dropout: {dropout}, Seed: {seed}")
                        
                        cmd = [
                            "python", MODEL_SCRIPT,
                            "--data_path", DATA_PATH,
                            "--seq_len", str(SEQ_LEN),
                            "--pred_len", str(PRED_LEN),
                            "--num_nodes", "7",
                            "--channel", str(channel),
                            "--batch_size", str(batch_size),
                            "--learning_rate", str(lr),
                            "--dropout", str(dropout),
                            "--epochs", str(EPOCHS),
                            "--es_patience", str(ES_PATIENCE),
                            "--embed_version", EMBED_VERSION,
                            "--seed", str(seed)
                        ]
                        
                        run_cmd(cmd)

    print("\n" + "=" * 60)
    print("✅ T3Time_FFT_Qwen_Lite 所有寻优实验已完成！")
    print(f"📊 结果已记录至: ./experiment_results.log")

if __name__ == "__main__":
    main()

