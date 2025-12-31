import subprocess
import os
import time

# 设置使用的显卡
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# --- 实验配置 ---
MODEL_SCRIPT = "train.py"
SEEDS = list(range(2020, 2041))           # 2020 到 2040
DATA_PATH = "ETTh1"
SEQ_LEN = 96
PRED_LEN = 96
EPOCHS = 150
ES_PATIENCE = 25
EMBED_VERSION = "original"

# --- 超参数寻优空间 ---
# 基于最佳结果: channel=256, batch_size=256, learning_rate=0.0001, dropout_n=0.4
LEARNING_RATES = [1e-4]      # 围绕最佳值 1e-4 进行寻优
DROPOUTS = [0.4]                # 围绕最佳值 0.4 进行寻优
CHANNELS = [256]                # 围绕最佳值 256 进行寻优
BATCH_SIZES = [256]                  # 围绕最佳值 256 进行寻优

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

    total_experiments = len(LEARNING_RATES) * len(DROPOUTS) * len(CHANNELS) * len(BATCH_SIZES) * len(SEEDS)
    count = 0
    print(f"🚀 开始 T3Time 原始模型参数寻优")
    print(f"目标模型: T3Time (Original)")
    print(f"数据集: {DATA_PATH}, 预测长度: {PRED_LEN}")
    print(f"预计总实验数: {total_experiments}")
    print("=" * 60)

    for lr in LEARNING_RATES:
        for dropout in DROPOUTS:
            for channel in CHANNELS:
                for batch_size in BATCH_SIZES:
                    for seed in SEEDS:
                        count += 1
                        print(f"\n>>> 进度: {count}/{total_experiments}")
                        print(f"    LR: {lr}, Dropout: {dropout}, Channel: {channel}, Batch: {batch_size}, Seed: {seed}")
                        
                        cmd = [
                            "python", MODEL_SCRIPT,
                            "--data_path", DATA_PATH,
                            "--seq_len", str(SEQ_LEN),
                            "--pred_len", str(PRED_LEN),
                            "--num_nodes", "7",
                            "--channel", str(channel),
                            "--batch_size", str(batch_size),
                            "--learning_rate", str(lr),
                            "--dropout_n", str(dropout),
                            "--epochs", str(EPOCHS),
                            "--es_patience", str(ES_PATIENCE),
                            "--embed_version", EMBED_VERSION,
                            "--seed", str(seed)
                        ]
                        
                        run_cmd(cmd)

    print("\n" + "=" * 60)
    print("✅ T3Time 原始模型所有寻优实验已完成！")
    print(f"📊 结果已记录至: ./experiment_results.log")

if __name__ == "__main__":
    main()

