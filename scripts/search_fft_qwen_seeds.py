import subprocess
import os
import time

# 设置使用的显卡
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# --- 实验配置 ---
MODEL_SCRIPT = "train_fft_qwen.py"
SEEDS = list(range(2020, 2051))           # 2020 到 2050
DATA_PATH = "ETTh1"
SEQ_LEN = 96
PRED_LEN = 96
EPOCHS = 150
ES_PATIENCE = 25
BATCH_SIZE = 256
LEARNING_RATE = 1e-4
DROPOUT_N = 0.4
CHANNEL = 256
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

    total_experiments = len(SEEDS)
    count = 0
    print(f"🚀 开始 T3Time_FFT_Qwen 模型种子寻优")
    print(f"目标模型: T3Time_FFT_Qwen (FFT + Qwen3-0.6B)")
    print(f"数据集: {DATA_PATH}, 预测长度: {PRED_LEN}")
    print(f"超参数: channel={CHANNEL}, batch_size={BATCH_SIZE}, lr={LEARNING_RATE}, dropout={DROPOUT_N}")
    print(f"预计总实验数: {total_experiments} (种子: {SEEDS[0]} - {SEEDS[-1]})")
    print("=" * 60)

    for seed in SEEDS:
        count += 1
        print(f"\n>>> 进度: {count}/{total_experiments} | Seed: {seed}")
        
        cmd = [
            "python", MODEL_SCRIPT,
            "--data_path", DATA_PATH,
            "--seq_len", str(SEQ_LEN),
            "--pred_len", str(PRED_LEN),
            "--num_nodes", "7",
            "--channel", str(CHANNEL),
            "--batch_size", str(BATCH_SIZE),
            "--learning_rate", str(LEARNING_RATE),
            "--dropout_n", str(DROPOUT_N),
            "--epochs", str(EPOCHS),
            "--es_patience", str(ES_PATIENCE),
            "--embed_version", EMBED_VERSION,
            "--seed", str(seed)
        ]
        
        run_cmd(cmd)

    print("\n" + "=" * 60)
    print("✅ T3Time_FFT_Qwen 所有种子寻优实验已完成！")
    print(f"📊 结果已记录至: ./experiment_results.log")

if __name__ == "__main__":
    main()

