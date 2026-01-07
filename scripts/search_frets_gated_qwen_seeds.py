import subprocess
import os

# 设置 PYTHONPATH
os.environ["PYTHONPATH"] = os.getcwd()

# 配置
DATA_PATH = "ETTh1"
PRED_LENS = [96] # 专注于 96 预测长度
SEEDS = list(range(2020, 2071)) # 2020 到 2070

def run_experiment(pred_len, seed):
    cmd = [
        "python", "train_frets_gated_qwen.py",
        "--data_path", DATA_PATH,
        "--seq_len", "96",
        "--pred_len", str(pred_len),
        "--num_nodes", "7",
        "--batch_size", "16",
        "--learning_rate", "1e-4",
        "--channel", "64",
        "--epochs", "100",
        "--es_patience", "10",
        "--seed", str(seed)
    ]
    print(f"\n>>> 正在运行: Pred_Len={pred_len}, Seed={seed}")
    subprocess.run(cmd)

if __name__ == "__main__":
    for pl in PRED_LENS:
        for s in SEEDS:
            run_experiment(pl, s)
