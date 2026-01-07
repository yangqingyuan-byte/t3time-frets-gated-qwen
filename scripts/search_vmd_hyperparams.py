"""
简单的 VMD 超参搜索脚本

搜索空间（可根据需要自行修改）：
  - vmd_k:      [3, 4, 5]
  - vmd_alpha:  [1000, 1500, 2000]
  - channel:    [32, 48]

脚本做的事情：
  1）遍历上述组合；
  2）为每组组合调用一次 train_vmd.py（通过子进程）；
  3）记录超参组合、日志目录路径；
  4）将所有组合记录到一个 CSV，方便后续用分析脚本统一查看。

注意：
  - 为了节省时间，默认 epochs=20, es_patience=10，与我们之前快速实验设置一致；
  - 如果你希望跑更长时间，可以在下方 DEFAULT_EPOCHS / DEFAULT_ES_PATIENCE 修改。
"""

import itertools
import os
import subprocess
import time
import csv
from datetime import datetime


# ---------------- 用户可调整的超参搜索空间 ----------------
# 适合“跑一晚上”的稍大搜索空间，如果觉得太多可以自行缩减
VMD_K_LIST = [3, 4, 5, 6]
VMD_ALPHA_LIST = [500, 1000, 1500, 2000, 2500]
CHANNEL_LIST = [32, 48, 64]

DATA_PATH = "ETTh1"
SEQ_LEN = 96
PRED_LEN = 96
NUM_NODES = 7
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
DROPOUT_N = 0.1
EPOCHS = 20
ES_PATIENCE = 10
EMBED_VERSION = "original"
VMD_ROOT = "./vmd_cache"

DEVICE = "cuda"
SAVE_ROOT = "./logs_vmd/"

# 结果汇总 CSV
RESULT_CSV = "./logs_vmd/vmd_hparam_search_{}.csv".format(
    datetime.now().strftime("%Y%m%d-%H%M%S")
)


def run_one_setting(vmd_k: int, vmd_alpha: float, channel: int) -> str:
    """
    跑一组超参，返回本次实验的日志目录路径（train_vmd.py 自己会创建目录并保存 best_model）。
    """
    cmd = [
        "python",
        "train_vmd.py",
        "--device",
        DEVICE,
        "--data_path",
        DATA_PATH,
        "--seq_len",
        str(SEQ_LEN),
        "--pred_len",
        str(PRED_LEN),
        "--channel",
        str(channel),
        "--num_nodes",
        str(NUM_NODES),
        "--batch_size",
        str(BATCH_SIZE),
        "--learning_rate",
        str(LEARNING_RATE),
        "--dropout_n",
        str(DROPOUT_N),
        "--epochs",
        str(EPOCHS),
        "--embed_version",
        EMBED_VERSION,
        "--vmd_k",
        str(vmd_k),
        "--vmd_alpha",
        str(vmd_alpha),
        "--vmd_root",
        VMD_ROOT,
        "--es_patience",
        str(ES_PATIENCE),
    ]

    print("\n" + "=" * 80)
    print(f"Running VMD search: k={vmd_k}, alpha={vmd_alpha}, channel={channel}")
    print("Command:", " ".join(cmd))
    print("=" * 80, flush=True)

    start_time = time.time()
    proc = subprocess.run(cmd)
    dur = time.time() - start_time
    print(f"Finished in {dur:.1f}s, return code={proc.returncode}")

    # train_vmd.py 的保存路径规则：
    # save_dir = os.path.join(
    #     args.save,
    #     args.data_path,
    #     f"vmd_k{args.vmd_k}_a{int(args.vmd_alpha)}_i{args.seq_len}_o{args.pred_len}_"
    #     f"c{args.channel}_el{args.e_layer}_dl{args.d_layer}_lr{args.learning_rate}_"
    #     f"dn{args.dropout_n}_bs{args.batch_size}_seed{args.seed}/",
    # )
    save_dir = os.path.join(
        SAVE_ROOT,
        DATA_PATH,
        f"vmd_k{vmd_k}_a{int(vmd_alpha)}_i{SEQ_LEN}_o{PRED_LEN}_"
        f"c{channel}_el1_dl1_lr{LEARNING_RATE}_"
        f"dn{DROPOUT_N}_bs{BATCH_SIZE}_seed2024/",
    )
    return save_dir


def main():
    os.makedirs(SAVE_ROOT, exist_ok=True)

    combos = list(itertools.product(VMD_K_LIST, VMD_ALPHA_LIST, CHANNEL_LIST))
    print(f"Total combos: {len(combos)}")
    print("Results CSV will be saved to:", RESULT_CSV)

    with open(RESULT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "vmd_k",
                "vmd_alpha",
                "channel",
                "log_dir",
                "notes",
            ]
        )

        for vmd_k, vmd_alpha, channel in combos:
            log_dir = run_one_setting(vmd_k, vmd_alpha, channel)
            writer.writerow(
                [
                    vmd_k,
                    vmd_alpha,
                    channel,
                    log_dir,
                    "查看对应日志中的 Test MSE/MAE 以比较表现",
                ]
            )
            f.flush()

    print("\nSearch finished. Summary CSV:", RESULT_CSV)


if __name__ == "__main__":
    main()


