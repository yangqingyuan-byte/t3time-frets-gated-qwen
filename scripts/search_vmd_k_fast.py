"""
快速扫 vmd_k 超参的小脚本

用途：
    在固定其它超参的前提下，仅调节 vmd_k（VMD 模态数），
    例如在 ETTh1 i96→o96 上从 5 尝试到 8，观察 Test MSE 的变化。

用法示例（与你当前实验一致的默认参数）：

    python scripts/search_vmd_k_fast.py \
      --data_path ETTh1 \
      --seq_len 96 \
      --pred_len 96 \
      --num_nodes 7 \
      --batch_size 16 \
      --learning_rate 1e-4 \
      --dropout_n 0.3 \
      --channel 32 \
      --epochs 80 \
      --es_patience 15 \
      --embed_version original \
      --vmd_k_min 5 \
      --vmd_k_max 8 \
      --vmd_alpha 2000 \
      --vmd_root ./vmd_cache \
      --seed 2024

脚本会对每一个 vmd_k 启动一次 train_fft_vmd.py，日志仍然写入
    logs_fft_vmd/ETTh1/fft_vmd_i96_o96_c32_k{K}_a{alpha}_...
你可以之后用 grep 查看对应的 Test MSE。
"""

import argparse
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    # 数据与任务设置
    parser.add_argument("--data_path", type=str, default="ETTh1")
    parser.add_argument("--num_nodes", type=int, default=7)
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--pred_len", type=int, default=96)
    # 训练相关
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--dropout_n", type=float, default=0.3)
    parser.add_argument("--channel", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--es_patience", type=int, default=15)
    parser.add_argument("--embed_version", type=str, default="original")
    parser.add_argument("--seed", type=int, default=2024)
    # VMD 相关
    parser.add_argument("--vmd_k_min", type=int, default=5, help="vmd_k 起始值（包含）")
    parser.add_argument("--vmd_k_max", type=int, default=8, help="vmd_k 结束值（包含）")
    parser.add_argument("--vmd_alpha", type=float, default=2000.0)
    parser.add_argument("--vmd_root", type=str, default="./vmd_cache")
    # 其它
    parser.add_argument(
        "--device", type=str, default="cuda", help="传给 train_fft_vmd.py 的 device"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print(
        f"快速 vmd_k 搜索：k 从 {args.vmd_k_min} 到 {args.vmd_k_max}，"
        f"数据集={args.data_path}, i{args.seq_len}->o{args.pred_len}"
    )
    print("=" * 60)
    sys.stdout.flush()

    for k in range(args.vmd_k_min, args.vmd_k_max + 1):
        print("\n" + "-" * 60)
        print(f"开始运行 vmd_k = {k}")
        print("-" * 60)
        sys.stdout.flush()

        cmd = [
            sys.executable,
            "train_fft_vmd.py",
            "--device",
            args.device,
            "--data_path",
            args.data_path,
            "--seq_len",
            str(args.seq_len),
            "--pred_len",
            str(args.pred_len),
            "--num_nodes",
            str(args.num_nodes),
            "--batch_size",
            str(args.batch_size),
            "--learning_rate",
            str(args.learning_rate),
            "--dropout_n",
            str(args.dropout_n),
            "--channel",
            str(args.channel),
            "--epochs",
            str(args.epochs),
            "--es_patience",
            str(args.es_patience),
            "--embed_version",
            args.embed_version,
            "--vmd_k",
            str(k),
            "--vmd_alpha",
            str(args.vmd_alpha),
            "--vmd_root",
            args.vmd_root,
            "--seed",
            str(args.seed),
        ]

        print("运行命令：", " ".join(cmd))
        sys.stdout.flush()

        try:
            # 直接把子进程输出打印到当前终端，便于实时观察 Test MSE
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[vmd_k={k}] 运行失败，returncode={e.returncode}")
            # 出错时继续下一个 k，而不是立即退出
            continue

    print("\n" + "=" * 60)
    print("vmd_k 快速搜索结束。可以在 logs_fft_vmd/ 下 grep 'Test MSE' 做对比。")
    print("=" * 60)


if __name__ == "__main__":
    main()


