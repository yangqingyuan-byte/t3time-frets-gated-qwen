"""
测试原始 T3Time（仅 FFT）的过拟合能力
用于对比基准，判断问题是否在 VMD 融合方式
"""
import os
import time
import random
import argparse

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

from data_provider.data_loader_emb import Dataset_ETT_hour
from models.T3Time import TriModal
from utils.metrics import MSE, MAE


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda", help="")
    parser.add_argument("--data_path", type=str, default="ETTh1", help="data path")
    parser.add_argument("--num_nodes", type=int, default=7, help="number of nodes")
    parser.add_argument("--seq_len", type=int, default=96, help="seq_len")
    parser.add_argument("--pred_len", type=int, default=96, help="out_len")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--dropout_n", type=float, default=0.2, help="dropout rate")
    parser.add_argument("--d_llm", type=int, default=768, help="hidden dimensions")
    parser.add_argument("--e_layer", type=int, default=1, help="layers of transformer encoder")
    parser.add_argument("--d_layer", type=int, default=1, help="layers of transformer decoder")
    parser.add_argument("--head", type=int, default=8, help="heads of attention")
    parser.add_argument("--channel", type=int, default=32, help="number of features")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="weight decay rate")
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=1000, help="过拟合测试：1000 epochs")
    parser.add_argument("--seed", type=int, default=2024, help="random seed")
    parser.add_argument("--embed_version", type=str, default="original", help="嵌入版本")
    return parser.parse_args()


def seed_it(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)


class trainer:
    def __init__(
        self,
        channel,
        num_nodes,
        seq_len,
        pred_len,
        dropout_n,
        d_llm,
        e_layer,
        d_layer,
        head,
        lrate,
        wdecay,
        device,
    ):
        self.model = TriModal(
            device=device,
            channel=channel,
            num_nodes=num_nodes,
            seq_len=seq_len,
            pred_len=pred_len,
            dropout_n=dropout_n,
            d_llm=d_llm,
            e_layer=e_layer,
            d_layer=d_layer,
            head=head,
        )
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=lrate, weight_decay=wdecay
        )
        self.loss = MSE
        self.MAE = MAE
        self.clip = 5
        print(
            "The number of trainable parameters: {}".format(
                self.model.count_trainable_params()
            )
        )
        print("The number of parameters: {}".format(self.model.param_num()))

    def train(self, input, mark, embeddings, real):
        self.model.train()
        self.optimizer.zero_grad()
        input = input.float()
        real = real.float()
        predict = self.model(input, mark, embeddings)
        loss = self.loss(predict, real)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = self.MAE(predict, real)
        return loss.item(), mae.item()


def load_data(args):
    train_set = Dataset_ETT_hour(
        flag="train",
        scale=True,
        size=[args.seq_len, 0, args.pred_len],
        data_path=args.data_path,
        num_nodes=args.num_nodes,
        embed_version=args.embed_version,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.num_workers,
    )

    return train_set, train_loader


def main():
    args = parse_args()
    train_set, train_loader = load_data(args)

    seed_it(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("原始 T3Time（仅 FFT）过拟合测试")
    print("预期：Training Loss 必须降到几乎为 0 (MSE < 0.00001)")
    print("=" * 60)

    engine = trainer(
        channel=args.channel,
        num_nodes=args.num_nodes,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        dropout_n=args.dropout_n,
        d_llm=args.d_llm,
        e_layer=args.e_layer,
        d_layer=args.d_layer,
        head=args.head,
        lrate=args.learning_rate,
        wdecay=args.weight_decay,
        device=device,
    )

    # 只取第一个 batch
    first_batch = next(iter(train_loader))
    x, y, x_mark, y_mark, embeddings = first_batch

    trainx = torch.Tensor(x).to(device)
    trainy = torch.Tensor(y).to(device)
    trainx_mark = torch.Tensor(x_mark).to(device)
    train_embedding = torch.Tensor(embeddings).to(device)

    print(f"过拟合测试数据形状: input={trainx.shape}, target={trainy.shape}")
    print(f"开始训练 {args.epochs} 个 epoch...\n")

    # 训练 1000 个 epoch
    for epoch in range(1, args.epochs + 1):
        metrics = engine.train(trainx, trainx_mark, train_embedding, trainy)
        train_loss, train_mae = metrics

        # 每 10 个 epoch 打印一次，最后 100 个 epoch 每 1 个打印一次
        if epoch % 10 == 0 or epoch > args.epochs - 100:
            print(
                f"Epoch: {epoch:04d}, Train Loss: {train_loss:.8f}, Train MAE: {train_mae:.8f}"
            )

            # 如果 loss 降到足够低，提前结束
            if train_loss < 0.00001:
                print(f"✓ 成功！Loss 已降到 {train_loss:.8f} < 0.00001")
                print("诊断：模型能够过拟合，说明代码结构正确，梯度流正常")
                break

    # 最终评估
    print("=" * 60)
    print("过拟合测试最终结果:")
    print(f"最终 Train Loss: {train_loss:.8f}")
    print(f"最终 Train MAE: {train_mae:.8f}")

    if train_loss < 0.00001:
        print("✓ 诊断：模型能够过拟合到接近 0，代码结构正确")
    else:
        print("✗ 诊断：模型无法过拟合，可能原因：")
        print("  1. 模型容量太小")
        print("  2. 归一化没做好")
        print("  3. 代码逻辑错误")
    print("=" * 60)


if __name__ == "__main__":
    main()

