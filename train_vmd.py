"""
训练基于 VMD 分支的 T3Time 变体 TriModalVMD。

不会修改原有 train.py / train_wavelet.py，只新增一个独立训练脚本。
"""
import os
import time
import random
import argparse

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

from data_provider.data_loader_vmd import Dataset_ETT_hour_VMD
from data_provider.data_loader_emb import Dataset_ETT_minute, Dataset_Custom
from models.T3Time_VMD import TriModalVMD
from utils.metrics import MSE, MAE, metric


# 简单的日志函数：同时输出到终端和当前实验目录下的 train.log
LOG_F = None


def log(*args, **kwargs):
    """
    打印到标准输出，并可选写入全局日志文件 LOG_F。
    """
    print(*args, **kwargs)
    if LOG_F is not None:
        print(*args, **kwargs, file=LOG_F, flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda", help="")
    parser.add_argument("--data_path", type=str, default="ETTh1", help="data path")
    parser.add_argument("--channel", type=int, default=32, help="number of features")
    parser.add_argument("--num_nodes", type=int, default=7, help="number of nodes")
    parser.add_argument("--seq_len", type=int, default=96, help="seq_len")
    parser.add_argument("--pred_len", type=int, default=96, help="out_len")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--dropout_n", type=float, default=0.2, help="dropout rate")
    parser.add_argument("--d_llm", type=int, default=768, help="hidden dimensions")
    parser.add_argument("--e_layer", type=int, default=1, help="layers of transformer encoder")
    parser.add_argument("--d_layer", type=int, default=1, help="layers of transformer decoder")
    parser.add_argument("--head", type=int, default=8, help="heads of attention")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="weight decay rate")
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--model_name", type=str, default="gpt2", help="llm")
    parser.add_argument("--epochs", type=int, default=150, help="")
    parser.add_argument("--seed", type=int, default=2024, help="random seed")
    parser.add_argument(
        "--es_patience", type=int, default=25, help="early stopping patience"
    )
    parser.add_argument(
        "--save", type=str, default="./logs_vmd/", help="save path for VMD model"
    )
    parser.add_argument(
        "--embed_version",
        type=str,
        default="original",
        help="嵌入版本标识（如 'original', 'wavelet', 'gpt2'）",
    )
    # VMD 相关
    parser.add_argument("--vmd_k", type=int, default=4, help="VMD 模态数 K")
    parser.add_argument(
        "--vmd_alpha", type=float, default=2000.0, help="VMD 带宽约束 alpha"
    )
    parser.add_argument(
        "--vmd_root",
        type=str,
        default="./vmd_cache",
        help="VMD 缓存根目录（按数据集/模态数划分子目录）",
    )
    return parser.parse_args()


class trainer:
    def __init__(
        self,
        scaler,
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
        epochs,
        vmd_k,
    ):
        self.model = TriModalVMD(
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
            vmd_modes=vmd_k,
        ).to(device)
        self.epochs = epochs
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=lrate, weight_decay=wdecay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=min(epochs, 50), eta_min=1e-6
        )
        self.loss = MSE
        self.MAE = MAE
        self.clip = 5
        log(
            "The number of trainable parameters: {}".format(
                self.model.count_trainable_params()
            )
        )
        log("The number of parameters: {}".format(self.model.param_num()))

    def train(self, input, mark, embeddings, x_modes, real):
        self.model.train()
        self.optimizer.zero_grad()
        predict = self.model(input, mark, embeddings, x_modes)
        loss = self.loss(predict, real)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = self.MAE(predict, real)
        return loss.item(), mae.item()

    def eval(self, input, mark, embeddings, x_modes, real_val):
        self.model.eval()
        with torch.no_grad():
            predict = self.model(input, mark, embeddings, x_modes)
        loss = self.loss(predict, real_val)
        mae = self.MAE(predict, real_val)
        return loss.item(), mae.item()


def load_data(args):
    """
    为 VMD 版本构建数据集：
      - ETTh1/ETTh2 使用 Dataset_ETT_hour_VMD（带 x_modes）
      - 其他数据集暂时沿用原 Dataset_ETT_minute / Dataset_Custom（不带 VMD）
    """
    if args.data_path in ["ETTh1", "ETTh2"]:
        data_class = Dataset_ETT_hour_VMD
    else:
        # 目前只做 ETTh1/ETTh2 的 VMD 示例，其他数据保持原样（不含 x_modes）
        data_map = {
            "ETTm1": Dataset_ETT_minute,
            "ETTm2": Dataset_ETT_minute,
        }
        data_class = data_map.get(args.data_path, Dataset_Custom)

    if data_class is Dataset_ETT_hour_VMD:
        common_kwargs = dict(
            scale=True,
            size=[args.seq_len, 0, args.pred_len],
            data_path=args.data_path,
            num_nodes=args.num_nodes,
            embed_version=args.embed_version,
            vmd_k=args.vmd_k,
            vmd_alpha=args.vmd_alpha,
            vmd_root=args.vmd_root,
        )
        train_set = data_class(flag="train", **common_kwargs)
        val_set = data_class(flag="val", **common_kwargs)
        test_set = data_class(flag="test", **common_kwargs)
    else:
        # 非 ETT hour 情况：暂不支持 VMD，只演示通路（x_modes 用零张量占位）
        common_kwargs = dict(
            scale=True,
            size=[args.seq_len, 0, args.pred_len],
            data_path=args.data_path,
            embed_version=args.embed_version,
        )
        train_set = data_class(flag="train", **common_kwargs)
        val_set = data_class(flag="val", **common_kwargs)
        test_set = data_class(flag="test", **common_kwargs)

    scaler = train_set.scaler

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.num_workers,
    )

    return train_set, val_set, test_set, train_loader, val_loader, test_loader, scaler


def seed_it(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)


def main():
    args = parse_args()
    (
        train_set,
        val_set,
        test_set,
        train_loader,
        val_loader,
        test_loader,
        scaler,
    ) = load_data(args)

    log()
    seed_it(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_val_loss = float("inf")
    test_log = float("inf")
    epochs_since_best_mse = 0

    save_dir = os.path.join(
        args.save,
        args.data_path,
        f"vmd_k{args.vmd_k}_a{int(args.vmd_alpha)}_i{args.seq_len}_o{args.pred_len}_"
        f"c{args.channel}_el{args.e_layer}_dl{args.d_layer}_lr{args.learning_rate}_"
        f"dn{args.dropout_n}_bs{args.batch_size}_seed{args.seed}/",
    )
    os.makedirs(save_dir, exist_ok=True)

    # 准备日志文件
    global LOG_F
    log_path = os.path.join(save_dir, "train.log")
    LOG_F = open(log_path, "w")

    his_loss = []
    val_time = []
    train_time = []
    log(args)

    try:
        engine = trainer(
            scaler=scaler,
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
            epochs=args.epochs,
            vmd_k=args.vmd_k,
        )

        log("Start training VMD model...")

        for epoch in range(1, args.epochs + 1):
            t1 = time.time()
            train_loss = []
            train_mae = []

            for batch in train_loader:
                if len(batch) == 6:
                    x, y, x_mark, y_mark, embeddings, x_modes = batch
                else:
                    # 非 VMD 数据集的占位（当前主要支持 ETTh1/ETTh2）
                    x, y, x_mark, y_mark, embeddings = batch
                    B, L, N = x.shape
                    x_modes = torch.zeros(B, L, N, args.vmd_k)

                trainx = torch.Tensor(x).to(device)
                trainy = torch.Tensor(y).to(device)
                trainx_mark = torch.Tensor(x_mark).to(device)
                train_embedding = torch.Tensor(embeddings).to(device)
                train_x_modes = torch.Tensor(x_modes).to(device)

                metrics = engine.train(
                    trainx, trainx_mark, train_embedding, train_x_modes, trainy
                )
                train_loss.append(metrics[0])
                train_mae.append(metrics[1])

            t2 = time.time()
            log(f"Epoch: {epoch:03d}, Training Time: {t2 - t1:.4f} secs")
            train_time.append(t2 - t1)

            # validation
            val_loss = []
            val_mae = []
            s1 = time.time()

            for batch in val_loader:
                if len(batch) == 6:
                    x, y, x_mark, y_mark, embeddings, x_modes = batch
                else:
                    x, y, x_mark, y_mark, embeddings = batch
                    B, L, N = x.shape
                    x_modes = torch.zeros(B, L, N, args.vmd_k)

                valx = torch.Tensor(x).to(device)
                valy = torch.Tensor(y).to(device)
                valx_mark = torch.Tensor(x_mark).to(device)
                val_embedding = torch.Tensor(embeddings).to(device)
                val_x_modes = torch.Tensor(x_modes).to(device)

                metrics = engine.eval(valx, valx_mark, val_embedding, val_x_modes, valy)
                val_loss.append(metrics[0])
                val_mae.append(metrics[1])

            s2 = time.time()
            log(f"Epoch: {epoch:03d}, Validation Time: {s2 - s1:.4f} secs")
            val_time.append(s2 - s1)

            mtrain_loss = np.mean(train_loss)
            mtrain_mae = np.mean(train_mae)
            mvalid_loss = np.mean(val_loss)
            mvalid_mae = np.mean(val_mae)

            his_loss.append(mvalid_loss)
            log("-----------------------")
            log(
                f"Epoch: {epoch:03d}, Train Loss: {mtrain_loss:.4f}, Train MAE: {mtrain_mae:.4f}"
            )
            log(
                f"Epoch: {epoch:03d}, Valid Loss: {mvalid_loss:.4f}, Valid MAE: {mvalid_mae:.4f}"
            )

            if mvalid_loss < best_val_loss:
                log("###Update best model###")
                best_val_loss = mvalid_loss
                epochs_since_best_mse = 0
                torch.save(
                    engine.model.state_dict(), os.path.join(save_dir, "best_model.pth")
                )

                # 在验证集提升时跑一遍测试集（可选，保持与原脚本风格接近）
                test_outputs = []
                test_y_all = []
                for batch in test_loader:
                    if len(batch) == 6:
                        x, y, x_mark, y_mark, embeddings, x_modes = batch
                    else:
                        x, y, x_mark, y_mark, embeddings = batch
                        B, L, N = x.shape
                        x_modes = torch.zeros(B, L, N, args.vmd_k)

                    testx = torch.Tensor(x).to(device)
                    testy = torch.Tensor(y).to(device)
                    testx_mark = torch.Tensor(x_mark).to(device)
                    test_embedding = torch.Tensor(embeddings).to(device)
                    test_x_modes = torch.Tensor(x_modes).to(device)
                    with torch.no_grad():
                        preds = engine.model(
                            testx, testx_mark, test_embedding, test_x_modes
                        )
                    test_outputs.append(preds)
                    test_y_all.append(testy)

                test_pre = torch.cat(test_outputs, dim=0)
                test_real = torch.cat(test_y_all, dim=0)

                amse = []
                amae = []
                for j in range(args.pred_len):
                    pred = test_pre[:, j, :].to(device)
                    real = test_real[:, j, :].to(device)
                    mse_j, mae_j = metric(pred, real)
                    amse.append(mse_j)
                    amae.append(mae_j)

                test_log = np.mean(amse)
                log(f"Test MSE: {np.mean(amse):.4f}, Test MAE: {np.mean(amae):.4f}")
            else:
                epochs_since_best_mse += 1
                if epochs_since_best_mse >= args.es_patience:
                    log("Early stopping triggered.")
                    break

            # scheduler
            engine.scheduler.step()

        log("Training finished.")
    finally:
        if LOG_F is not None:
            LOG_F.close()


if __name__ == "__main__":
    main()

