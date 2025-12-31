"""
训练 DLinear 模型用于时间序列预测
"""
import os
import time
import random
import argparse

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

from data_provider.data_loader_emb import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
from 独立Dlinear import DLinear
from utils.metrics import MSE, MAE, metric


class DLinearConfig:
    """DLinear 配置类"""
    def __init__(self, seq_len=96, pred_len=96, enc_in=7, moving_avg=25):
        self.task_name = 'long_term_forecast'
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.moving_avg = moving_avg
        self.num_class = 10  # 仅分类任务需要，预测任务不使用


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda", help="")
    parser.add_argument("--data_path", type=str, default="ETTh1", help="data path")
    parser.add_argument("--num_nodes", type=int, default=7, help="number of nodes")
    parser.add_argument("--seq_len", type=int, default=96, help="seq_len")
    parser.add_argument("--pred_len", type=int, default=96, help="out_len")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay rate")
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=150, help="")
    parser.add_argument("--seed", type=int, default=2024, help="random seed")
    parser.add_argument("--es_patience", type=int, default=25, help="early stopping patience")
    parser.add_argument("--save", type=str, default="./logs_dlinear/", help="save path")
    parser.add_argument("--embed_version", type=str, default="original", help="嵌入版本（本模型不使用，但保持兼容）")
    parser.add_argument("--moving_avg", type=int, default=25, help="DLinear moving average kernel size")
    parser.add_argument("--individual", action="store_true", help="use individual model for each variate")
    return parser.parse_args()


class trainer:
    def __init__(
        self,
        scaler,
        num_nodes,
        seq_len,
        pred_len,
        moving_avg,
        individual,
        lrate,
        wdecay,
        device,
        epochs,
    ):
        config = DLinearConfig(
            seq_len=seq_len,
            pred_len=pred_len,
            enc_in=num_nodes,
            moving_avg=moving_avg,
        )
        self.model = DLinear(config, individual=individual).to(device)
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
        print(
            "The number of trainable parameters: {}".format(
                self.model.count_trainable_params() if hasattr(self.model, 'count_trainable_params') 
                else sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            )
        )
        print("The number of parameters: {}".format(
            self.model.param_num() if hasattr(self.model, 'param_num')
            else sum(p.numel() for p in self.model.parameters())
        ))

    def train(self, input, mark, embeddings, real):
        self.model.train()
        self.optimizer.zero_grad()
        # 确保输入是 float32
        input = input.float()
        real = real.float()
        predict = self.model(input)
        loss = self.loss(predict, real)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = self.MAE(predict, real)
        return loss.item(), mae.item()

    def eval(self, input, mark, embeddings, real_val):
        self.model.eval()
        with torch.no_grad():
            # 确保输入是 float32
            input = input.float()
            real_val = real_val.float()
            predict = self.model(input)
        loss = self.loss(predict, real_val)
        mae = self.MAE(predict, real_val)
        return loss.item(), mae.item()


def load_data(args):
    data_map = {
        "ETTh1": Dataset_ETT_hour,
        "ETTh2": Dataset_ETT_hour,
        "ETTm1": Dataset_ETT_minute,
        "ETTm2": Dataset_ETT_minute,
    }
    data_class = data_map.get(args.data_path, Dataset_Custom)
    train_set = data_class(
        flag="train",
        scale=True,
        size=[args.seq_len, 0, args.pred_len],
        data_path=args.data_path,
        embed_version=args.embed_version,
    )
    val_set = data_class(
        flag="val",
        scale=True,
        size=[args.seq_len, 0, args.pred_len],
        data_path=args.data_path,
        embed_version=args.embed_version,
    )
    test_set = data_class(
        flag="test",
        scale=True,
        size=[args.seq_len, 0, args.pred_len],
        data_path=args.data_path,
        embed_version=args.embed_version,
    )

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


# 全局日志文件句柄
LOG_F = None


def log(*args, **kwargs):
    # 从 kwargs 中提取 flush，避免重复传递
    flush_val = kwargs.pop('flush', False)
    print(*args, **kwargs, flush=flush_val)
    if LOG_F is not None:
        print(*args, **kwargs, file=LOG_F, flush=True)


def main():
    global LOG_F
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
        f"dlinear_i{args.seq_len}_o{args.pred_len}_ma{args.moving_avg}_"
        f"ind{args.individual}_lr{args.learning_rate}_bs{args.batch_size}_seed{args.seed}/",
    )
    os.makedirs(save_dir, exist_ok=True)

    # 打开日志文件
    log_path = os.path.join(save_dir, "train.log")
    LOG_F = open(log_path, "w")

    try:
        his_loss = []
        val_time = []
        train_time = []
        log(args)

        engine = trainer(
            scaler=scaler,
            num_nodes=args.num_nodes,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            moving_avg=args.moving_avg,
            individual=args.individual,
            lrate=args.learning_rate,
            wdecay=args.weight_decay,
            device=device,
            epochs=args.epochs,
        )

        log("Start training DLinear model...", flush=True)

        for epoch in range(1, args.epochs + 1):
            t1 = time.time()
            train_loss = []
            train_mae = []

            for batch in train_loader:
                if len(batch) == 5:
                    x, y, x_mark, y_mark, embeddings = batch
                else:
                    x, y, x_mark, y_mark = batch
                    embeddings = None

                trainx = torch.Tensor(x).to(device)
                trainy = torch.Tensor(y).to(device)
                trainx_mark = torch.Tensor(x_mark).to(device) if x_mark is not None else None
                train_embedding = torch.Tensor(embeddings).to(device) if embeddings is not None else None

                metrics = engine.train(trainx, trainx_mark, train_embedding, trainy)
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
                if len(batch) == 5:
                    x, y, x_mark, y_mark, embeddings = batch
                else:
                    x, y, x_mark, y_mark = batch
                    embeddings = None

                valx = torch.Tensor(x).to(device)
                valy = torch.Tensor(y).to(device)
                valx_mark = torch.Tensor(x_mark).to(device) if x_mark is not None else None
                val_embedding = torch.Tensor(embeddings).to(device) if embeddings is not None else None

                metrics = engine.eval(valx, valx_mark, val_embedding, valy)
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

                # 在验证集提升时跑一遍测试集
                test_outputs = []
                test_y_all = []
                for batch in test_loader:
                    if len(batch) == 5:
                        x, y, x_mark, y_mark, embeddings = batch
                    else:
                        x, y, x_mark, y_mark = batch
                        embeddings = None

                    testx = torch.Tensor(x).to(device).float()
                    testy = torch.Tensor(y).to(device).float()
                    testx_mark = torch.Tensor(x_mark).to(device).float() if x_mark is not None else None
                    test_embedding = torch.Tensor(embeddings).to(device).float() if embeddings is not None else None

                    with torch.no_grad():
                        preds = engine.model(testx)
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

