"""
训练基于小波变换的T3Time模型，使用Qwen3-0.6B嵌入
"""
import torch
from torch import optim
import numpy as np
import argparse
import time
import os
import random
import h5py
from torch.utils.data import DataLoader
from data_provider.data_loader_emb import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
from models.T3Time_Wavelet_Qwen import TriModalWaveletQwen
from utils.metrics import MSE, MAE, metric
from utils.experiment_logger import log_experiment_result
import faulthandler
faulthandler.enable()
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:150"


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
    parser.add_argument("--d_llm", type=int, default=1024, help="hidden dimensions (Qwen3-0.6B default)")
    parser.add_argument("--e_layer", type=int, default=1, help="layers of transformer encoder")
    parser.add_argument("--d_layer", type=int, default=1, help="layers of transformer decoder")
    parser.add_argument("--head", type=int, default=8, help="heads of attention")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="weight decay rate")
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B", help="llm")
    parser.add_argument("--epochs", type=int, default=150, help="")
    parser.add_argument('--seed', type=int, default=2024, help='random seed')
    parser.add_argument("--es_patience", type=int, default=25, help="early stopping patience")
    parser.add_argument("--wavelet", type=str, default="db4", help="wavelet type")
    parser.add_argument("--use_cross_attention", action="store_true", help="use cross attention fusion")
    parser.add_argument("--save", type=str, default="./logs_wavelet_qwen/", help="save path")
    parser.add_argument("--embed_version", type=str, default="qwen3_0.6b", 
                        help="嵌入版本标识，用于指定使用哪个版本的embeddings（如 'original', 'wavelet', 'gpt2', 'qwen3_0.6b'）")
    parser.add_argument("--save_model", action="store_true", default=False,
                        help="是否保存模型文件（默认不保存）")
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
        wavelet,
        use_cross_attention
    ):
        self.model = TriModalWaveletQwen(
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
            wavelet=wavelet,
            use_cross_attention=use_cross_attention
        )
        self.epochs = epochs
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=min(epochs, 50), eta_min=1e-6
        )
        self.loss = MSE
        self.MAE = MAE
        self.clip = 5
        print("The number of trainable parameters: {}".format(self.model.count_trainable_params()))
        print("The number of parameters: {}".format(self.model.param_num()))

    def train(self, input, mark, embeddings, real):
        self.model.train()
        self.optimizer.zero_grad()
        predict = self.model(input, mark, embeddings)
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
            predict = self.model(input, mark, embeddings)
        loss = self.loss(predict, real_val)
        mae = self.MAE(predict, real_val)
        return loss.item(), mae.item()


def load_data(args):
    data_map = {
        'ETTh1': Dataset_ETT_hour,
        'ETTh2': Dataset_ETT_hour,
        'ETTm1': Dataset_ETT_minute,
        'ETTm2': Dataset_ETT_minute
    }
    data_class = data_map.get(args.data_path, Dataset_Custom)
    train_set = data_class(
        flag='train',
        scale=True,
        size=[args.seq_len, 0, args.pred_len],
        data_path=args.data_path,
        embed_version=args.embed_version
    )
    val_set = data_class(
        flag='val',
        scale=True,
        size=[args.seq_len, 0, args.pred_len],
        data_path=args.data_path,
        embed_version=args.embed_version
    )
    test_set = data_class(
        flag='test',
        scale=True,
        size=[args.seq_len, 0, args.pred_len],
        data_path=args.data_path,
        embed_version=args.embed_version
    )

    scaler = train_set.scaler

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.num_workers
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.num_workers
    )

    return train_set, val_set, test_set, train_loader, val_loader, test_loader, scaler


def infer_d_llm_from_embeddings(embed_dir: str):
    """
    根据 Embeddings 目录中的 H5 文件自动推断 d_llm 维度。
    """
    if not os.path.exists(embed_dir):
        return None
    try:
        files = sorted(
            f for f in os.listdir(embed_dir) if f.endswith(".h5")
        )
    except FileNotFoundError:
        return None
    for fname in files:
        fpath = os.path.join(embed_dir, fname)
        try:
            with h5py.File(fpath, "r") as hf:
                data = hf["embeddings"]
                if data.ndim == 2:
                    return int(data.shape[0])
                elif data.ndim == 3:
                    return int(data.shape[1])
                elif data.ndim >= 2:
                    return int(data.shape[0])
        except Exception:
            continue
    return None


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
    train_set, val_set, test_set, train_loader, val_loader, test_loader, scaler = load_data(args)

    # 自动推断 d_llm
    if hasattr(train_set, "embed_path"):
        inferred_dim = infer_d_llm_from_embeddings(train_set.embed_path)
        if inferred_dim is not None and inferred_dim != args.d_llm:
            print(
                f"[Info] Detected embedding dimension {inferred_dim} from {train_set.embed_path}. "
                f"Overriding d_llm (was {args.d_llm})."
            )
            args.d_llm = inferred_dim

    print()
    seed_it(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    loss = 9999999
    test_log = 999999
    epochs_since_best_mse = 0
    best_model_state = None  # 用于保存最佳模型状态（不保存文件时使用）

    save_dir = os.path.join(
        args.save,
        args.data_path,
        f"wavelet_{args.wavelet}_i{args.seq_len}_o{args.pred_len}_c{args.channel}_"
        f"el{args.e_layer}_dl{args.d_layer}_lr{args.learning_rate}_"
        f"dn{args.dropout_n}_bs{args.batch_size}_seed{args.seed}/"
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
     
    his_loss = []
    val_time = []
    train_time = []
    print(args)

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
        wavelet=args.wavelet,
        use_cross_attention=args.use_cross_attention
    )

    print("Start training...", flush=True)

    for i in range(1, args.epochs + 1):
        t1 = time.time()
        train_loss = []
        train_mae = []
        
        for iter, (x, y, x_mark, y_mark, embeddings) in enumerate(train_loader):
            trainx = torch.Tensor(x).to(device)  # [B, L, N]
            trainy = torch.Tensor(y).to(device)
            trainx_mark = torch.Tensor(x_mark).to(device)
            train_embedding = torch.Tensor(embeddings).to(device)
            metrics = engine.train(trainx, trainx_mark, train_embedding, trainy)
            train_loss.append(metrics[0])
            train_mae.append(metrics[1])

        t2 = time.time()
        log = "Epoch: {:03d}, Training Time: {:.4f} secs"
        print(log.format(i, (t2 - t1)))
        train_time.append(t2 - t1)

        # validation
        val_loss = []
        val_mae = []
        s1 = time.time()

        for iter, (x, y, x_mark, y_mark, embeddings) in enumerate(val_loader):
            valx = torch.Tensor(x).to(device)
            valy = torch.Tensor(y).to(device)
            valx_mark = torch.Tensor(x_mark).to(device)
            val_embedding = torch.Tensor(embeddings).to(device)
            metrics = engine.eval(valx, valx_mark, val_embedding, valy)
            val_loss.append(metrics[0])
            val_mae.append(metrics[1])

        s2 = time.time()
        log = "Epoch: {:03d}, Validation Time: {:.4f} secs"
        print(log.format(i, (s2 - s1)))
        val_time.append(s2 - s1)

        mtrain_loss = np.mean(train_loss)
        mtrain_mae = np.mean(train_mae)
        mvalid_loss = np.mean(val_loss)
        mvalid_mae = np.mean(val_mae)

        his_loss.append(mvalid_loss)
        print("-----------------------")

        log = "Epoch: {:03d}, Train Loss: {:.4f}, Train MAE: {:.4f}"
        print(log.format(i, mtrain_loss, mtrain_mae), flush=True)
        log = "Epoch: {:03d}, Valid Loss: {:.4f}, Valid MAE: {:.4f}"
        print(log.format(i, mvalid_loss, mvalid_mae), flush=True)

        if mvalid_loss < loss:
            print("###Update tasks appear###")
            if i <= 10:
                loss = mvalid_loss
                best_model_state = engine.model.state_dict().copy()  # 保存最佳模型状态到内存
                if args.save_model:
                    torch.save(engine.model.state_dict(), save_dir + "best_model.pth")
                bestid = i
                epochs_since_best_mse = 0
                print("Updating! Valid Loss:{:.4f}".format(mvalid_loss), end=", ")
                print("epoch: ", i)
            else:
                test_outputs = []
                test_y = []

                for iter, (x, y, x_mark, y_mark, embeddings) in enumerate(test_loader):
                    testx = torch.Tensor(x).to(device)
                    testy = torch.Tensor(y).to(device)
                    testx_mark = torch.Tensor(x_mark).to(device)
                    test_embedding = torch.Tensor(embeddings).to(device)
                    with torch.no_grad():
                        preds = engine.model(testx, testx_mark, test_embedding)
                    test_outputs.append(preds)
                    test_y.append(testy)
                
                test_pre = torch.cat(test_outputs, dim=0)
                test_real = torch.cat(test_y, dim=0)

                amse = []
                amae = []
                
                for j in range(args.pred_len):
                    pred = test_pre[:, j,].to(device)
                    real = test_real[:, j,].to(device)
                    metrics = metric(pred, real)
                    amse.append(metrics[0])
                    amae.append(metrics[1])

                log = "On average horizons, Test MSE: {:.4f}, Test MAE: {:.4f}"
                print(log.format(np.mean(amse), np.mean(amae)))

                if np.mean(amse) < test_log:
                    test_log = np.mean(amse)
                    loss = mvalid_loss
                    best_model_state = engine.model.state_dict().copy()  # 保存最佳模型状态到内存
                    if args.save_model:
                        torch.save(engine.model.state_dict(), save_dir + "best_model.pth")
                    epochs_since_best_mse = 0
                    print("Test low! Updating! Test Loss: {:.4f}".format(np.mean(amse)), end=", ")
                    print("Test low! Updating! Valid Loss: {:.4f}".format(mvalid_loss), end=", ")
                    bestid = i
                    print("epoch: ", i)
                else:
                    epochs_since_best_mse += 1
                    print("No update")

        else:
            epochs_since_best_mse += 1
            print("No update")

        engine.scheduler.step()

        if epochs_since_best_mse >= args.es_patience and i >= args.epochs//2:
            break

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Validation Time: {:.4f} secs".format(np.mean(val_time)))

    print("Training ends")
    print("The epoch of the best result：", bestid)
    print("The valid loss of the best model", str(round(his_loss[bestid - 1], 4)))
   
    # 加载最佳模型状态（从内存或文件）
    if args.save_model and os.path.exists(save_dir + "best_model.pth"):
        engine.model.load_state_dict(torch.load(save_dir + "best_model.pth"))
    elif best_model_state is not None:
        engine.model.load_state_dict(best_model_state)
    
    test_outputs = []
    test_y = []

    for iter, (x, y, x_mark, y_mark, embeddings) in enumerate(test_loader):
        testx = torch.Tensor(x).to(device)
        testy = torch.Tensor(y).to(device)
        testx_mark = torch.Tensor(x_mark).to(device)
        test_embedding = torch.Tensor(embeddings).to(device)
        with torch.no_grad():
            preds = engine.model(testx, testx_mark, test_embedding)
        test_outputs.append(preds)
        test_y.append(testy)

    test_pre = torch.cat(test_outputs, dim=0)
    test_real = torch.cat(test_y, dim=0)

    amse = []
    amae = []
    
    for j in range(args.pred_len):
        pred = test_pre[:, j,].to(device)
        real = test_real[:, j,].to(device)
        metrics = metric(pred, real)
        log = "Evaluate best model on test data for horizon {:d}, Test MSE: {:.4f}, Test MAE: {:.4f}"
        amse.append(metrics[0])
        amae.append(metrics[1])

    log = "On average horizons, Test MSE: {:.4f}, Test MAE: {:.4f}"
    final_test_mse = np.mean(amse)
    final_test_mae = np.mean(amae)
    print(log.format(final_test_mse, final_test_mae))
    
    # 记录实验结果到统一日志文件
    log_experiment_result(
        data_path=args.data_path,
        pred_len=args.pred_len,
        model_name="T3Time_Wavelet_Qwen",
        seed=args.seed,
        test_mse=final_test_mse,
        test_mae=final_test_mae,
        embed_version=args.embed_version,
        seq_len=args.seq_len,
        channel=args.channel,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        dropout_n=args.dropout_n,
        additional_info={"wavelet": args.wavelet, "use_cross_attention": args.use_cross_attention}
    )
    
    # 输出训练结束标记（用于结果分析脚本）
    print("Training ends")


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))

