"""
训练带 Gated Attention 和 动态统计特征注入 (方案二) 的小波变换 T3Time 模型
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
from models.T3Time_Wavelet_Gated_Stat_Qwen import TriModalWaveletGatedStatQwen
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
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--dropout_n", type=float, default=0.2, help="dropout rate")
    parser.add_argument("--d_llm", type=int, default=1024, help="hidden dimensions")
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
    parser.add_argument("--save", type=str, default="./logs_wavelet_gated_stat_qwen/", help="save path")
    parser.add_argument("--embed_version", type=str, default="qwen3_0.6b", help="嵌入版本")
    parser.add_argument("--save_model", action="store_true", default=False, help="是否保存模型文件")
    return parser.parse_args()


class trainer:
    def __init__(self, scaler, channel, num_nodes, seq_len, pred_len, dropout_n, d_llm, e_layer, d_layer, head, lrate, wdecay, device, epochs, wavelet, use_cross_attention):
        self.model = TriModalWaveletGatedStatQwen(
            device=device, channel=channel, num_nodes=num_nodes, seq_len=seq_len, pred_len=pred_len,
            dropout_n=dropout_n, d_llm=d_llm, e_layer=e_layer, d_layer=d_layer, head=head,
            wavelet=wavelet, use_cross_attention=use_cross_attention
        )
        self.epochs = epochs
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=min(epochs, 50), eta_min=1e-6)
        self.loss = MSE
        self.MAE = MAE
        self.clip = 5
        print("The number of trainable parameters: {}".format(self.model.count_trainable_params()))

    def train(self, input, mark, embeddings, real):
        self.model.train()
        self.optimizer.zero_grad()
        predict = self.model(input, mark, embeddings)
        loss = self.loss(predict, real)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        return loss.item(), self.MAE(predict, real).item()
    
    def eval(self, input, mark, embeddings, real_val):
        self.model.eval()
        with torch.no_grad():
            predict = self.model(input, mark, embeddings)
        return self.loss(predict, real_val).item(), self.MAE(predict, real_val).item()


def load_data(args):
    data_map = {'ETTh1': Dataset_ETT_hour, 'ETTh2': Dataset_ETT_hour, 'ETTm1': Dataset_ETT_minute, 'ETTm2': Dataset_ETT_minute}
    data_class = data_map.get(args.data_path, Dataset_Custom)
    train_set = data_class(flag='train', scale=True, size=[args.seq_len, 0, args.pred_len], data_path=args.data_path, embed_version=args.embed_version)
    val_set = data_class(flag='val', scale=True, size=[args.seq_len, 0, args.pred_len], data_path=args.data_path, embed_version=args.embed_version)
    test_set = data_class(flag='test', scale=True, size=[args.seq_len, 0, args.pred_len], data_path=args.data_path, embed_version=args.embed_version)
    return train_set, val_set, test_set, DataLoader(train_set, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=args.num_workers), DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=args.num_workers), DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=args.num_workers), train_set.scaler


def infer_d_llm_from_embeddings(embed_dir: str):
    if not os.path.exists(embed_dir): return None
    try:
        files = sorted(f for f in os.listdir(embed_dir) if f.endswith(".h5"))
        for fname in files:
            with h5py.File(os.path.join(embed_dir, fname), 'r') as f:
                if 'data' in f:
                    data = f['data']
                    if len(data.shape) == 2: return data.shape[0]
                    elif len(data.shape) == 3: return data.shape[1]
    except Exception as e:
        print(f"Error inferring d_llm: {e}")
    return None


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    embed_dir = f"./embeddings/{args.data_path}_{args.embed_version}/"
    inferred_d_llm = infer_d_llm_from_embeddings(embed_dir)
    if inferred_d_llm:
        print(f"Detected embedding dimension {inferred_d_llm} from {embed_dir}. Overriding d_llm (was {args.d_llm}).")
        args.d_llm = inferred_d_llm

    train_set, val_set, test_set, train_loader, val_loader, test_loader, scaler = load_data(args)
    engine = trainer(scaler, args.channel, args.num_nodes, args.seq_len, args.pred_len, args.dropout_n, args.d_llm, args.e_layer, args.d_layer, args.head, args.learning_rate, args.weight_decay, args.device, args.epochs, args.wavelet, args.use_cross_attention)

    save_path = args.save + args.data_path + f"/stat_i{args.seq_len}_o{args.pred_len}_c{args.channel}_lr{args.learning_rate}_dn{args.dropout_n}_bs{args.batch_size}_seed{args.seed}/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    best_val_loss = float('inf')
    best_test_mae = float('inf')
    best_model_state = None
    patience_counter = 0

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        train_loss, train_mae = [], []
        for i, (input, real, mark, mark_y, embeddings) in enumerate(train_loader):
            input, mark, embeddings, real = input.to(args.device), mark.to(args.device), embeddings.to(args.device), real.to(args.device)
            loss, mae = engine.train(input, mark, embeddings, real)
            train_loss.append(loss)
            train_mae.append(mae)

        val_loss, val_mae = [], []
        for i, (input, real, mark, mark_y, embeddings) in enumerate(val_loader):
            input, mark, embeddings, real = input.to(args.device), mark.to(args.device), embeddings.to(args.device), real.to(args.device)
            loss, mae = engine.eval(input, mark, embeddings, real)
            val_loss.append(loss)
            val_mae.append(mae)

        avg_val_loss = np.mean(val_loss)
        print(f"Epoch {epoch+1}: Train Loss {np.mean(train_loss):.4f}, Val Loss {avg_val_loss:.4f}, Time {time.time()-epoch_start_time:.2f}s")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            if args.save_model:
                torch.save(engine.model.state_dict(), os.path.join(save_path, "best_model.pth"))
            else:
                best_model_state = engine.model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= args.es_patience:
                print("Early stopping")
                break
        engine.scheduler.step()

    if args.save_model:
        engine.model.load_state_dict(torch.load(os.path.join(save_path, "best_model.pth")))
    elif best_model_state is not None:
        engine.model.load_state_dict(best_model_state)

    test_mse, test_mae = [], []
    for i, (input, real, mark, mark_y, embeddings) in enumerate(test_loader):
        input, mark, embeddings, real = input.to(args.device), mark.to(args.device), embeddings.to(args.device), real.to(args.device)
        mse, mae = engine.eval(input, mark, embeddings, real)
        test_mse.append(mse)
        test_mae.append(mae)
    
    final_mse, final_mae = np.mean(test_mse), np.mean(test_mae)
    print(f"\nFinal Test MSE: {final_mse:.6f}, Test MAE: {final_mae:.6f}")
    
    log_experiment_result(
        data_path=args.data_path,
        pred_len=args.pred_len,
        model_name="T3Time_Wavelet_Gated_Stat_Qwen",
        seed=args.seed,
        test_mse=final_mse,
        test_mae=final_mae,
        embed_version=args.embed_version,
        seq_len=args.seq_len,
        channel=args.channel,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        dropout_n=args.dropout_n,
        additional_info={"stat_injection": True}
    )

if __name__ == "__main__":
    main()

