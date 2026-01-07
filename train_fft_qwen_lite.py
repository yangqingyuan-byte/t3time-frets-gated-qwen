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
from models.T3Time_FFT_Qwen_Lite import TriModalFFTQwenLite
from utils.metrics import MSE, MAE, metric
from utils.experiment_logger import log_experiment_result

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="ETTh1", help="data path")
    parser.add_argument("--channel", type=int, default=64, help="hidden channel size")
    parser.add_argument("--num_nodes", type=int, default=7, help="number of nodes")
    parser.add_argument("--seq_len", type=int, default=96, help="input sequence length")
    parser.add_argument("--pred_len", type=int, default=96, help="prediction length")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
    parser.add_argument("--d_llm", type=int, default=1024, help="LLM embedding dimension")
    parser.add_argument("--epochs", type=int, default=100, help="training epochs")
    parser.add_argument('--seed', type=int, default=2024, help='random seed')
    parser.add_argument("--es_patience", type=int, default=15, help="early stopping patience")    
    parser.add_argument("--embed_version", type=str, default="qwen3_0.6b", help="embedding version")
    parser.add_argument("--save_model", action="store_true", default=False, help="whether to save model")
    return parser.parse_args()

class trainer:
    def __init__(self, model, lrate, wdecay, epochs):
        self.model = model
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-6)
        self.loss = MSE
        self.MAE = MAE

    def train(self, input, mark, embeddings, real):
        self.model.train()
        self.optimizer.zero_grad()
        predict = self.model(input, mark, embeddings)
        loss = self.loss(predict, real)
        loss.backward()
        self.optimizer.step()
        return loss.item(), self.MAE(predict, real).item()
    
    def eval(self, input, mark, embeddings, real_val):
        self.model.eval()
        with torch.no_grad():
            predict = self.model(input, mark, embeddings)
        return self.loss(predict, real_val).item(), self.MAE(predict, real_val).item()

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data Loading
    data_map = {'ETTh1': Dataset_ETT_hour, 'ETTh2': Dataset_ETT_hour, 'ETTm1': Dataset_ETT_minute, 'ETTm2': Dataset_ETT_minute}
    data_class = data_map.get(args.data_path, Dataset_Custom)
    train_set = data_class(flag='train', scale=True, size=[args.seq_len, 0, args.pred_len], data_path=args.data_path, embed_version=args.embed_version)
    val_set = data_class(flag='val', scale=True, size=[args.seq_len, 0, args.pred_len], data_path=args.data_path, embed_version=args.embed_version)
    test_set = data_class(flag='test', scale=True, size=[args.seq_len, 0, args.pred_len], data_path=args.data_path, embed_version=args.embed_version)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    model = TriModalFFTQwenLite(device=device, channel=args.channel, num_nodes=args.num_nodes, seq_len=args.seq_len, pred_len=args.pred_len, d_llm=args.d_llm, dropout=args.dropout)
    engine = trainer(model, args.learning_rate, 1e-3, args.epochs)

    print(f"Model Parameters: {model.count_trainable_params()}")

    best_val_loss = float('inf')
    best_test_mse = float('inf')
    best_test_mae = float('inf')
    patience_cnt = 0

    for epoch in range(1, args.epochs + 1):
        train_losses = []
        for x, y, x_mark, y_mark, emb in train_loader:
            loss, _ = engine.train(x.to(device), x_mark.to(device), emb.to(device), y.to(device))
            train_losses.append(loss)
        
        val_losses = []
        for x, y, x_mark, y_mark, emb in val_loader:
            loss, _ = engine.eval(x.to(device), x_mark.to(device), emb.to(device), y.to(device))
            val_losses.append(loss)
        
        m_val_loss = np.mean(val_losses)
        print(f"Epoch {epoch}: Train Loss {np.mean(train_losses):.4f}, Val Loss {m_val_loss:.4f}")
        
        if m_val_loss < best_val_loss:
            best_val_loss = m_val_loss
            patience_cnt = 0
            # Test
            test_mses, test_maes = [], []
            for x, y, x_mark, y_mark, emb in test_loader:
                mse, mae = engine.eval(x.to(device), x_mark.to(device), emb.to(device), y.to(device))
                test_mses.append(mse)
                test_maes.append(mae)
            best_test_mse, best_test_mae = np.mean(test_mses), np.mean(test_maes)
            print(f" >>> New Best Test MSE: {best_test_mse:.4f}")
        else:
            patience_cnt += 1
        
        engine.scheduler.step()
        if patience_cnt >= args.es_patience:
            break

    log_experiment_result(args.data_path, args.pred_len, "T3Time_FFT_Qwen_Lite", args.seed, best_test_mse, best_test_mae, args.embed_version)
    print(f"Final Best Test MSE: {best_test_mse:.4f}, MAE: {best_test_mae:.4f}")

if __name__ == "__main__":
    main()

