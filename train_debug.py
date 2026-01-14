import torch
from torch import optim
import numpy as np
import argparse
import time
import os
import random
import h5py
import importlib.util
from torch.utils.data import DataLoader
from data_provider.data_loader_emb import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
from utils.metrics import MSE, MAE, metric
from utils.experiment_logger import log_experiment_result
import faulthandler
faulthandler.enable()
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:150"

# 导入 T3Time原模型.py 中的 TriModal
spec = importlib.util.spec_from_file_location("T3Time原模型", "models/T3Time原模型.py")
T3Time原模型 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T3Time原模型)
TriModal = T3Time原模型.TriModal

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda", help="")
    parser.add_argument("--data_path", type=str, default="ETTh1", help="data path")
    parser.add_argument("--channel", type=int, default=32, help="number of features")
    parser.add_argument("--num_nodes", type=int, default=7, help="number of nodes")
    parser.add_argument("--seq_len", type=int, default=96, help="seq_len")
    parser.add_argument("--pred_len", type=int, default=96, help="out_len")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--dropout_n", type=float, default=0.2, help="dropout rate of neural network layers")
    parser.add_argument("--d_llm", type=int, default=768, help="hidden dimensions")
    parser.add_argument("--e_layer", type=int, default=1, help="layers of transformer encoder")
    parser.add_argument("--d_layer", type=int, default=1, help="layers of transformer decoder")
    parser.add_argument("--head", type=int, default=8, help="heads of attention")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="weight decay rate")
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--model_name", type=str, default="gpt2", help="llm")
    parser.add_argument("--epochs", type=int, default=1, help="只运行1个epoch用于调试")
    parser.add_argument('--seed', type=int, default=2024, help='random seed')
    parser.add_argument("--es_patience", type=int, default=25, help="quit if no improvement after this many iterations")    
    parser.add_argument("--save", type=str, default="./logs_debug/" + str(time.strftime("%Y-%m-%d-%H:%M:%S")) + "-", help="save path")
    parser.add_argument("--embed_version", type=str, default="qwen3_0.6b", 
                        help="嵌入版本标识，用于指定使用哪个版本的embeddings（如 'original', 'wavelet', 'gpt2', 'qwen3_0.6b'）")
    parser.add_argument("--save_model", action="store_true", default=False,
                        help="是否保存模型文件（默认不保存）")
    parser.add_argument("--max_batches", type=int, default=3, help="每个epoch最多打印多少个batch的维度信息（默认3个）")
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
        epochs
    ):
        self.model = TriModal(
            device=device, channel=channel, num_nodes=num_nodes, seq_len=seq_len, pred_len=pred_len, 
            dropout_n=dropout_n, d_llm=d_llm, e_layer=e_layer, d_layer=d_layer, head=head
        )
        self.epochs = epochs
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=min(epochs, 50), eta_min=1e-6)
        self.loss = MSE
        self.MAE = MAE
        self.clip = 5
        print("="*80)
        print("模型参数统计")
        print("="*80)
        print("The number of trainable parameters: {}".format(self.model.count_trainable_params()))
        print("The number of parameters: {}".format(self.model.param_num()))
        print("="*80)


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
            predict = self.model(input,mark, embeddings)
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
    train_set = data_class(flag='train', scale=True, size=[args.seq_len, 0, args.pred_len], 
                          data_path=args.data_path, embed_version=args.embed_version)
    val_set = data_class(flag='val', scale=True, size=[args.seq_len, 0, args.pred_len], 
                        data_path=args.data_path, embed_version=args.embed_version)
    test_set = data_class(flag='test', scale=True, size=[args.seq_len, 0, args.pred_len], 
                          data_path=args.data_path, embed_version=args.embed_version)

    scaler = train_set.scaler

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=args.num_workers)

    return train_set, val_set, test_set, train_loader, val_loader, test_loader, scaler


def infer_d_llm_from_embeddings(embed_dir: str):
    """
    根据 Embeddings 目录中的 H5 文件自动推断 d_llm 维度。
    约定：H5 中 key 为 'embeddings'，形状为 (d_llm, num_nodes) 或 (batch, d_llm, num_nodes)。
    对于 2D: shape[0] 是 d_llm，shape[1] 是 num_nodes
    对于 3D: shape[1] 是 d_llm，shape[2] 是 num_nodes
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
                    # 形状为 (d_llm, num_nodes)
                    return int(data.shape[0])
                elif data.ndim == 3:
                    # 形状为 (batch, d_llm, num_nodes)
                    return int(data.shape[1])
                elif data.ndim >= 2:
                    # 其他情况，尝试第一维
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

    # 如果存在预生成的 Embeddings，则尝试自动推断 d_llm 维度（支持 GPT2 / Qwen3 等）
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

    path = os.path.join(args.save, args.data_path, 
                        f"{args.pred_len}_{args.channel}_{args.e_layer}_{args.d_layer}_{args.learning_rate}_{args.dropout_n}_{args.seed}/")
    if not os.path.exists(path):
        os.makedirs(path)
     
    his_loss = []
    val_time = []
    train_time = []
    print("="*80)
    print("训练参数配置")
    print("="*80)
    print(args)
    print("="*80)

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
        epochs=args.epochs
    )

    print("\n开始训练（调试模式 - 打印维度信息）...", flush=True)
    print(f"每个epoch最多打印前 {args.max_batches} 个batch的维度信息\n")

    for i in range(1, args.epochs + 1):

        t1 = time.time()
        train_loss = []
        train_mae = []
        
        for iter, (x,y,x_mark,y_mark, embeddings) in enumerate(train_loader):
            trainx = torch.Tensor(x).to(device) # [B, L, N]
            trainy = torch.Tensor(y).to(device)
            trainx_mark = torch.Tensor(x_mark).to(device) 
            train_embedding = torch.Tensor(embeddings).to(device)
            
            # 只在前几个batch打印维度信息
            if iter < args.max_batches:
                print(f"\n{'='*80}")
                print(f"Batch {iter + 1}/{args.max_batches} - 训练阶段")
                print(f"{'='*80}")
            
            metrics = engine.train(trainx, trainx_mark, train_embedding, trainy)
            train_loss.append(metrics[0])
            train_mae.append(metrics[1])
            
            # 只在打印的batch后显示loss
            if iter < args.max_batches:
                print(f"\n[Batch {iter + 1}] Loss: {metrics[0]:.6f}, MAE: {metrics[1]:.6f}")

        t2 = time.time()
        log = "Epoch: {:03d}, Training Time: {:.4f} secs"
        print(log.format(i, (t2 - t1)))
        train_time.append(t2 - t1)

        # validation
        val_loss = []
        val_mae = []
        s1 = time.time()

        for iter, (x,y,x_mark,y_mark, embeddings) in enumerate(val_loader):
            valx = torch.Tensor(x).to(device)
            valy = torch.Tensor(y).to(device)
            valx_mark = torch.Tensor(x_mark).to(device)
            val_embedding = torch.Tensor(embeddings).to(device)
            
            # 只在第一个batch打印验证阶段的维度信息
            if iter == 0:
                print(f"\n{'='*80}")
                print(f"验证阶段 - Batch 1")
                print(f"{'='*80}")
            
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

        log = "Epoch: {:03d}, Train Loss: {:.4f}, Train MAE: {:.4f} "
        print(
            log.format(i, mtrain_loss, mtrain_mae),
            flush=True,
        )
        log = "Epoch: {:03d}, Valid Loss: {:.4f}, Valid MAE: {:.4f}"
        print(
            log.format(i, mvalid_loss, mvalid_mae),
            flush=True,
        )

    # Output consumption
    print("\n" + "="*80)
    print("训练完成")
    print("="*80)
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Validation Time: {:.4f} secs".format(np.mean(val_time)))
    print("="*80)

if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("\n总耗时: {:.4f} 秒".format(t2 - t1))
