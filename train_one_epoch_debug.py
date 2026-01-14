"""
只运行一个epoch的训练脚本，用于调试和查看维度信息
使用方法: python train_one_epoch_debug.py --data ETTh1 --embed_version original
"""
import torch
from torch import optim
import numpy as np
import argparse
import time
import os
import random
import importlib.util
from torch.utils.data import DataLoader
from data_provider.data_loader_emb import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
from utils.metrics import MSE, MAE, metric
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
    parser = argparse.ArgumentParser(description="单epoch训练调试脚本 - 显示维度信息")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument("--data_path", type=str, default="ETTh1", help="数据集路径")
    parser.add_argument("--channel", type=int, default=32, help="特征维度")
    parser.add_argument("--num_nodes", type=int, default=7, help="变量数量")
    parser.add_argument("--seq_len", type=int, default=96, help="输入序列长度")
    parser.add_argument("--pred_len", type=int, default=96, help="预测长度")
    parser.add_argument("--batch_size", type=int, default=32, help="批大小")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--dropout_n", type=float, default=0.2, help="Dropout率")
    parser.add_argument("--d_llm", type=int, default=768, help="LLM嵌入维度")
    parser.add_argument("--e_layer", type=int, default=1, help="编码器层数")
    parser.add_argument("--d_layer", type=int, default=1, help="解码器层数")
    parser.add_argument("--head", type=int, default=8, help="注意力头数")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="权重衰减")
    parser.add_argument("--num_workers", type=int, default=10, help="数据加载工作进程数")
    parser.add_argument('--seed', type=int, default=2024, help='随机种子')
    parser.add_argument("--embed_version", type=str, default="original", 
                        help="嵌入版本标识（如 'original', 'qwen3_0.6b'）")
    parser.add_argument("--max_batches", type=int, default=3, 
                        help="最多打印多少个batch的维度信息（默认3个）")
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
    ):
        self.model = TriModal(
            device=device, channel=channel, num_nodes=num_nodes, seq_len=seq_len, pred_len=pred_len, 
            dropout_n=dropout_n, d_llm=d_llm, e_layer=e_layer, d_layer=d_layer, head=head
        )
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = MSE
        self.MAE = MAE
        self.clip = 5
        print("="*80)
        print("模型参数统计")
        print("="*80)
        print("可训练参数数量: {}".format(self.model.count_trainable_params()))
        print("总参数数量: {}".format(self.model.param_num()))
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
    
    print("\n" + "="*80)
    print("单Epoch训练调试脚本 - 显示维度信息")
    print("="*80)
    print(f"数据集: {args.data_path}")
    print(f"嵌入版本: {args.embed_version}")
    print(f"批大小: {args.batch_size}")
    print(f"最多打印前 {args.max_batches} 个batch的维度信息")
    print("="*80 + "\n")
    
    train_set, val_set, test_set, train_loader, val_loader, test_loader, scaler = load_data(args)
    
    print(f"训练集大小: {len(train_set)}")
    print(f"验证集大小: {len(val_set)}")
    print(f"测试集大小: {len(test_set)}")
    print(f"训练批次数: {len(train_loader)}")
    print(f"验证批次数: {len(val_loader)}")
    print(f"测试批次数: {len(test_loader)}\n")

    seed_it(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}\n")

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
    )

    print("\n" + "="*80)
    print("开始训练（1个Epoch）")
    print("="*80 + "\n")

    # 训练阶段
    t1 = time.time()
    train_loss = []
    train_mae = []
    
    for iter, (x, y, x_mark, y_mark, embeddings) in enumerate(train_loader):
        trainx = torch.Tensor(x).to(device)
        trainy = torch.Tensor(y).to(device)
        trainx_mark = torch.Tensor(x_mark).to(device)
        train_embedding = torch.Tensor(embeddings).to(device)
        
        # 只在前几个batch打印维度信息
        if iter < args.max_batches:
            print(f"\n{'='*80}")
            print(f"训练阶段 - Batch {iter + 1}/{args.max_batches}")
            print(f"{'='*80}")
        
        metrics = engine.train(trainx, trainx_mark, train_embedding, trainy)
        train_loss.append(metrics[0])
        train_mae.append(metrics[1])
        
        if iter < args.max_batches:
            print(f"\n[Batch {iter + 1}] Loss: {metrics[0]:.6f}, MAE: {metrics[1]:.6f}")
            print("="*80 + "\n")

    t2 = time.time()
    mtrain_loss = np.mean(train_loss)
    mtrain_mae = np.mean(train_mae)
    
    print(f"\n训练完成 - 用时: {t2-t1:.2f}秒")
    print(f"平均训练 Loss: {mtrain_loss:.6f}, 平均训练 MAE: {mtrain_mae:.6f}\n")

    # 验证阶段
    print("="*80)
    print("开始验证")
    print("="*80 + "\n")
    
    s1 = time.time()
    val_loss = []
    val_mae = []
    
    for iter, (x, y, x_mark, y_mark, embeddings) in enumerate(val_loader):
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
        
        if iter == 0:
            print(f"\n[验证 Batch 1] Loss: {metrics[0]:.6f}, MAE: {metrics[1]:.6f}")
            print("="*80 + "\n")

    s2 = time.time()
    mvalid_loss = np.mean(val_loss)
    mvalid_mae = np.mean(val_mae)
    
    print(f"\n验证完成 - 用时: {s2-s1:.2f}秒")
    print(f"平均验证 Loss: {mvalid_loss:.6f}, 平均验证 MAE: {mvalid_mae:.6f}\n")

    # 总结
    print("="*80)
    print("训练总结")
    print("="*80)
    print(f"训练 Loss: {mtrain_loss:.6f}, 训练 MAE: {mtrain_mae:.6f}")
    print(f"验证 Loss: {mvalid_loss:.6f}, 验证 MAE: {mvalid_mae:.6f}")
    print(f"总用时: {s2-t1:.2f}秒")
    print("="*80)

if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print(f"\n脚本总耗时: {t2-t1:.2f}秒")
