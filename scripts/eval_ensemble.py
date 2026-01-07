import sys
import os
# 【关键修复】 将项目根目录加入 python path，解决 ModuleNotFoundError
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import torch
import numpy as np
import json
from datetime import datetime
from torch.utils.data import DataLoader
from models.T3Time_Learnable_Wavelet_Packet_Gated_Pro_Qwen import TriModalLearnableWaveletPacketGatedProQwen
from data_provider.data_loader_emb import Dataset_ETT_hour, Dataset_Custom
from utils.metrics import metric

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seeds = [2024, 2025, 2026]  # 确保这些 seed 的 checkpoint 已经存在
    
    # Data Provider 配置
    # 确保这里的配置与 train.py 一致
    data_set = Dataset_ETT_hour(
        root_path='./dataset/',
        data_path='ETTh1.csv',
        flag='test',
        size=[96, 0, 96],
        features='M',
        embed_version='qwen3_0.6b'
    )
    loader = DataLoader(
        data_set, 
        batch_size=16, 
        shuffle=False, 
        num_workers=4, 
        drop_last=True
    )
    
    all_preds = []
    trues = []
    
    # 获取真实值
    print("Loading True Values...")
    for i, (bx, by, bxm, bym, emb) in enumerate(loader):
        trues.append(by.detach().cpu().numpy())
    trues = np.concatenate(trues, axis=0)
    
    # 遍历 Seed 进行预测
    for seed in seeds:
        checkpoint_path = f"./checkpoints/checkpoint_seed_{seed}.pth"
        if not os.path.exists(checkpoint_path):
            print(f"Warning: {checkpoint_path} not found, skipping...")
            continue
            
        print(f"Evaluating Model Seed {seed}...")
        
        # 初始化模型 (参数需与训练时一致: dropout_n=0.5, wp_level=2)
        model = TriModalLearnableWaveletPacketGatedProQwen(
            device=device, channel=128, num_nodes=7, seq_len=96, pred_len=96, 
            dropout_n=0.5, wp_level=2
        ).to(device)
        
        model.load_state_dict(torch.load(checkpoint_path))
        model.eval()
        
        preds = []
        with torch.no_grad():
            for i, (bx, by, bxm, bym, emb) in enumerate(loader):
                bx, by = bx.to(device).float(), by.to(device).float()
                bxm, emb = bxm.to(device).float(), emb.to(device).float()
                
                outputs = model(bx, bxm, emb)
                preds.append(outputs.detach().cpu().numpy())
        
        all_preds.append(np.concatenate(preds, axis=0))
        
    if not all_preds:
        print("No models evaluated!")
        return

    # 【集成核心】 取平均
    ensemble_preds = np.mean(all_preds, axis=0)
    
    # 计算指标
    mse, mae = metric(torch.tensor(ensemble_preds), torch.tensor(trues))
    print("\n" + "="*40)
    print(f"🏆 Final Ensemble Result (Seeds: {len(all_preds)})")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print("="*40)

    # 【新增】 将集成结果写入日志文件
    log_path = os.path.join(parent_dir, "experiment_results.log")
    res = {
        "model_id": "T3Time_Pro_Qwen_SOTA_V30_Ensemble",  # 标记为集成版
        "data_path": "ETTh1",
        "pred_len": 96,
        "test_mse": float(mse),
        "test_mae": float(mae),
        "model": "TriModal_Ensemble_V30",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "seed": "Ensemble_3Seeds",  # 标记种子
        "seq_len": 96,
        "channel": 128,
        "batch_size": 16,
        "learning_rate": 0.0001,
        "dropout_n": 0.5,
        "weight_decay": 0.001,
        "wp_level": 2,
        "note": f"Average of Seeds {', '.join(map(str, seeds[:len(all_preds)]))}"
    }
    
    with open(log_path, "a") as f:
        f.write(json.dumps(res) + "\n")
    print(f"✅ Ensemble result saved to {log_path}")

if __name__ == "__main__":
    main()
