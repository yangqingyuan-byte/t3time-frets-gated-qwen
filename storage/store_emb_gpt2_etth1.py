"""
生成 ETTh1 数据集的 GPT-2 嵌入（训练、验证、测试集）
使用方法：
    python storage/store_emb_gpt2_etth1.py --divide train
    python storage/store_emb_gpt2_etth1.py --divide val
    python storage/store_emb_gpt2_etth1.py --divide test
"""
import torch
import sys
import os
import time
import h5py
import argparse

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader
from data_provider.data_loader_save import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
from storage.gen_prompt_emb import GenPromptEmb

def parse_args():
    parser = argparse.ArgumentParser(description="生成 ETTh1 数据集的 GPT-2 嵌入")
    parser.add_argument("--device", type=str, default="cuda", help="设备 (cuda/cpu)")
    parser.add_argument("--data_path", type=str, default="ETTh1", help="数据集路径")
    parser.add_argument("--num_nodes", type=int, default=7, help="变量数量")
    parser.add_argument("--input_len", type=int, default=96, help="输入序列长度")
    parser.add_argument("--output_len", type=int, default=96, help="输出序列长度")
    parser.add_argument("--batch_size", type=int, default=1, help="批大小（建议保持为1）")
    parser.add_argument("--d_model", type=int, default=768, help="GPT-2 的隐藏层维度")
    parser.add_argument("--l_layers", type=int, default=12, help="GPT-2 的 Transformer 层数")
    parser.add_argument("--model_name", type=str, default="gpt2", help="GPT-2 模型名称")
    parser.add_argument("--divide", type=str, default="train", choices=["train", "val", "test"],
                        help="数据集划分：train/val/test")
    parser.add_argument("--num_workers", type=int, default=min(10, os.cpu_count()), help="数据加载器工作进程数")
    parser.add_argument("--embed_version", type=str, default="original",
                        help="嵌入版本标识，用于区分不同版本生成的嵌入（默认 'original'）")
    return parser.parse_args()

def get_dataset(data_path, flag, input_len, output_len):
    """获取数据集"""
    datasets = {
        'ETTh1': Dataset_ETT_hour,
        'ETTh2': Dataset_ETT_hour,
        'ETTm1': Dataset_ETT_minute,
        'ETTm2': Dataset_ETT_minute
    }
    dataset_class = datasets.get(data_path, Dataset_Custom)
    return dataset_class(flag=flag, size=[input_len, 0, output_len], data_path=data_path)

def save_embeddings(args):
    """生成并保存嵌入"""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载数据集
    print(f"加载 {args.divide} 数据集...")
    dataset = get_dataset(args.data_path, args.divide, args.input_len, args.output_len)
    data_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        drop_last=False, 
        num_workers=args.num_workers
    )
    
    print(f"数据集大小: {len(dataset)} 个样本")
    print(f"批次数: {len(data_loader)}")
    
    # 初始化 GPT-2 嵌入生成器
    print(f"初始化 GPT-2 模型 ({args.model_name})...")
    gen_prompt_emb = GenPromptEmb(
        device=device,
        input_len=args.input_len,
        data_path=args.data_path,
        model_name=args.model_name,
        d_model=args.d_model,
        layer=args.l_layers,
        divide=args.divide
    ).to(device)
    gen_prompt_emb.eval()  # 设置为评估模式
    
    # 创建保存目录
    save_path = f"./Embeddings/{args.data_path}/{args.embed_version}/{args.divide}/"
    os.makedirs(save_path, exist_ok=True)
    print(f"\n保存路径: {save_path}")
    print(f"嵌入版本: {args.embed_version}")
    print(f"数据集划分: {args.divide}")
    print("="*80)
    
    # 生成并保存嵌入
    start_time = time.time()
    total_batches = len(data_loader)
    
    # 根据总批次数决定进度显示频率
    if total_batches <= 10:
        progress_interval = 1  # 每个批次都显示
    elif total_batches <= 100:
        progress_interval = 10  # 每10个批次显示
    elif total_batches <= 1000:
        progress_interval = 50  # 每50个批次显示
    else:
        progress_interval = 100  # 每100个批次显示
    
    print(f"开始处理，共 {total_batches} 个批次，每 {progress_interval} 个批次显示一次进度...")
    print("-"*80)
    
    for i, (x, y, x_mark, y_mark) in enumerate(data_loader):
        # 实时进度显示
        if (i + 1) % progress_interval == 0 or i == 0 or (i + 1) == total_batches:
            elapsed = time.time() - start_time
            progress_pct = (i + 1) / total_batches * 100
            
            # 计算预估剩余时间
            if i > 0:
                avg_time_per_batch = elapsed / (i + 1)
                remaining_batches = total_batches - (i + 1)
                eta_seconds = avg_time_per_batch * remaining_batches
                eta_str = f", 预计剩余: {eta_seconds/60:.1f}分钟"
            else:
                eta_str = ""
            
            print(f"[进度] {i + 1}/{total_batches} ({progress_pct:.1f}%) | "
                  f"已用时: {elapsed/60:.2f}分钟{eta_str}")
        
        # 生成嵌入
        with torch.no_grad():
            embeddings = gen_prompt_emb.generate_embeddings(x.to(device), x_mark.to(device))
        
        # 保存为 H5 文件
        file_path = f"{save_path}{i}.h5"
        with h5py.File(file_path, 'w') as hf:
            hf.create_dataset('embeddings', data=embeddings.cpu().numpy())
    
    total_time = time.time() - start_time
    print("="*80)
    print(f"完成！共处理 {len(data_loader)} 个批次")
    print(f"总耗时: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")
    print(f"平均每个批次: {total_time/len(data_loader):.2f} 秒")
    print(f"嵌入文件保存在: {save_path}")
    
if __name__ == "__main__":
    args = parse_args()
    print("="*80)
    print("GPT-2 嵌入生成脚本")
    print("="*80)
    print(f"数据集: {args.data_path}")
    print(f"划分: {args.divide}")
    print(f"输入长度: {args.input_len}")
    print(f"输出长度: {args.output_len}")
    print(f"GPT-2 模型: {args.model_name}")
    print(f"嵌入维度: {args.d_model}")
    print("="*80)
    
    t1 = time.time()
    save_embeddings(args)
    t2 = time.time()
    print(f"\n总耗时: {(t2 - t1)/60:.4f} 分钟")
