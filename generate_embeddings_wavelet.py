"""
生成训练/验证/测试集的embeddings
用于小波变换模型
注意：此脚本复用原有的embedding生成逻辑，因为embeddings与频域处理方式无关
"""
import torch
import sys
import os
import time
import h5py
import argparse
from torch.utils.data import DataLoader
from data_provider.data_loader_save import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'storage'))
from gen_prompt_emb import GenPromptEmb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda", help="")
    parser.add_argument("--data_path", type=str, default="ETTh1")
    parser.add_argument("--num_nodes", type=int, default=7)
    parser.add_argument("--input_len", type=int, default=96)
    parser.add_argument("--output_len", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--l_layers", type=int, default=12)
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--divide", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--num_workers", type=int, default=min(10, os.cpu_count()))
    parser.add_argument("--embed_version", type=str, default="wavelet", 
                        help="嵌入版本标识，用于区分不同版本生成的嵌入（如 'original', 'wavelet', 'gpt2'）")
    return parser.parse_args()


def get_dataset(data_path, flag, input_len, output_len):
    datasets = {
        'ETTh1': Dataset_ETT_hour,
        'ETTh2': Dataset_ETT_hour,
        'ETTm1': Dataset_ETT_minute,
        'ETTm2': Dataset_ETT_minute
    }
    dataset_class = datasets.get(data_path, Dataset_Custom)
    return dataset_class(flag=flag, size=[input_len, 0, output_len], data_path=data_path)


def save_embeddings(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载数据集
    dataset = get_dataset(args.data_path, args.divide, args.input_len, args.output_len)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers
    )
    
    print(f"Dataset: {args.data_path}, Split: {args.divide}, Samples: {len(dataset)}")
    
    # 初始化embedding生成器
    gen_prompt_emb = GenPromptEmb(
        device=device,
        input_len=args.input_len,
        data_path=args.data_path,
        model_name=args.model_name,
        d_model=args.d_model,
        layer=args.l_layers,
        divide=args.divide
    ).to(device)
    
    # 创建保存目录（添加版本标识）
    save_path = f"./Embeddings/{args.data_path}/{args.embed_version}/{args.divide}/"
    os.makedirs(save_path, exist_ok=True)
    print(f"Saving embeddings to: {save_path}")
    print(f"Embedding version: {args.embed_version}")
    
    # 生成并保存embeddings
    start_time = time.time()
    for i, (x, y, x_mark, y_mark) in enumerate(data_loader):
        try:
            embeddings = gen_prompt_emb.generate_embeddings(x.to(device), x_mark.to(device))
            
            file_path = os.path.join(save_path, f"{i}.h5")
            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('embeddings', data=embeddings.cpu().numpy())
            
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                print(f"Processed {i+1}/{len(data_loader)} samples, "
                      f"Time: {elapsed:.2f}s, Avg: {elapsed/(i+1):.3f}s/sample")
        
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    total_time = time.time() - start_time
    print(f"\nCompleted! Total time: {total_time/60:.2f} minutes")
    print(f"Average time per sample: {total_time/len(data_loader):.3f} seconds")


if __name__ == "__main__":
    args = parse_args()
    save_embeddings(args)

