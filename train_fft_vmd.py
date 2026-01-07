"""
训练 FFT + VMD 并行融合模型
保留原有的 FFT 频域分支，新增 VMD 作为辅助特征
"""
import os
import time
import random
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from data_provider.data_loader_vmd import Dataset_ETT_hour_VMD
from models.T3Time_FFT_VMD import TriModalFFTVMD
from utils.metrics import MSE, MAE, metric
from utils.experiment_logger import log_experiment_result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda", help="")
    parser.add_argument("--data_path", type=str, default="ETTh1", help="data path")
    parser.add_argument("--num_nodes", type=int, default=7, help="number of nodes")
    parser.add_argument("--seq_len", type=int, default=96, help="seq_len")
    parser.add_argument("--pred_len", type=int, default=96, help="out_len")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
    # 为了防止仅保留 VMD 分支后过拟合过强，默认略微增大 dropout
    parser.add_argument("--dropout_n", type=float, default=0.3, help="dropout rate")
    parser.add_argument("--d_llm", type=int, default=768, help="hidden dimensions")
    parser.add_argument("--e_layer", type=int, default=1, help="layers of transformer encoder")
    parser.add_argument("--d_layer", type=int, default=1, help="layers of transformer decoder")
    parser.add_argument("--head", type=int, default=8, help="heads of attention")
    parser.add_argument("--channel", type=int, default=32, help="number of features")
    parser.add_argument("--num_cma_heads", type=int, default=4, help="number of CMA heads")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="weight decay rate")
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=150, help="")
    parser.add_argument("--seed", type=int, default=2024, help="random seed")
    parser.add_argument("--es_patience", type=int, default=25, help="early stopping patience")
    parser.add_argument("--save", type=str, default="./logs_fft_vmd/", help="save path")
    parser.add_argument("--embed_version", type=str, default="original", help="嵌入版本")
    parser.add_argument("--vmd_k", type=int, default=5, help="VMD 模态数 K")
    parser.add_argument("--vmd_alpha", type=float, default=2000.0, help="VMD alpha 参数")
    parser.add_argument("--vmd_root", type=str, default="./vmd_cache", help="VMD 缓存根目录")
    parser.add_argument("--overfit_test", action="store_true", help="过拟合测试模式：只用一个 batch，训练 1000 个 epoch")
    parser.add_argument("--zero_vmd_test", action="store_true", help="信息量分析：将 VMD 输入设为 0，测试 VMD 分支是否真的有用")
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
        num_cma_heads,
        vmd_modes,
        lrate,
        wdecay,
        device,
        epochs,
    ):
        self.model = TriModalFFTVMD(
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
            num_cma_heads=num_cma_heads,
            vmd_modes=vmd_modes,
        )
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
                self.model.count_trainable_params()
            )
        )
        print("The number of parameters: {}".format(self.model.param_num()))

    def train(self, input, mark, embeddings, real, x_modes):
        self.model.train()
        self.optimizer.zero_grad()
        input = input.float()
        real = real.float()
        predict = self.model(input, mark, embeddings, x_modes)
        loss = self.loss(predict, real)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = self.MAE(predict, real)
        return loss.item(), mae.item()

    def eval(self, input, mark, embeddings, real_val, x_modes):
        self.model.eval()
        with torch.no_grad():
            input = input.float()
            real_val = real_val.float()
            predict = self.model(input, mark, embeddings, x_modes)
        loss = self.loss(predict, real_val)
        mae = self.MAE(predict, real_val)
        return loss.item(), mae.item()


def load_data(args):
    train_set = Dataset_ETT_hour_VMD(
        flag="train",
        scale=True,
        size=[args.seq_len, 0, args.pred_len],
        data_path=args.data_path,
        num_nodes=args.num_nodes,
        embed_version=args.embed_version,
        vmd_k=args.vmd_k,
        vmd_alpha=args.vmd_alpha,
        vmd_root=args.vmd_root,
    )
    val_set = Dataset_ETT_hour_VMD(
        flag="val",
        scale=True,
        size=[args.seq_len, 0, args.pred_len],
        data_path=args.data_path,
        num_nodes=args.num_nodes,
        embed_version=args.embed_version,
        vmd_k=args.vmd_k,
        vmd_alpha=args.vmd_alpha,
        vmd_root=args.vmd_root,
    )
    test_set = Dataset_ETT_hour_VMD(
        flag="test",
        scale=True,
        size=[args.seq_len, 0, args.pred_len],
        data_path=args.data_path,
        num_nodes=args.num_nodes,
        embed_version=args.embed_version,
        vmd_k=args.vmd_k,
        vmd_alpha=args.vmd_alpha,
        vmd_root=args.vmd_root,
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
    best_test_mae = float("inf")
    epochs_since_best_mse = 0
    best_model_state = None  # 用于保存最佳模型状态（不保存文件时使用）

    save_dir = os.path.join(
        args.save,
        args.data_path,
        f"fft_vmd_i{args.seq_len}_o{args.pred_len}_c{args.channel}_"
        f"k{args.vmd_k}_a{int(args.vmd_alpha)}_lr{args.learning_rate}_"
        f"dn{args.dropout_n}_bs{args.batch_size}_seed{args.seed}/",
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
            channel=args.channel,
            num_nodes=args.num_nodes,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            dropout_n=args.dropout_n,
            d_llm=args.d_llm,
            e_layer=args.e_layer,
            d_layer=args.d_layer,
            head=args.head,
            num_cma_heads=args.num_cma_heads,
            vmd_modes=args.vmd_k,
            lrate=args.learning_rate,
            wdecay=args.weight_decay,
            device=device,
            epochs=args.epochs,
        )

        log("Start training FFT + VMD parallel fusion model...", flush=True)

        # ========== 过拟合测试模式 ==========
        if args.overfit_test:
            log("=" * 60, flush=True)
            log("过拟合测试模式：只用一个 batch，训练 1000 个 epoch", flush=True)
            log("预期：Training Loss 必须降到几乎为 0 (MSE < 0.00001)", flush=True)
            log("=" * 60, flush=True)
            
            # 只取第一个 batch
            first_batch = next(iter(train_loader))
            x, y, x_mark, y_mark, embeddings, x_modes = first_batch
            
            trainx = torch.Tensor(x).to(device)
            trainy = torch.Tensor(y).to(device)
            trainx_mark = torch.Tensor(x_mark).to(device)
            train_embedding = torch.Tensor(embeddings).to(device)
            train_x_modes = torch.Tensor(x_modes).to(device)
            
            # 信息量分析：将 VMD 输入设为 0
            if args.zero_vmd_test:
                log("=" * 60, flush=True)
                log("信息量分析：将 VMD 输入设为 0", flush=True)
                log("如果效果没变差，说明 VMD 分支是纯噪声", flush=True)
                log("=" * 60, flush=True)
                train_x_modes = torch.zeros_like(train_x_modes)
            
            log(f"过拟合测试数据形状: input={trainx.shape}, target={trainy.shape}, x_modes={train_x_modes.shape}", flush=True)
            log(f"开始训练 1000 个 epoch...", flush=True)
            
            # 训练 1000 个 epoch
            for epoch in range(1, 1001):
                metrics = engine.train(
                    trainx, trainx_mark, train_embedding, trainy, train_x_modes
                )
                train_loss, train_mae = metrics
                
                # 每 10 个 epoch 打印一次，最后 100 个 epoch 每 1 个打印一次
                if epoch % 10 == 0 or epoch > 900:
                    log(f"Epoch: {epoch:04d}, Train Loss: {train_loss:.8f}, Train MAE: {train_mae:.8f}", flush=True)
                    
                    # 梯度检查和门控权重检查（每 100 个 epoch 检查一次）
                    if epoch % 100 == 0:
                        log("=" * 60, flush=True)
                        log("梯度流检查:", flush=True)
                        grad_stats = {}
                        for name, param in engine.model.named_parameters():
                            if param.grad is not None:
                                grad_mean = param.grad.abs().mean().item()
                                grad_max = param.grad.abs().max().item()
                                grad_stats[name] = (grad_mean, grad_max)
                        
                        # 检查 VMD 分支的梯度
                        vmd_grads = {k: v for k, v in grad_stats.items() if 'vmd' in k.lower()}
                        fft_grads = {k: v for k, v in grad_stats.items() if 'frequency' in k.lower() or 'fft' in k.lower()}
                        time_grads = {k: v for k, v in grad_stats.items() if 'time' in k.lower() and 'vmd' not in k.lower()}
                        
                        if vmd_grads:
                            log("VMD 分支梯度统计:", flush=True)
                            for name, (mean, max_val) in list(vmd_grads.items())[:5]:  # 只显示前5个
                                log(f"  {name}: mean={mean:.6e}, max={max_val:.6e}", flush=True)
                            vmd_mean_grad = np.mean([v[0] for v in vmd_grads.values()])
                            log(f"  VMD 平均梯度: {vmd_mean_grad:.6e}", flush=True)
                        
                        if fft_grads:
                            fft_mean_grad = np.mean([v[0] for v in fft_grads.values()])
                            log(f"FFT 分支平均梯度: {fft_mean_grad:.6e}", flush=True)
                        
                        if time_grads:
                            time_mean_grad = np.mean([v[0] for v in time_grads.values()])
                            log(f"时域分支平均梯度: {time_mean_grad:.6e}", flush=True)
                        
                        # 诊断
                        if vmd_grads:
                            vmd_mean_grad = np.mean([v[0] for v in vmd_grads.values()])
                            if vmd_mean_grad < 1e-6:
                                log("✗ 警告：VMD 分支梯度接近 0，可能被门控机制屏蔽", flush=True)
                            else:
                                log("✓ VMD 分支梯度正常", flush=True)
                        
                        # ========== 门控权重检查 ==========
                        log("", flush=True)
                        log("门控权重检查:", flush=True)
                        engine.model.eval()
                        with torch.no_grad():
                            # 前向传播获取门控值
                            predict = engine.model(trainx, trainx_mark, train_embedding, train_x_modes)
                            
                            # 手动计算门控值（需要访问中间结果）
                            # 由于无法直接访问，我们需要修改模型来返回门控值
                            # 或者通过 hook 来获取
                            gate_values = []
                            
                            def gate_hook(module, input, output):
                                gate_values.append(output.detach().cpu())
                            
                            # 注册 hook
                            handle = engine.model.rich_horizon_gate.register_forward_hook(gate_hook)
                            
                            # 重新前向传播
                            _ = engine.model(trainx, trainx_mark, train_embedding, train_x_modes)
                            
                            # 移除 hook
                            handle.remove()
                            
                            if gate_values:
                                gate = gate_values[0]  # [B, C, 1]
                                gate_mean = gate.mean().item()
                                gate_min = gate.min().item()
                                gate_max = gate.max().item()
                                gate_std = gate.std().item()
                                
                                log(f"  门控值统计: mean={gate_mean:.4f}, min={gate_min:.4f}, max={gate_max:.4f}, std={gate_std:.4f}", flush=True)
                                
                                # 统计接近 0 和接近 1 的通道比例
                                gate_flat = gate.flatten()
                                near_zero = (gate_flat < 0.1).float().mean().item() * 100
                                near_one = (gate_flat > 0.9).float().mean().item() * 100
                                balanced = ((gate_flat >= 0.3) & (gate_flat <= 0.7)).float().mean().item() * 100
                                
                                log(f"  门控值分布: <0.1={near_zero:.1f}%, >0.9={near_one:.1f}%, [0.3,0.7]={balanced:.1f}%", flush=True)
                                
                                # 诊断
                                if gate_mean < 0.1:
                                    log("  ✗ 警告：门控值整体接近 0，频域特征（FFT+VMD）被严重屏蔽", flush=True)
                                    log("    说明模型更依赖时域特征，VMD 分支可能无效", flush=True)
                                elif gate_mean > 0.9:
                                    log("  ✓ 门控值接近 1，模型更依赖频域特征", flush=True)
                                else:
                                    log(f"  ✓ 门控值平衡，模型同时使用时域和频域特征（均值={gate_mean:.3f}）", flush=True)
                        
                        # ========== FFT 和 VMD 分支输出统计 ==========
                        log("", flush=True)
                        log("FFT 和 VMD 分支输出统计:", flush=True)
                        engine.model.train()  # 切换回训练模式以获取调试信息
                        with torch.no_grad():
                            _ = engine.model(trainx, trainx_mark, train_embedding, train_x_modes)
                            
                            if hasattr(engine.model, '_debug_freq_enc'):
                                freq_enc = engine.model._debug_freq_enc
                                vmd_enc = engine.model._debug_vmd_enc
                                freq_vmd_fused = engine.model._debug_freq_vmd_fused
                                
                                log(f"  FFT 分支输出: mean={freq_enc.mean().item():.4f}, std={freq_enc.std().item():.4f}, norm={freq_enc.norm().item():.4f}", flush=True)
                                log(f"  VMD 分支输出: mean={vmd_enc.mean().item():.4f}, std={vmd_enc.std().item():.4f}, norm={vmd_enc.norm().item():.4f}", flush=True)
                                log(f"  融合后输出: mean={freq_vmd_fused.mean().item():.4f}, std={freq_vmd_fused.std().item():.4f}, norm={freq_vmd_fused.norm().item():.4f}", flush=True)
                                
                                # 计算 FFT 和 VMD 的相似度（余弦相似度）
                                freq_flat = freq_enc.flatten(0, 1)  # [B*C, N]
                                vmd_flat = vmd_enc.flatten(0, 1)   # [B*C, N]
                                cosine_sim = F.cosine_similarity(freq_flat, vmd_flat, dim=1).mean().item()
                                log(f"  FFT-VMD 余弦相似度: {cosine_sim:.4f}", flush=True)
                                
                                # 诊断
                                if abs(freq_enc.norm().item() - vmd_enc.norm().item()) > 10:
                                    log("  ⚠ 警告：FFT 和 VMD 分支输出幅度差异较大，可能导致融合不平衡", flush=True)
                                if cosine_sim > 0.9:
                                    log("  ⚠ 警告：FFT 和 VMD 分支输出高度相似，VMD 可能没有提供额外信息", flush=True)
                                elif cosine_sim < 0.1:
                                    log("  ✓ FFT 和 VMD 分支输出差异较大，VMD 提供了不同信息", flush=True)
                        
                        engine.model.train()
                        log("=" * 60, flush=True)
                    
                    # 如果 loss 降到足够低，提前结束
                    if train_loss < 0.00001:
                        log(f"✓ 成功！Loss 已降到 {train_loss:.8f} < 0.00001", flush=True)
                        log("诊断：模型能够过拟合，说明代码结构正确，梯度流正常", flush=True)
                        break
            
            # 最终评估
            log("=" * 60, flush=True)
            log("过拟合测试最终结果:", flush=True)
            log(f"最终 Train Loss: {train_loss:.8f}", flush=True)
            log(f"最终 Train MAE: {train_mae:.8f}", flush=True)
            
            if train_loss < 0.00001:
                log("✓ 诊断：模型能够过拟合到接近 0，代码结构正确", flush=True)
                log("  如果全量数据效果差，可能是特征分布不一致（VMD 端点效应）", flush=True)
            else:
                log("✗ 诊断：模型无法过拟合，可能原因：", flush=True)
                log("  1. 梯度断了（VMD 预处理没接上梯度）", flush=True)
                log("  2. 模型容量太小", flush=True)
                log("  3. 归一化没做好", flush=True)
                log("  4. 代码逻辑错误", flush=True)
            log("=" * 60, flush=True)
            return

        # ========== 正常训练模式 ==========
        for epoch in range(1, args.epochs + 1):
            t1 = time.time()
            train_loss = []
            train_mae = []

            for batch in train_loader:
                x, y, x_mark, y_mark, embeddings, x_modes = batch

                trainx = torch.Tensor(x).to(device)
                trainy = torch.Tensor(y).to(device)
                trainx_mark = torch.Tensor(x_mark).to(device)
                train_embedding = torch.Tensor(embeddings).to(device)
                train_x_modes = torch.Tensor(x_modes).to(device)

                metrics = engine.train(
                    trainx, trainx_mark, train_embedding, trainy, train_x_modes
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
                x, y, x_mark, y_mark, embeddings, x_modes = batch

                valx = torch.Tensor(x).to(device)
                valy = torch.Tensor(y).to(device)
                valx_mark = torch.Tensor(x_mark).to(device)
                val_embedding = torch.Tensor(embeddings).to(device)
                val_x_modes = torch.Tensor(x_modes).to(device)

                metrics = engine.eval(
                    valx, valx_mark, val_embedding, valy, val_x_modes
                )
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
                best_model_state = engine.model.state_dict().copy()  # 保存最佳模型状态到内存
                if args.save_model:
                    torch.save(
                        engine.model.state_dict(), os.path.join(save_dir, "best_model.pth")
                    )

                # 在验证集提升时跑一遍测试集
                test_outputs = []
                test_y_all = []
                for batch in test_loader:
                    x, y, x_mark, y_mark, embeddings, x_modes = batch

                    testx = torch.Tensor(x).to(device).float()
                    testy = torch.Tensor(y).to(device).float()
                    testx_mark = torch.Tensor(x_mark).to(device).float()
                    test_embedding = torch.Tensor(embeddings).to(device).float()
                    test_x_modes = torch.Tensor(x_modes).to(device).float()

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

                test_log = float(np.mean(amse))
                best_test_mae = float(np.mean(amae))
                log(f"Test MSE: {test_log:.4f}, Test MAE: {best_test_mae:.4f}")
            else:
                epochs_since_best_mse += 1
                if epochs_since_best_mse >= args.es_patience:
                    log("Early stopping triggered.")
                    break

            # scheduler
            engine.scheduler.step()

        log("Training finished.")
        # 训练结束后再汇总打印一次最佳测试集指标，方便查看
        if test_log < float("inf"):
            log(
                f"[Summary] Best Test MSE: {test_log:.4f}, Best Test MAE: {best_test_mae:.4f}"
            )
            # 记录实验结果到统一日志文件
            log_experiment_result(
                data_path=args.data_path,
                pred_len=args.pred_len,
                model_name="T3Time_FFT_VMD",
                seed=args.seed,
                test_mse=test_log,
                test_mae=best_test_mae,
                embed_version=args.embed_version,
                seq_len=args.seq_len,
                channel=args.channel,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                dropout_n=args.dropout_n,
                additional_info={"vmd_k": args.vmd_k, "vmd_alpha": args.vmd_alpha}
            )
    finally:
        if LOG_F is not None:
            LOG_F.close()


if __name__ == "__main__":
    main()

