import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Cross_Modal_Align import CrossModal
from .T3Time_Wavelet_Gated_Shape_Qwen import GatedTransformerEncoderLayer

class LearnableLiftingLayer(nn.Module):
    """
    可学习的提升方案小波层 (Lifting Scheme)
    实现: Split -> Predict -> Update
    """
    def __init__(self, channel):
        super().__init__()
        # 预测算子 P: 用偶数点预测奇数点，得到细节分量 (High-freq)
        self.P = nn.Sequential(
            nn.Conv1d(channel, channel, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(channel, channel, kernel_size=3, padding=1)
        )
        # 更新算子 U: 用细节分量更新偶数点，得到近似分量 (Low-freq)
        self.U = nn.Sequential(
            nn.Conv1d(channel, channel, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(channel, channel, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # x: [B, C, L]
        # 1. Split (分裂)
        xe = x[:, :, 0::2] # 偶数位置
        xo = x[:, :, 1::2] # 奇数位置
        
        # 2. Predict (预测)
        # d = xo - P(xe)
        # 这里的 d 捕捉的是信号的突变/高频细节
        d = xo - self.P(xe)
        
        # 3. Update (更新)
        # s = xe + U(d)
        # 这里的 s 保持了信号的均值/低频轮廓
        s = xe + self.U(d)
        
        return s, d # 返回近似分量和细节分量

class TriModalLearnableWaveletGatedShapeQwen(nn.Module):
    def __init__(self, device, channel=64, d_ff=None, num_nodes=7, seq_len=96, pred_len=96, 
                 dropout_n=0.2, d_llm=1024, e_layer=1, levels=3):
        super().__init__()
        self.device = device
        self.channel = channel
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_ff = d_ff or 4 * channel
        self.levels = levels
        
        # 1. 嵌入层
        self.ts_embed = nn.Linear(num_nodes, channel)
        # 针对可学习小波的初始嵌入
        self.wavelet_init_embed = nn.Linear(num_nodes, channel)

        # 2. 可学习小波分支 (Lifting Tree)
        self.lifting_layers = nn.ModuleList([
            LearnableLiftingLayer(channel) for _ in range(levels)
        ])
        
        # 3. 编码器
        self.ts_encoder = nn.ModuleList([
            GatedTransformerEncoderLayer(channel, nhead=8, dim_feedforward=self.d_ff, dropout=dropout_n)
            for _ in range(e_layer)
        ])
        
        self.wavelet_encoder = GatedTransformerEncoderLayer(channel, nhead=8, dim_feedforward=self.d_ff, dropout=dropout_n)
        
        self.prompt_encoder = nn.ModuleList([
            GatedTransformerEncoderLayer(d_llm, nhead=8, dim_feedforward=d_llm*2, dropout=dropout_n)
            for _ in range(e_layer)
        ])

        # 4. 融合与输出
        self.cma = CrossModal(
            d_model=channel, n_heads=1, d_k=channel, d_v=channel, 
            d_ff=self.d_ff, dropout=dropout_n
        )
        self.out_proj = nn.Linear(channel, num_nodes)
        self.proj_to_pred = nn.Linear(seq_len, pred_len)

    def forward(self, x, x_mark, embeddings):
        B, L, N = x.shape
        # 标准化
        means = x.mean(1, keepdim=True)
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x /= stdev

        # --- Prompt 处理 ---
        prompt_feat = embeddings.float().squeeze(-1).permute(0, 2, 1) # [B, N, d_llm]
        for layer in self.prompt_encoder:
            prompt_feat = layer(prompt_feat)

        # --- 时间域编码 ---
        ts_feat = self.ts_embed(x)
        for layer in self.ts_encoder:
            ts_feat = layer(ts_feat)

        # --- 可学习小波编码 (关键改进) ---
        # w_feat: [B, L, C]
        w_feat = self.wavelet_init_embed(x).permute(0, 2, 1) # [B, C, L]
        
        current_s = w_feat
        all_details = []
        
        # 多尺度分解
        for lifting in self.lifting_layers:
            # 如果长度是奇数，先填充
            if current_s.size(-1) % 2 != 0:
                current_s = F.pad(current_s, (0, 1), mode='replicate')
            
            current_s, d = lifting(current_s)
            all_details.append(d)
        
        # 将各尺度的细节分量和最后的近似分量对齐到原长度并聚合
        # 这相当于一种“特征金字塔”式的频域聚合
        aggregated_wavelet = F.interpolate(current_s, size=L, mode='linear', align_corners=False)
        for d in all_details:
            d_interp = F.interpolate(d, size=L, mode='linear', align_corners=False)
            aggregated_wavelet += d_interp
            
        wavelet_encoded = self.wavelet_encoder(aggregated_wavelet.permute(0, 2, 1))

        # --- 跨模态融合 ---
        fused = self.cma(ts_feat, wavelet_encoded, wavelet_encoded)
        
        # --- 输出 ---
        out = self.out_proj(fused).permute(0, 2, 1)
        out = self.proj_to_pred(out).permute(0, 2, 1)
        
        # 反标准化
        out = out * stdev + means
        return out

