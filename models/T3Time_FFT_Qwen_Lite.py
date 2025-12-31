import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.StandardNorm import Normalize

class series_decomp(nn.Module):
    """
    序列分解模块：将序列分解为趋势（Trend）和季节性（Seasonal/Residual）
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2)

    def forward(self, x):
        # x: [B, L, N]
        # moving_mean 计算趋势
        moving_mean = self.moving_avg(x.permute(0, 2, 1)).permute(0, 2, 1)
        res = x - moving_mean
        return res, moving_mean

class TriModalFFTQwenLite(nn.Module):
    """
    T3Time_FFT_Qwen Lite 版本:
    - 借鉴 DLinear: 显式处理趋势项 (Trend)
    - 借鉴 T3Time: 频域 (FFT) + Qwen 嵌入 (Prompt) 引导
    - 结构: 纯线性 + 门控，无 Transformer 模块，极速且轻量
    """
    def __init__(self, device="cuda", channel=64, num_nodes=7, seq_len=96, pred_len=96, d_llm=1024, dropout=0.1):
        super().__init__()
        self.device = device
        self.channel = channel
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.pred_len = pred_len

        # 1. 归一化层 (RevIN)
        self.normalize_layers = Normalize(self.num_nodes, affine=False).to(self.device)

        # 2. 序列分解 (核大小 25 是 DLinear 的常用默认值)
        self.decomp = series_decomp(25)
        
        # 3. 趋势分支 (Trend): 独立的线性层处理
        self.trend_model = nn.Linear(seq_len, pred_len).to(self.device)

        # 4. 季节性/频域分支 (Seasonal/Spectral)
        # 时域投影
        self.seasonal_proj = nn.Linear(seq_len, channel).to(self.device)
        # 频域投影 (FFT 后的幅度谱)
        self.freq_proj = nn.Linear(seq_len // 2 + 1, channel).to(self.device)
        # Prompt (Qwen) 投影
        self.prompt_proj = nn.Linear(d_llm, channel).to(self.device)
        
        # 5. 轻量化门控融合 (Gated Fusion)
        # 通过学习一个门控权重来融合 时域、频域和 Prompt 信息
        self.fusion_gate = nn.Sequential(
            nn.Linear(channel * 3, channel),
            nn.Sigmoid()
        ).to(self.device)
        
        # 6. 输出投影
        self.out_proj = nn.Linear(channel, pred_len).to(self.device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mark, embeddings):
        # x: [B, L, N]
        input_x = x.float()
        embeddings = embeddings.float()
        
        # --- A. 归一化 ---
        input_x = self.normalize_layers(input_x, 'norm') # [B, L, N]
        
        # --- B. 分解 ---
        seasonal_init, trend_init = self.decomp(input_x)

        # --- C. 趋势分支 ---
        # trend_init: [B, L, N] -> [B, N, L] -> Linear -> [B, N, P]
        trend_out = self.trend_model(trend_init.permute(0, 2, 1)).permute(0, 2, 1) # [B, P, N]

        # --- D. 季节性/频域分支 (Qwen 引导) ---
        # 1. 频域特征 (FFT)
        # seasonal_init: [B, L, N] -> [B, N, L] -> rfft -> [B, N, Lf]
        freq_complex = torch.fft.rfft(seasonal_init.permute(0, 2, 1), dim=-1)
        freq_mag = torch.abs(freq_complex) 
        freq_feat = F.gelu(self.freq_proj(freq_mag)) # [B, N, C]

        # 2. 季节性时域投影
        seasonal_feat = F.gelu(self.seasonal_proj(seasonal_init.permute(0, 2, 1))) # [B, N, C]

        # 3. Prompt 特征 (Qwen)
        # embeddings: [B, E, N, 1] -> [B, N, E]
        prompt_feat = embeddings.squeeze(-1).permute(0, 2, 1) 
        prompt_feat = F.gelu(self.prompt_proj(prompt_feat)) # [B, N, C]

        # 4. 融合 (Gated Fusion)
        combined = torch.cat([seasonal_feat, freq_feat, prompt_feat], dim=-1) # [B, N, 3C]
        gate = self.fusion_gate(combined) # [B, N, C]
        
        # 核心逻辑：用学习到的门控权重融合特征
        fused = seasonal_feat * gate + freq_feat * (1 - gate) + prompt_feat * 0.1
        
        # 5. 输出季节性预测
        seasonal_out = self.out_proj(self.dropout(fused)).permute(0, 2, 1) # [B, P, N]

        # --- E. 最终合并与反归一化 ---
        out = trend_out + seasonal_out
        out = self.normalize_layers(out, 'denorm')
        
        return out

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

