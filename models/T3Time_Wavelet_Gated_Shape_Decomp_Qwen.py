import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np
from layers.StandardNorm import Normalize

class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.moving_avg.kernel_size[0] - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.moving_avg.kernel_size[0]) // 2, 1)
        x_padded = torch.cat([front, x, end], dim=1)
        res = self.moving_avg(x_padded.permute(0, 2, 1))
        res = res.permute(0, 2, 1)
        moving_mean = res
        res = x - moving_mean
        return moving_mean, res

class GatedTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.gate_proj = nn.Linear(d_model, d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        nx = self.norm1(src)
        attn_output, _ = self.self_attn(nx, nx, nx)
        gate = torch.sigmoid(self.gate_proj(nx))
        x = src + self.dropout1(attn_output * gate)
        nx = self.norm2(x)
        ff_output = self.linear2(self.dropout(F.gelu(self.linear1(nx))))
        x = x + self.dropout2(ff_output)
        return x

class WaveletTransform(nn.Module):
    def __init__(self, wavelet='db4'):
        super().__init__()
        self.wavelet = wavelet
        wavelet_obj = pywt.Wavelet(wavelet)
        self.register_buffer('dec_lo', torch.tensor(wavelet_obj.dec_lo, dtype=torch.float32).view(1, 1, -1))
        self.register_buffer('dec_hi', torch.tensor(wavelet_obj.dec_hi, dtype=torch.float32).view(1, 1, -1))
    
    def _dwt_step(self, x):
        B, N, L = x.shape
        pad_len = self.dec_lo.shape[-1] - 1
        x_padded = F.pad(x.reshape(B*N, 1, -1), (pad_len, pad_len), mode='circular')
        cA = F.conv1d(x_padded, self.dec_lo, stride=2)
        cD = F.conv1d(x_padded, self.dec_hi, stride=2)
        return cA.reshape(B, N, -1), cD.reshape(B, N, -1)
    
    def forward(self, x):
        level = pywt.dwt_max_level(x.shape[-1], self.wavelet)
        coeffs = []
        current = x
        for _ in range(level):
            cA, cD = self._dwt_step(current)
            coeffs.insert(0, cD)
            current = cA
        coeffs.insert(0, cA)
        return coeffs

class TriModalWaveletGatedShapeDecompQwen(nn.Module):
    def __init__(self, device="cuda", channel=64, num_nodes=7, seq_len=96, pred_len=96,
                 dropout_n=0.1, d_llm=1024, e_layer=1, d_layer=1, wavelet='db4'):
        super().__init__()
        self.device, self.channel, self.num_nodes, self.seq_len, self.pred_len = device, channel, num_nodes, seq_len, pred_len
        
        self.normalize_layers = Normalize(num_nodes, affine=False).to(device)
        self.decomp = series_decomp(25).to(device)
        
        # 趋势预测 (极简线性)
        self.trend_model = nn.Linear(seq_len, pred_len).to(device)
        
        # 残差预测 (Gated + Wavelet + LLM)
        self.wavelet_transform = WaveletTransform(wavelet=wavelet).to(device)
        self.wavelet_proj = nn.ModuleList([nn.Linear(1, channel).to(device) for _ in range(10)])
        self.wavelet_encoder = GatedTransformerEncoderLayer(channel, 8, dropout=dropout_n).to(device)
        
        self.ts_proj = nn.Linear(seq_len, channel).to(device)
        self.ts_encoder = GatedTransformerEncoderLayer(channel, 8, dropout=dropout_n).to(device)
        
        self.prompt_encoder = GatedTransformerEncoderLayer(d_llm, 8, dropout=dropout_n).to(device)
        
        from layers.Cross_Modal_Align import CrossModal
        self.cma = CrossModal(d_model=num_nodes, n_heads=1, d_ff=channel, n_layers=1).to(device)
        self.decoder = nn.Linear(channel, pred_len).to(device)

    def forward(self, x, x_mark, embeddings):
        B, L, N = x.shape
        x = x.float()
        x_norm = self.normalize_layers(x, 'norm')
        
        # 1. 分解
        trend, seasonal = self.decomp(x_norm)
        
        # 2. 趋势路径
        trend_pred = self.trend_model(trend.permute(0, 2, 1)).permute(0, 2, 1)
        
        # 3. 残差路径
        # 时域
        s_feat = self.ts_proj(seasonal.permute(0, 2, 1)) # [B, N, C]
        s_feat = self.ts_encoder(s_feat)
        
        # 小波域
        wavelet_coeffs = self.wavelet_transform(seasonal.permute(0, 2, 1))
        w_feats = []
        for i, coeff in enumerate(wavelet_coeffs):
            B_s, N_s, Lp = coeff.shape
            tokens = self.wavelet_proj[min(i, 9)](coeff.unsqueeze(-1).reshape(B_s*N_s, Lp, 1))
            w_feats.append(self.wavelet_encoder(tokens).mean(dim=1).reshape(B, N, self.channel))
        w_feat = torch.stack(w_feats, dim=1).mean(dim=1)
        
        # 融合与对齐
        p_feat = self.prompt_encoder(embeddings.float().squeeze(-1).permute(0, 2, 1))
        aligned = self.cma((s_feat + w_feat).permute(0, 2, 1), p_feat.permute(0, 2, 1), p_feat.permute(0, 2, 1))
        
        seasonal_pred = self.decoder(aligned.permute(0, 2, 1)).permute(0, 2, 1)
        
        # 4. 重构
        out = trend_pred + seasonal_pred
        return self.normalize_layers(out, 'denorm')

