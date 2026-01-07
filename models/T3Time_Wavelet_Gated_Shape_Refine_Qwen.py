import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np
from layers.StandardNorm import Normalize

class SELayer(nn.Module):
    """
    轻量级通道注意力，用于增强变量间的关联
    """
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, N, C]
        b, n, c = x.size()
        y = self.avg_pool(x.transpose(1, 2)).view(b, c)
        y = self.fc(y).view(b, 1, c)
        return x * y.expand_as(x)

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

class WeightedWaveletFusion(nn.Module):
    def __init__(self, embed_dim, num_levels=6): # db4 on 96 seq usually has 4-6 levels
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )
        # 可学习的层级权重
        self.level_weights = nn.Parameter(torch.ones(10)) # 预留10层

    def forward(self, wavelet_features):
        pooled_list = []
        for feat in wavelet_features: # [B*N, Lp, C]
            weights = F.softmax(self.attention(feat), dim=1)
            pooled = (feat * weights).sum(dim=1) # [B*N, C]
            pooled_list.append(pooled)
        
        stacked = torch.stack(pooled_list, dim=1) # [B*N, num_levels, C]
        L = stacked.size(1)
        # 应用可学习的层级权重
        w = F.softmax(self.level_weights[:L], dim=0).view(1, L, 1)
        return (stacked * w).sum(dim=1)

class TriModalWaveletGatedShapeRefineQwen(nn.Module):
    def __init__(self, device="cuda", channel=64, num_nodes=7, seq_len=96, pred_len=96,
                 dropout_n=0.1, d_llm=1024, e_layer=1, d_layer=1, head=8, wavelet='db4'):
        super().__init__()
        self.device, self.channel, self.num_nodes, self.seq_len, self.pred_len = device, channel, num_nodes, seq_len, pred_len
        
        self.normalize_layers = Normalize(num_nodes, affine=False).to(device)
        self.ts_proj = nn.Linear(seq_len, channel).to(device)
        self.ts_encoder = nn.ModuleList([GatedTransformerEncoderLayer(channel, head, dropout=dropout_n) for _ in range(e_layer)]).to(device)
        
        self.wavelet_transform = WaveletTransform(wavelet=wavelet).to(device)
        self.wavelet_proj = nn.ModuleList([nn.Linear(1, channel).to(device) for _ in range(10)])
        self.wavelet_encoder = GatedTransformerEncoderLayer(channel, head, dropout=dropout_n).to(device)
        self.wavelet_fusion = WeightedWaveletFusion(channel).to(device)
        
        self.se_block = SELayer(channel).to(device)
        self.fusion_gate = nn.Linear(channel * 2, channel).to(device)
        
        self.prompt_encoder = nn.ModuleList([GatedTransformerEncoderLayer(d_llm, head, dropout=dropout_n) for _ in range(e_layer)]).to(device)
        
        from layers.Cross_Modal_Align import CrossModal
        self.cma = CrossModal(d_model=num_nodes, n_heads=1, d_ff=channel, n_layers=1).to(device)
        self.decoder = nn.Linear(channel, pred_len).to(device)

    def forward(self, x, x_mark, embeddings):
        B, L, N = x.shape
        x = x.float()
        x_norm = self.normalize_layers(x, 'norm') # [B, L, N]
        
        # 时域
        t_feat = self.ts_proj(x_norm.permute(0, 2, 1)) # [B, N, C]
        for layer in self.ts_encoder:
            t_feat = layer(t_feat)
        
        # 小波域 (带层级加权)
        wavelet_coeffs = self.wavelet_transform(x_norm.permute(0, 2, 1))
        w_feats_raw = []
        for i, coeff in enumerate(wavelet_coeffs):
            BN, Lp = coeff.size(0)*coeff.size(1), coeff.size(2)
            tokens = self.wavelet_proj[min(i, 9)](coeff.unsqueeze(-1).reshape(BN, Lp, 1))
            w_feats_raw.append(self.wavelet_encoder(tokens))
        w_feat = self.wavelet_fusion(w_feats_raw).reshape(B, N, self.channel)
        
        # 融合 + 通道注意力 (SE)
        fused = self.fusion_gate(torch.cat([t_feat, w_feat], dim=-1))
        fused = self.se_block(fused)
        
        # Prompt
        p_feat = embeddings.float().squeeze(-1).permute(0, 2, 1)
        for layer in self.prompt_encoder:
            p_feat = layer(p_feat)
        
        # Alignment
        aligned = self.cma(fused.permute(0, 2, 1), p_feat.permute(0, 2, 1), p_feat.permute(0, 2, 1))
        
        # Out
        out = self.decoder(aligned.permute(0, 2, 1)).permute(0, 2, 1)
        return self.normalize_layers(out, 'denorm')

