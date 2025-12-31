import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np
from layers.StandardNorm import Normalize

class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)
        self.w3 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class Inception_Block(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=3):
        super(Inception_Block, self).__init__()
        self.kernels = [3, 5, 7]
        c_part = out_channels // num_kernels
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, c_part if i < num_kernels - 1 else out_channels - c_part * (num_kernels - 1),
                      kernel_size=k, padding=k // 2)
            for i, k in enumerate(self.kernels)
        ])
        self.bottleneck = nn.Conv1d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # x: [B, C, L]
        outputs = [conv(x) for conv in self.convs]
        out = torch.cat(outputs, dim=1)
        return self.bottleneck(out)

class AdaptiveDenoising(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.threshold = nn.Parameter(torch.ones(1, 1, channel) * 0.01)

    def forward(self, x):
        return torch.sign(x) * torch.relu(torch.abs(x) - self.threshold)

class GatedTransformerEncoderLayerPro(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.gate_proj = nn.Linear(d_model, d_model)
        self.swiglu = SwiGLU(d_model, dim_feedforward)
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
        x = x + self.dropout2(self.swiglu(nx))
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

class TriModalWaveletShapeProQwen(nn.Module):
    def __init__(self, device="cuda", channel=64, num_nodes=7, seq_len=96, pred_len=96,
                 dropout_n=0.1, d_llm=1024, e_layer=1, d_layer=1, wavelet='db4'):
        super().__init__()
        self.device, self.channel, self.num_nodes, self.seq_len, self.pred_len = device, channel, num_nodes, seq_len, pred_len
        
        self.normalize_layers = Normalize(num_nodes, affine=False).to(device)
        
        # 1. Time Path: Inception + Gated Transformer
        self.inception = Inception_Block(1, channel).to(device)
        self.ts_encoder = nn.ModuleList([GatedTransformerEncoderLayerPro(channel, 8, dim_feedforward=channel*2, dropout=dropout_n) for _ in range(e_layer)]).to(device)
        
        # 2. Wavelet Path: Adaptive Denoising + Gated Transformer
        self.wavelet_transform = WaveletTransform(wavelet=wavelet).to(device)
        self.denoising = AdaptiveDenoising(channel).to(device)
        self.wavelet_proj = nn.ModuleList([nn.Linear(1, channel).to(device) for _ in range(10)])
        self.wavelet_encoder = GatedTransformerEncoderLayerPro(channel, 8, dim_feedforward=channel*2, dropout=dropout_n).to(device)
        
        # 3. LLM Path
        self.prompt_encoder = nn.ModuleList([GatedTransformerEncoderLayerPro(d_llm, 8, dim_feedforward=d_llm*2, dropout=dropout_n) for _ in range(e_layer)]).to(device)
        
        # 4. Fusion & CMA
        from layers.Cross_Modal_Align import CrossModal
        self.cma = CrossModal(d_model=num_nodes, n_heads=1, d_ff=channel, n_layers=1).to(device)
        self.fusion_gate = nn.Linear(channel * 2, channel).to(device)
        
        # 5. Decoder
        self.decoder = nn.Linear(channel, pred_len).to(device)

    def forward(self, x, x_mark, embeddings):
        B, L, N = x.shape
        x = x.float()
        x_norm = self.normalize_layers(x, 'norm') # [B, L, N]
        
        # Time path
        # [B, L, N] -> [B*N, 1, L] -> Inception -> [B*N, C, L] -> Mean -> [B, N, C]
        t_in = x_norm.permute(0, 2, 1).reshape(B*N, 1, L)
        t_feat = self.inception(t_in).mean(dim=-1).reshape(B, N, self.channel)
        for layer in self.ts_encoder: t_feat = layer(t_feat)
        
        # Wavelet path
        wavelet_coeffs = self.wavelet_transform(x_norm.permute(0, 2, 1))
        w_feats = []
        for i, coeff in enumerate(wavelet_coeffs):
            B_sub, N_sub, Lp = coeff.shape
            tokens = self.wavelet_proj[min(i, 9)](coeff.unsqueeze(-1).reshape(B_sub*N_sub, Lp, 1))
            tokens = self.denoising(tokens)
            w_feats.append(self.wavelet_encoder(tokens).mean(dim=1).reshape(B, N, self.channel))
        w_feat = torch.stack(w_feats, dim=1).mean(dim=1)
        
        # Fusion
        fused = self.fusion_gate(torch.cat([t_feat, w_feat], dim=-1))
        
        # LLM Info
        p_feat = embeddings.float().squeeze(-1).permute(0, 2, 1)
        for layer in self.prompt_encoder: p_feat = layer(p_feat)
        
        # CMA
        aligned = self.cma(fused.permute(0, 2, 1), p_feat.permute(0, 2, 1), p_feat.permute(0, 2, 1))
        
        # Reconstruct
        out = self.decoder(aligned.permute(0, 2, 1)).permute(0, 2, 1)
        return self.normalize_layers(out, 'denorm')

