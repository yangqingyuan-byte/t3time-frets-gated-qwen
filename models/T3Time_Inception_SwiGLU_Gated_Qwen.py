"""
T3Time with Inception Module, SwiGLU FFN, Gated Attention, and Wavelet Transform.
核心改进：
1. Inception Module: 在编码前提取 3, 5, 7 多尺度卷积特征。
2. SwiGLU: 将 Transformer 的 FFN 替换为门控线性单元（Llama 架构）。
3. 结合 NIPS 2025 Gated Attention 与 Qwen 嵌入。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np
from layers.StandardNorm import Normalize
from layers.Cross_Modal_Align import CrossModal

class SwiGLU(nn.Module):
    """ SwiGLU 激活函数模块 (Llama 风格) """
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_model, d_ff)
        self.w3 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # Swish(x * W1) * (x * W2) * W3
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class Inception_Block(nn.Module):
    """ 多尺度卷积模块 """
    def __init__(self, in_channels, out_channels, kernels=[3, 5, 7]):
        super().__init__()
        # 确保通道数分配均匀，最后一组承接余数以匹配 out_channels
        self.n_kernels = len(kernels)
        self.out_per_kernel = out_channels // self.n_kernels
        self.convs = nn.ModuleList()
        for i in range(self.n_kernels):
            # 最后一个分支处理余数
            c_out = self.out_per_kernel if i < self.n_kernels - 1 else out_channels - (self.out_per_kernel * (self.n_kernels - 1))
            self.convs.append(nn.Conv1d(in_channels, c_out, kernel_size=kernels[i], padding=kernels[i]//2))
        self.bottleneck = nn.Conv1d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # x: [B, C, L]
        res = [conv(x) for conv in self.convs]
        res = torch.cat(res, dim=1) # 拼接后的维度总和必然等于 out_channels
        return self.bottleneck(res)

class GatedSwiGLUTransformerLayer(nn.Module):
    """ 集成了 NIPS 2025 Gated Attention 和 SwiGLU FFN 的编码层 """
    def __init__(self, d_model, nhead, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.gate_proj = nn.Linear(d_model, d_model)
        
        # 替换为 SwiGLU
        self.swiglu = SwiGLU(d_model, d_ff)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        # Attention + Gating
        nx = self.norm1(src)
        attn_out, _ = self.self_attn(nx, nx, nx)
        gate = torch.sigmoid(self.gate_proj(nx))
        src = src + self.dropout1(attn_out * gate)
        
        # SwiGLU FFN
        nx = self.norm2(src)
        ff_out = self.swiglu(nx)
        src = src + self.dropout2(ff_out)
        return src

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

class TriModalInceptionSwiGLUGatedQwen(nn.Module):
    def __init__(
        self, device="cuda", channel=32, num_nodes=7, seq_len=96, pred_len=96,
        dropout_n=0.1, d_llm=1024, e_layer=1, d_layer=1, d_ff=128, head=8, wavelet='db4'
    ):
        super().__init__()
        self.device, self.channel, self.num_nodes, self.seq_len, self.pred_len = device, channel, num_nodes, seq_len, pred_len
        
        self.normalize_layers = Normalize(self.num_nodes, affine=False).to(self.device)
        
        # 1. Inception 模块提取局部特征
        self.inception = Inception_Block(self.num_nodes, self.channel).to(self.device)
        # 将池化后的序列长度映射到 channel 维度
        self.length_to_feature = nn.Linear(self.seq_len, self.channel).to(self.device)
        
        # 2. 增强型编码器 (Gated Attention + SwiGLU)
        self.ts_encoder = nn.ModuleList([
            GatedSwiGLUTransformerLayer(self.channel, head, d_ff=d_ff, dropout=dropout_n) 
            for _ in range(e_layer)
        ]).to(self.device)
        
        self.wavelet_transform = WaveletTransform(wavelet=wavelet).to(self.device)
        self.wavelet_proj = nn.ModuleList([nn.Linear(1, self.channel).to(self.device) for _ in range(10)])
        self.wavelet_encoder = GatedSwiGLUTransformerLayer(self.channel, head, d_ff=d_ff, dropout=dropout_n).to(self.device)
        
        from models.T3Time_Wavelet_Gated_Qwen import WaveletAttentionPooling
        self.wavelet_pool = WaveletAttentionPooling(self.channel).to(self.device)
        
        self.cross_attn_fusion = nn.MultiheadAttention(self.channel, head, dropout=dropout_n, batch_first=True).to(self.device)
        self.fusion_norm = nn.LayerNorm(self.channel).to(self.device)

        self.prompt_encoder = nn.ModuleList([
            GatedSwiGLUTransformerLayer(d_llm, head, d_ff=d_ff, dropout=dropout_n) 
            for _ in range(e_layer)
        ]).to(self.device)
        
        from layers.Cross_Modal_Align import CrossModal
        self.num_cma_heads = 4
        self.cma_heads = nn.ModuleList([
            CrossModal(d_model=self.num_nodes, n_heads=1, d_ff=d_ff, norm='LayerNorm', attn_dropout=dropout_n, dropout=dropout_n, pre_norm=True).to(self.device) 
            for _ in range(self.num_cma_heads)
        ])
        
        from models.T3Time import AdaptiveDynamicHeadsCMA
        self.adaptive_dynamic_heads_cma = AdaptiveDynamicHeadsCMA(self.num_cma_heads, self.num_nodes, self.channel, self.device)
        self.residual_alpha = nn.Parameter(torch.ones(self.channel) * 0.5).to(self.device)
        
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(self.channel, head, batch_first=True, norm_first=True), 
            num_layers=d_layer
        ).to(self.device)
        self.c_to_length = nn.Linear(self.channel, self.pred_len).to(self.device)

    def forward(self, input_data, input_data_mark, embeddings):
        B, L, N = input_data.shape
        input_data = input_data.float()
        embeddings = embeddings.float().squeeze(-1).permute(0, 2, 1) # [B, N, d_llm]
        
        # 归一化
        input_data_norm = self.normalize_layers(input_data, 'norm') # [B, L, N]
        
        # 1. Inception 局部特征提取
        # input_data_norm.permute(0, 2, 1) -> [B, N, L]
        local_features = self.inception(input_data_norm.permute(0, 2, 1)) # [B, C, L]
        
        # 2. 时域处理
        time_feat = self.length_to_feature(local_features) # [B, C, C]
        # 注意：这里为了匹配 Transformer 习惯，我们视 C 为序列长度，C 为维度
        # 在 T3Time 逻辑中，通常是 [B, N, C]
        time_feat = time_feat.permute(0, 2, 1) # [B, L_new, C]
        for layer in self.ts_encoder: time_feat = layer(time_feat)
        
        # 3. 小波处理
        wavelet_coeffs = self.wavelet_transform(input_data_norm.permute(0, 2, 1))
        wavelet_feats = []
        for i, node in enumerate(wavelet_coeffs):
            B_w, N_w, L_w = node.shape
            tokens = self.wavelet_proj[min(i, 9)](node.unsqueeze(-1).reshape(B_w*N_w, L_w, 1))
            wavelet_feats.append(self.wavelet_encoder(tokens))
        freq_feat = self.wavelet_pool(wavelet_feats).reshape(B, N, self.channel)
        
        # 4. 融合
        # 调整 time_feat 维度以匹配 [B, N, C]
        # 如果 L_new 不等于 N，我们需要投影。在 ETTh1 中 N=7。
        if time_feat.shape[1] != self.num_nodes:
            # 简化处理：将 inception 输出的序列维度投影回 N
            time_feat = F.adaptive_avg_pool1d(time_feat.permute(0, 2, 1), self.num_nodes).permute(0, 2, 1)
        
        fused_attn, _ = self.cross_attn_fusion(time_feat, freq_feat, freq_feat)
        fused_features = self.fusion_norm(fused_attn + time_feat).permute(0, 2, 1) # [B, C, N]
        
        # 5. Prompt 处理
        prompt_feat = embeddings
        for layer in self.prompt_encoder: prompt_feat = layer(prompt_feat)
        prompt_feat = prompt_feat.permute(0, 2, 1) # [B, d_llm, N]
        
        # 6. CMA + 残差
        cma_outs = [h(fused_features, prompt_feat, prompt_feat) for h in self.cma_heads]
        fused_cma = self.adaptive_dynamic_heads_cma(cma_outs)
        alpha = self.residual_alpha.view(1, -1, 1)
        cross_out = (alpha * fused_cma + (1 - alpha) * fused_features).permute(0, 2, 1)
        
        # 7. 解码
        dec_out = self.decoder(cross_out, cross_out)
        dec_out = self.c_to_length(dec_out).permute(0, 2, 1)
        return self.normalize_layers(dec_out, 'denorm')

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

