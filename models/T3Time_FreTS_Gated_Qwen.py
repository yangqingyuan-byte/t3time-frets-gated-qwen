"""
T3Time_FreTS_Gated_Qwen 最终版本
基于最佳配置的简化版本（不带消融选项）
固定配置：
- 使用 FreTS Component（可学习频域MLP）
- 使用稀疏化机制（sparsity_threshold=0.009）
- 使用改进门控（基于归一化输入）
- 使用 Gate 融合机制
- FreTS scale=0.018
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.StandardNorm import Normalize
from layers.Cross_Modal_Align import CrossModal

class GatedTransformerEncoderLayer(nn.Module):
    """改进的门控 Transformer 编码器层（使用改进门控机制）"""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu, 
                 layer_norm_eps=1e-5):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.gate_proj = nn.Linear(d_model, d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        x = src
        nx = self.norm1(x)
        attn_output, _ = self.self_attn(nx, nx, nx, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        
        # 改进门控: 基于归一化后的输入
        gate = torch.sigmoid(self.gate_proj(nx))
        attn_output = attn_output * gate
        x = x + self.dropout1(attn_output)
        nx = self.norm2(x)
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(nx))))
        x = x + self.dropout2(ff_output)
        return x

class FreTSComponent(nn.Module):
    """FreTS Component: 可学习的频域 MLP"""
    def __init__(self, channel, seq_len, sparsity_threshold=0.009, scale=0.018, dropout=0.1):
        super().__init__()
        self.sparsity_threshold = sparsity_threshold
        self.r = nn.Parameter(scale * torch.randn(channel, channel))
        self.i = nn.Parameter(scale * torch.randn(channel, channel))
        self.rb = nn.Parameter(torch.zeros(channel))
        self.ib = nn.Parameter(torch.zeros(channel))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B_N, L, C = x.shape
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
        o_real = F.relu(torch.einsum('blc,cd->bld', x_fft.real, self.r) - torch.einsum('blc,cd->bld', x_fft.imag, self.i) + self.rb)
        o_imag = F.relu(torch.einsum('blc,cd->bld', x_fft.imag, self.r) + torch.einsum('blc,cd->bld', x_fft.real, self.i) + self.ib)
        y = torch.stack([o_real, o_imag], dim=-1)
        
        # 稀疏化机制
        y = F.softshrink(y, lambd=self.sparsity_threshold)
        
        y = torch.view_as_complex(y)
        out = torch.fft.irfft(y, n=L, dim=1, norm="ortho")
        return self.dropout(out)

class AttentionPooling(nn.Module):
    """注意力池化"""
    def __init__(self, embed_dim):
        super().__init__()
        self.attention = nn.Sequential(nn.Linear(embed_dim, embed_dim // 2), nn.ReLU(), nn.Linear(embed_dim // 2, 1))
    def forward(self, x):
        attn_weights = F.softmax(self.attention(x), dim=1)
        return (x * attn_weights).sum(dim=1)

class TriModalFreTSGatedQwen(nn.Module):
    """
    T3Time_FreTS_Gated_Qwen 最终版本
    固定使用最佳配置：
    - FreTS Component（可学习频域MLP）
    - 稀疏化机制（sparsity_threshold=0.009）
    - 改进门控（基于归一化输入）
    - Gate 融合机制
    - FreTS scale=0.018
    """
    def __init__(self, device="cuda", channel=32, num_nodes=7, seq_len=96, pred_len=96, 
                 dropout_n=0.1, d_llm=1024, e_layer=1, d_layer=1, d_ff=32, head=8):
        super().__init__()
        self.device, self.channel, self.num_nodes, self.seq_len, self.pred_len, self.d_llm = device, channel, num_nodes, seq_len, pred_len, d_llm
        
        # 归一化层
        self.normalize_layers = Normalize(num_nodes, affine=False).to(device)
        
        # 时域分支
        self.length_to_feature = nn.Linear(self.seq_len, self.channel).to(self.device)
        self.ts_encoder = nn.ModuleList([
            GatedTransformerEncoderLayer(d_model=self.channel, nhead=head, dropout=dropout_n) 
            for _ in range(e_layer)
        ]).to(self.device)
        
        # 频域分支：使用 FreTS Component
        self.fre_projection = nn.Linear(1, self.channel).to(self.device)
        self.frets_branch = FreTSComponent(
            self.channel, self.seq_len, 
            sparsity_threshold=0.009,  # 最佳配置
            scale=0.018,  # 最佳配置
            dropout=dropout_n
        ).to(self.device)
        self.fre_encoder = GatedTransformerEncoderLayer(
            d_model=self.channel, nhead=head, dropout=dropout_n
        ).to(self.device)
        self.fre_pool = AttentionPooling(self.channel).to(self.device)
        
        # 融合机制：Gate 融合
        self.fusion_gate = nn.Sequential(
            nn.Linear(channel * 2 + 1, channel // 2),
            nn.ReLU(),
            nn.Linear(channel // 2, channel),
            nn.Sigmoid()
        ).to(device)
        
        # Prompt 编码器
        self.prompt_encoder = nn.ModuleList([
            GatedTransformerEncoderLayer(d_model=self.d_llm, nhead=head, dropout=dropout_n) 
            for _ in range(e_layer)
        ]).to(self.device)
        
        # CMA
        from layers.Cross_Modal_Align import CrossModal
        self.cma_heads = nn.ModuleList([CrossModal(d_model=self.num_nodes, n_heads=1, d_ff=d_ff, norm='LayerNorm', attn_dropout=dropout_n, dropout=dropout_n, pre_norm=True, activation="gelu", res_attention=True, n_layers=1, store_attn=False).to(self.device) for _ in range(4)])
        from models.T3Time import AdaptiveDynamicHeadsCMA
        self.adaptive_dynamic_heads_cma = AdaptiveDynamicHeadsCMA(num_heads=4, num_nodes=num_nodes, channel=self.channel, device=self.device)
        self.residual_alpha = nn.Parameter(torch.ones(self.channel) * 0.5).to(self.device)
        
        # 解码器
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=self.channel, nhead=head, batch_first=True, norm_first=True, dropout=dropout_n), num_layers=d_layer).to(self.device)
        self.c_to_length = nn.Linear(self.channel, self.pred_len, bias=True).to(self.device)

    def forward(self, input_data, input_data_mark, embeddings):
        # 1. RevIN 归一化
        x = input_data.float()
        x_norm = self.normalize_layers(x, 'norm') 
        
        # embeddings 输入: [B, d_llm, N, 1] -> squeeze(-1): [B, d_llm, N] -> permute(0, 2, 1): [B, N, d_llm]
        embeddings = embeddings.float().squeeze(-1).permute(0, 2, 1)  # [B, d_llm, N, 1] -> [B, d_llm, N] -> [B, N, d_llm]
        x_perm = x_norm.permute(0, 2, 1) # [B, N, L]
        B, N, L = x_perm.shape
        
        # 时域处理
        time_encoded = self.length_to_feature(x_perm)
        for layer in self.ts_encoder: 
            time_encoded = layer(time_encoded)
        
        # 频域处理：使用 FreTS Component
        fre_input = self.fre_projection(x_perm.reshape(B*N, L, 1))
        fre_processed = self.frets_branch(fre_input)
        fre_pooled = self.fre_pool(fre_processed)
        fre_encoded = self.fre_encoder(fre_pooled.reshape(B, N, self.channel))
        
        # 融合机制：Gate 融合（Horizon-Aware Gate）
        horizon_info = torch.full((B, N, 1), self.pred_len / 100.0, device=self.device)
        gate_input = torch.cat([time_encoded, fre_encoded, horizon_info], dim=-1)
        gate = self.fusion_gate(gate_input)
        fused_features = (time_encoded + gate * fre_encoded).permute(0, 2, 1)
        
        # CMA 和 Decoder
        prompt_feat = embeddings  # [B, N, d_llm]
        for layer in self.prompt_encoder: 
            prompt_feat = layer(prompt_feat)  # [B, N, d_llm]
        prompt_feat = prompt_feat.permute(0, 2, 1)  # [B, N, d_llm] -> [B, d_llm, N]
        cma_outputs = [cma_head(fused_features, prompt_feat, prompt_feat) for cma_head in self.cma_heads]
        fused_cma = self.adaptive_dynamic_heads_cma(cma_outputs)
        alpha = self.residual_alpha.view(1, -1, 1)
        cross_out = (alpha * fused_cma + (1 - alpha) * fused_features).permute(0, 2, 1)
        dec_out = self.decoder(cross_out, cross_out)
        dec_out = self.c_to_length(dec_out).permute(0, 2, 1)
        
        # 2. RevIN 反归一化
        return self.normalize_layers(dec_out, 'denorm')

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
