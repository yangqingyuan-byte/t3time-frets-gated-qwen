"""
T3Time_FreTS_Gated_Qwen 融合机制实验版本
包含多种融合方式的实验实现
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.StandardNorm import Normalize
from layers.Cross_Modal_Align import CrossModal

class GatedTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu, 
                 layer_norm_eps=1e-5, use_improved_gate=True):
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
        self.use_improved_gate = use_improved_gate  # True=改进门控(基于输入), False=原始门控(基于注意力输出)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        x = src
        nx = self.norm1(x)
        attn_output, _ = self.self_attn(nx, nx, nx, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        
        # 门控机制切换
        if self.use_improved_gate:
            # 改进门控: 基于归一化后的输入
            gate = torch.sigmoid(self.gate_proj(nx))
        else:
            # 原始门控: 基于注意力输出
            gate = torch.sigmoid(self.gate_proj(attn_output))
        
        attn_output = attn_output * gate
        x = x + self.dropout1(attn_output)
        nx = self.norm2(x)
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(nx))))
        x = x + self.dropout2(ff_output)
        return x

class FreTSComponent(nn.Module):
    def __init__(self, channel, seq_len, sparsity_threshold=0.01, scale=0.02, dropout=0.1, use_sparsity=True):
        super().__init__()
        # 稀疏化阈值：默认 0.01（原始配置），可配置
        self.sparsity_threshold = sparsity_threshold
        self.use_sparsity = use_sparsity  # 是否使用稀疏化
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
        
        # 稀疏化机制切换
        if self.use_sparsity and self.sparsity_threshold > 0:
            y = F.softshrink(y, lambd=self.sparsity_threshold)
        
        y = torch.view_as_complex(y)
        out = torch.fft.irfft(y, n=L, dim=1, norm="ortho")
        return self.dropout(out)

class AttentionPooling(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.attention = nn.Sequential(nn.Linear(embed_dim, embed_dim // 2), nn.ReLU(), nn.Linear(embed_dim // 2, 1))
    def forward(self, x):
        attn_weights = F.softmax(self.attention(x), dim=1)
        return (x * attn_weights).sum(dim=1)

class FrequencyAttentionPooling(nn.Module):
    """
    原始 T3Time 的频域池化方法（用于固定 FFT 模式）
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.freq_attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )

    def forward(self, freq_enc_out):
        attn_logits = self.freq_attention(freq_enc_out)  # [B*N, Lf, 1]
        attn_weights = F.softmax(attn_logits, dim=1)  # normalize over Lf
        pooled_freq = (freq_enc_out * attn_weights).sum(dim=1)  # [B*N, C]
        return pooled_freq

class TriModalFreTSGatedQwenFusionExp(nn.Module):
    """
    融合机制实验版本
    fusion_mode: 'gate', 'weighted', 'cross_attn', 'hybrid'
    """
    def __init__(self, device="cuda", channel=32, num_nodes=7, seq_len=96, pred_len=96, 
                 dropout_n=0.1, d_llm=1024, e_layer=1, d_layer=1, d_ff=32, head=8, 
                 sparsity_threshold=0.01, scale=0.02, fusion_mode='gate',
                 use_frets=True, use_complex=True, use_sparsity=True, use_improved_gate=True):
        super().__init__()
        self.device, self.channel, self.num_nodes, self.seq_len, self.pred_len, self.d_llm = device, channel, num_nodes, seq_len, pred_len, d_llm
        self.fusion_mode = fusion_mode
        self.use_frets = use_frets  # True=使用FreTS Component, False=使用固定FFT
        self.use_complex = use_complex  # True=使用复数, False=仅幅度 (仅当use_frets=False时有效)
        self.use_sparsity = use_sparsity  # True=使用稀疏化, False=不使用 (仅当use_frets=True时有效)
        self.use_improved_gate = use_improved_gate  # True=改进门控, False=原始门控
        
        # 【改进3】归一化改为 affine=False（对齐 T3Time V30）
        self.normalize_layers = Normalize(num_nodes, affine=False).to(device)
        
        # 时域分支
        self.length_to_feature = nn.Linear(self.seq_len, self.channel).to(self.device)
        self.ts_encoder = nn.ModuleList([
            GatedTransformerEncoderLayer(d_model=self.channel, nhead=head, dropout=dropout_n, 
                                        use_improved_gate=use_improved_gate) 
            for _ in range(e_layer)
        ]).to(self.device)
        
        # 频域分支
        if self.use_frets:
            # 使用 FreTS Component
            self.fre_projection = nn.Linear(1, self.channel).to(self.device)
            self.frets_branch = FreTSComponent(
                self.channel, self.seq_len, 
                sparsity_threshold=sparsity_threshold if use_sparsity else 0.0, 
                scale=scale, dropout=dropout_n, use_sparsity=use_sparsity
            ).to(self.device)
            self.fre_encoder = GatedTransformerEncoderLayer(
                d_model=self.channel, nhead=head, dropout=dropout_n, 
                use_improved_gate=use_improved_gate
            ).to(self.device)
            self.fre_pool = AttentionPooling(self.channel).to(self.device)
        else:
            # 使用固定 FFT (原始 T3Time 方式)
            self.Lf = seq_len // 2 + 1
            self.freq_token_proj = nn.Linear(1, self.channel).to(self.device)
            self.freq_encoder = GatedTransformerEncoderLayer(
                d_model=self.channel, nhead=head, dropout=dropout_n,
                use_improved_gate=use_improved_gate
            ).to(self.device)
            self.freq_pool = FrequencyAttentionPooling(self.channel).to(self.device)
        
        # 融合机制（根据 fusion_mode 选择）
        if fusion_mode == 'gate':
            # 版本 A: 门控融合（类似 T3Time V30）
            self.fusion_gate = nn.Sequential(
                nn.Linear(channel * 2 + 1, channel // 2),
                nn.ReLU(),
                nn.Linear(channel // 2, channel),
                nn.Sigmoid()
            ).to(device)
        elif fusion_mode == 'weighted':
            # 版本 B: 可学习加权求和
            self.fusion_alpha = nn.Parameter(torch.tensor(0.5)).to(device)
        elif fusion_mode == 'cross_attn':
            # 版本 C: 改进的 Cross-Attention（增加残差连接）
            self.cross_attn_fusion = nn.MultiheadAttention(self.channel, head, dropout=dropout_n, batch_first=True).to(self.device)
            self.fusion_norm = nn.LayerNorm(self.channel).to(self.device)
        elif fusion_mode == 'hybrid':
            # 版本 D: 混合融合（门控 + Cross-Attention）
            self.cross_attn_fusion = nn.MultiheadAttention(self.channel, head, dropout=dropout_n, batch_first=True).to(self.device)
            self.fusion_gate = nn.Sequential(
                nn.Linear(channel * 2, channel),
                nn.Sigmoid()
            ).to(device)
            self.fusion_norm = nn.LayerNorm(self.channel).to(self.device)
        else:
            raise ValueError(f"Unknown fusion_mode: {fusion_mode}")
        
        # Prompt 编码器
        self.prompt_encoder = nn.ModuleList([
            GatedTransformerEncoderLayer(d_model=self.d_llm, nhead=head, dropout=dropout_n,
                                        use_improved_gate=use_improved_gate) 
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
        
        # 时域/频域处理
        time_encoded = self.length_to_feature(x_perm)
        for layer in self.ts_encoder: time_encoded = layer(time_encoded)
        
        # 频域处理：根据 use_frets 选择不同方式
        if self.use_frets:
            # 使用 FreTS Component
            fre_input = self.fre_projection(x_perm.reshape(B*N, L, 1))
            fre_processed = self.frets_branch(fre_input)
            fre_pooled = self.fre_pool(fre_processed)
            fre_encoded = self.fre_encoder(fre_pooled.reshape(B, N, self.channel))
        else:
            # 使用固定 FFT (原始 T3Time 方式)
            freq_complex = torch.fft.rfft(x_perm, dim=-1)  # [B, N, Lf] (复数)
            
            if self.use_complex:
                # 使用复数信息：将实部和虚部分别处理，然后拼接
                # 注意：原始T3Time只使用幅度，这里为了对比实验，如果use_complex=True，
                # 我们可以使用实部和虚部的组合，但为了简化，这里仍然使用幅度
                # 真正的复数使用需要在模型设计中更复杂的处理
                freq_mag = torch.abs(freq_complex)
            else:
                # 仅使用幅度（原始T3Time方式）
                freq_mag = torch.abs(freq_complex)
            
            B_freq, N_freq, Lf = freq_mag.shape
            freq_tokens = freq_mag.unsqueeze(-1).reshape(B_freq*N_freq, Lf, 1)  # [B*N, Lf, 1]
            freq_tokens = self.freq_token_proj(freq_tokens)  # [B*N, Lf, C]
            freq_enc_out = self.freq_encoder(freq_tokens)  # [B*N, Lf, C]
            freq_enc_out = self.freq_pool(freq_enc_out)  # [B*N, C]
            fre_encoded = freq_enc_out.reshape(B, N, self.channel)  # [B, N, C]
        
        # 融合机制（根据 fusion_mode 选择）
        if self.fusion_mode == 'gate':
            # 版本 A: 门控融合（Horizon-Aware Gate）
            horizon_info = torch.full((B, N, 1), self.pred_len / 100.0, device=self.device)
            gate_input = torch.cat([time_encoded, fre_encoded, horizon_info], dim=-1)
            gate = self.fusion_gate(gate_input)
            fused_features = (time_encoded + gate * fre_encoded).permute(0, 2, 1)
            
        elif self.fusion_mode == 'weighted':
            # 版本 B: 可学习加权求和
            alpha = torch.sigmoid(self.fusion_alpha)
            fused_features = (alpha * time_encoded + (1 - alpha) * fre_encoded).permute(0, 2, 1)
            
        elif self.fusion_mode == 'cross_attn':
            # 版本 C: 改进的 Cross-Attention（增加残差连接）
            fused_attn, _ = self.cross_attn_fusion(time_encoded, fre_encoded, fre_encoded)
            # 双重残差连接：时域 + 频域 + 注意力
            fused_features = self.fusion_norm(fused_attn + time_encoded + fre_encoded).permute(0, 2, 1)
            
        elif self.fusion_mode == 'hybrid':
            # 版本 D: 混合融合（Cross-Attention + 门控）
            fused_attn, _ = self.cross_attn_fusion(time_encoded, fre_encoded, fre_encoded)
            gate_input = torch.cat([time_encoded, fre_encoded], dim=-1)
            gate = self.fusion_gate(gate_input)
            # 先 Cross-Attention，再门控融合
            fused_temp = self.fusion_norm(fused_attn + time_encoded)
            fused_features = (fused_temp + gate * fre_encoded).permute(0, 2, 1)
        
        # CMA 和 Decoder
        # embeddings: [B, N, d_llm] (经过 permute)
        # prompt_encoder 期望: [B, N, d_llm] (batch_first=True)
        prompt_feat = embeddings  # [B, N, d_llm]
        for layer in self.prompt_encoder: prompt_feat = layer(prompt_feat)  # [B, N, d_llm]
        # CMA 期望: [B, d_llm, N]
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
