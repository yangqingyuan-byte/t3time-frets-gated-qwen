"""
T3Time_FreTS_Gated_Qwen_LongSeq 改进版本
针对长序列预测优化的改进：
1. 改进 Horizon 归一化：使用 pred_len / 1000.0（类似 T3Time）
2. 改进融合公式：从残差式改为加权平均（gate * freq + (1-gate) * time）
3. 动态稀疏化阈值：根据预测长度调整
4. 使用全局上下文：基于全局池化（类似 RichHorizonGate）
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
    """FreTS Component: 可学习的频域 MLP（支持动态稀疏化阈值）"""
    def __init__(self, channel, seq_len, sparsity_threshold=0.009, scale=0.018, dropout=0.1, 
                 base_seq_len=96, use_dynamic_sparsity=True):
        super().__init__()
        self.base_sparsity_threshold = sparsity_threshold
        self.base_seq_len = base_seq_len
        self.use_dynamic_sparsity = use_dynamic_sparsity
        self.r = nn.Parameter(scale * torch.randn(channel, channel))
        self.i = nn.Parameter(scale * torch.randn(channel, channel))
        self.rb = nn.Parameter(torch.zeros(channel))
        self.ib = nn.Parameter(torch.zeros(channel))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pred_len=None):
        B_N, L, C = x.shape
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
        o_real = F.relu(torch.einsum('blc,cd->bld', x_fft.real, self.r) - torch.einsum('blc,cd->bld', x_fft.imag, self.i) + self.rb)
        o_imag = F.relu(torch.einsum('blc,cd->bld', x_fft.imag, self.r) + torch.einsum('blc,cd->bld', x_fft.real, self.i) + self.ib)
        y = torch.stack([o_real, o_imag], dim=-1)
        
        # 动态稀疏化机制：长序列时降低阈值
        if self.use_dynamic_sparsity and pred_len is not None:
            # 根据预测长度调整稀疏化阈值
            # 长序列时降低阈值，保留更多信息
            sparsity_threshold = self.base_sparsity_threshold * (self.base_seq_len / max(pred_len, self.base_seq_len))
        else:
            sparsity_threshold = self.base_sparsity_threshold
        
        y = F.softshrink(y, lambd=sparsity_threshold)
        
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

class RichHorizonGate(nn.Module):
    """
    改进的 Rich Horizon Gate（类似 T3Time）
    基于全局池化和改进的 horizon 归一化
    """
    def __init__(self, embed_dim, horizon_norm='div1000'):
        super().__init__()
        self.horizon_norm = horizon_norm
        self.gate_mlp = nn.Sequential(
            nn.Linear(embed_dim + 1, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
            nn.Sigmoid()
        )

    def forward(self, time_embed: torch.Tensor, freq_embed: torch.Tensor, horizon: int) -> torch.Tensor:
        """
        Args:
            time_embed: [B, N, C] 时域特征
            freq_embed: [B, N, C] 频域特征
            horizon: 预测长度
        Returns:
            gate: [B, C, 1] 门控值
        """
        # 全局池化：对节点维度求平均
        B, N, C = time_embed.shape
        pooled_time = time_embed.mean(dim=1)  # [B, C]
        pooled_freq = freq_embed.mean(dim=1)  # [B, C]
        pooled_embed = (pooled_time + pooled_freq) / 2  # [B, C] 融合全局上下文
        
        # 改进的 horizon 归一化
        if self.horizon_norm == 'div1000':
            horizon_tensor = torch.full((B, 1), float(horizon) / 1000.0, device=time_embed.device)
        elif self.horizon_norm == 'log':
            horizon_tensor = torch.full((B, 1), torch.log(torch.tensor(horizon / 96.0, device=time_embed.device)).item(), device=time_embed.device)
        else:
            horizon_tensor = torch.full((B, 1), float(horizon) / 100.0, device=time_embed.device)
        
        gating_input = torch.cat([pooled_embed, horizon_tensor], dim=1)  # [B, C+1]
        gate = self.gate_mlp(gating_input).unsqueeze(-1)  # [B, C, 1]
        return gate

class TriModalFreTSGatedQwenLongSeq(nn.Module):
    """
    T3Time_FreTS_Gated_Qwen_LongSeq 改进版本
    针对长序列预测的优化：
    1. 改进 Horizon 归一化（÷1000）
    2. 加权平均融合公式
    3. 动态稀疏化阈值
    4. 全局上下文门控
    """
    def __init__(self, device="cuda", channel=32, num_nodes=7, seq_len=96, pred_len=96, 
                 dropout_n=0.1, d_llm=1024, e_layer=1, d_layer=1, d_ff=32, head=8,
                 sparsity_threshold=0.009, frets_scale=0.018, use_dynamic_sparsity=True,
                 horizon_norm='div1000', fusion_mode='weighted_avg'):
        super().__init__()
        self.device, self.channel, self.num_nodes, self.seq_len, self.pred_len, self.d_llm = device, channel, num_nodes, seq_len, pred_len, d_llm
        self.use_dynamic_sparsity = use_dynamic_sparsity
        self.fusion_mode = fusion_mode
        
        # 归一化层
        self.normalize_layers = Normalize(num_nodes, affine=False).to(device)
        
        # 时域分支
        self.length_to_feature = nn.Linear(self.seq_len, self.channel).to(self.device)
        self.ts_encoder = nn.ModuleList([
            GatedTransformerEncoderLayer(d_model=self.channel, nhead=head, dropout=dropout_n) 
            for _ in range(e_layer)
        ]).to(self.device)
        
        # 频域分支：使用 FreTS Component（支持动态稀疏化）
        self.fre_projection = nn.Linear(1, self.channel).to(self.device)
        self.frets_branch = FreTSComponent(
            self.channel, self.seq_len, 
            sparsity_threshold=sparsity_threshold,
            scale=frets_scale,
            dropout=dropout_n,
            base_seq_len=seq_len,
            use_dynamic_sparsity=use_dynamic_sparsity
        ).to(self.device)
        self.fre_encoder = GatedTransformerEncoderLayer(
            d_model=self.channel, nhead=head, dropout=dropout_n
        ).to(self.device)
        self.fre_pool = AttentionPooling(self.channel).to(self.device)
        
        # 融合机制：使用 RichHorizonGate（全局上下文 + 改进归一化）
        self.rich_horizon_gate = RichHorizonGate(self.channel, horizon_norm=horizon_norm).to(device)
        
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
            time_encoded = layer(time_encoded)  # [B, N, C]
        
        # 频域处理：使用 FreTS Component（传入 pred_len 用于动态稀疏化）
        fre_input = self.fre_projection(x_perm.reshape(B*N, L, 1))
        fre_processed = self.frets_branch(fre_input, pred_len=self.pred_len)  # 传入 pred_len
        fre_pooled = self.fre_pool(fre_processed)
        fre_encoded = self.fre_encoder(fre_pooled.reshape(B, N, self.channel))  # [B, N, C]
        
        # 融合机制：使用 RichHorizonGate + 加权平均
        # RichHorizonGate 需要 [B, N, C] 格式的输入
        gate = self.rich_horizon_gate(time_encoded, fre_encoded, self.pred_len)  # [B, C, 1]
        
        # 将特征转换为 [B, C, N] 格式用于融合
        time_encoded_perm = time_encoded.permute(0, 2, 1)  # [B, C, N]
        freq_encoded_perm = fre_encoded.permute(0, 2, 1)  # [B, C, N]
        
        # 改进的融合公式：加权平均（而非残差式）
        if self.fusion_mode == 'weighted_avg':
            # 加权平均：gate * freq + (1 - gate) * time
            fused_features = gate * freq_encoded_perm + (1 - gate) * time_encoded_perm  # [B, C, N]
        else:
            # 残差式：time + gate * freq（原始方式）
            fused_features = time_encoded_perm + gate * freq_encoded_perm  # [B, C, N]
        
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
