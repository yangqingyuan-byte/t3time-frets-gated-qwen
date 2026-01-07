"""
T3Time_FreTS_Gated_Qwen_LongSeq_v2 改进版本
完全参考 T3Time 的流程和结构，但使用 FreTS Component 替代固定 FFT
关键改进：
1. 完全按照 T3Time 的流程顺序
2. RichHorizonGate 只基于时域特征（与 T3Time 一致）
3. 频域处理：FreTS Component → 编码器 → 池化（与 T3Time 的 FFT → 编码器 → 池化 对应）
4. 融合公式：gate * freq + (1 - gate) * time（与 T3Time 一致）
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

class FrequencyAttentionPooling(nn.Module):
    """
    Learnable, attention-weighted pooling over frequency bins（与 T3Time 一致）
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.freq_attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )

    def forward(self, freq_enc_out):
        attn_logits  = self.freq_attention(freq_enc_out)           # [B*N, Lf, 1]
        attn_weights = F.softmax(attn_logits, dim=1)               # normalize over Lf
        pooled_freq  = (freq_enc_out * attn_weights).sum(dim=1)    # [B*N, C]
        return pooled_freq

class RichHorizonGate(nn.Module):
    """
    Rich Horizon Gate（完全按照 T3Time 的实现）
    Each channel has its own gate that depends both 
    on the global context (pooled) and on the forecast horizon.
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.gate_mlp = nn.Sequential(
            nn.Linear(embed_dim + 1, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
            nn.Sigmoid()
        )

    def forward(self, embedding: torch.Tensor, horizon: int) -> torch.Tensor:
        """
        Args:
            embedding: [B, C, N] 时域编码后的特征（与 T3Time 一致）
            horizon: 预测长度
        Returns:
            gate: [B, C, 1] 门控值
        """
        B, C, N = embedding.size()
        pooled_embed = embedding.mean(dim=2)                                # [B, C]
        horizon_tensor = torch.full((B, 1), float(horizon) / 1000.0, device=embedding.device)

        gating_input = torch.cat([pooled_embed, horizon_tensor], dim=1)     # [B, C+1]
        gate = self.gate_mlp(gating_input).unsqueeze(-1)                    # [B, C, 1]
        return gate

class TriModalFreTSGatedQwenLongSeqV2(nn.Module):
    """
    T3Time_FreTS_Gated_Qwen_LongSeq_v2
    完全参考 T3Time 的流程，但使用 FreTS Component 替代固定 FFT
    关键改进：
    1. 完全按照 T3Time 的流程顺序
    2. RichHorizonGate 只基于时域特征（与 T3Time 一致）
    3. 频域处理：FreTS Component → 编码器 → 池化（与 T3Time 的 FFT → 编码器 → 池化 对应）
    4. 融合公式：gate * freq + (1 - gate) * time（与 T3Time 一致）
    """
    def __init__(self, device="cuda", channel=32, num_nodes=7, seq_len=96, pred_len=96, 
                 dropout_n=0.1, d_llm=1024, e_layer=1, d_layer=1, d_ff=32, head=8,
                 sparsity_threshold=0.009, frets_scale=0.018, use_dynamic_sparsity=True):
        super().__init__()
        self.device, self.channel, self.num_nodes, self.seq_len, self.pred_len, self.d_llm = device, channel, num_nodes, seq_len, pred_len, d_llm
        self.use_dynamic_sparsity = use_dynamic_sparsity
        
        # 归一化层
        self.normalize_layers = Normalize(num_nodes, affine=False).to(device)
        
        # 时域分支（与 T3Time 一致）
        self.length_to_feature = nn.Linear(self.seq_len, self.channel).to(self.device)
        self.ts_encoder = GatedTransformerEncoderLayer(
            d_model=self.channel,
            nhead=head,
            dim_feedforward=4 * self.channel,  # 与 T3Time 一致
            dropout=dropout_n,
        ).to(self.device)
        
        # 频域分支：使用 FreTS Component（替代 T3Time 的固定 FFT）
        # 流程：FreTS Component → 投影 → 编码器 → 池化（对应 T3Time 的 FFT → 投影 → 编码器 → 池化）
        self.frets_branch = FreTSComponent(
            self.channel, self.seq_len, 
            sparsity_threshold=sparsity_threshold,
            scale=frets_scale,
            dropout=dropout_n,
            base_seq_len=seq_len,
            use_dynamic_sparsity=use_dynamic_sparsity
        ).to(self.device)
        
        # 频域投影（对应 T3Time 的 freq_token_proj）
        self.freq_token_proj = nn.Linear(1, self.channel).to(self.device)
        
        # 频域编码器（与 T3Time 一致）
        self.freq_encoder = GatedTransformerEncoderLayer(
            d_model=self.channel,
            nhead=head,
            dim_feedforward=4 * self.channel,  # 与 T3Time 一致
            dropout=dropout_n,
        ).to(self.device)
        
        # 频域池化（与 T3Time 一致）
        self.freq_pool = FrequencyAttentionPooling(self.channel).to(self.device)
        
        # Rich Horizon Gate（与 T3Time 完全一致）
        self.rich_horizon_gate = RichHorizonGate(self.channel).to(self.device)
        
        # Prompt 编码器（与 T3Time 一致）
        self.prompt_encoder = GatedTransformerEncoderLayer(
            d_model=self.d_llm,
            nhead=head,
            dim_feedforward=4 * self.d_llm,  # 与 T3Time 一致
            dropout=dropout_n,
        ).to(self.device)
        
        # CMA（与 T3Time 一致）
        from layers.Cross_Modal_Align import CrossModal
        self.num_cma_heads = 4
        self.cma_heads = nn.ModuleList([
            CrossModal(d_model=self.num_nodes, n_heads=1, d_ff=d_ff, norm='LayerNorm', 
                      attn_dropout=dropout_n, dropout=dropout_n, pre_norm=True, 
                      activation="gelu", res_attention=True, n_layers=1, store_attn=False).to(self.device) 
            for _ in range(self.num_cma_heads)
        ])
        
        from models.T3Time import AdaptiveDynamicHeadsCMA
        self.adaptive_dynamic_heads_cma = AdaptiveDynamicHeadsCMA(
            num_heads=self.num_cma_heads, num_nodes=num_nodes, channel=self.channel, device=self.device
        )
        
        self.residual_alpha = nn.Parameter(torch.ones(self.channel) * 0.5).to(self.device)
        
        # 解码器（与 T3Time 一致）
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.channel, nhead=head, batch_first=True, norm_first=True, dropout=dropout_n
        ).to(self.device)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=d_layer).to(self.device)
        
        # 投影层（与 T3Time 一致）
        self.c_to_length = nn.Linear(self.channel, self.pred_len, bias=True).to(self.device)

    def frequency_domain_processing(self, input_data, pred_len=None):
        """
        频域处理（参考 T3Time，但使用 FreTS Component）
        Args:
            input_data: [B, N, L]
            pred_len: 预测长度（用于动态稀疏化）
        Returns:
            freq_enc_out: [B, N, C]
        """
        B, N, L = input_data.shape
        
        # 使用 FreTS Component 替代固定 FFT
        # 1. 准备输入（与 T3Time 的 FFT 对应）
        fre_input = input_data.unsqueeze(-1).reshape(B * N, L, 1)  # [B*N, L, 1]
        fre_input = self.freq_token_proj(fre_input)  # [B*N, L, C]（投影到channel维度）
        
        # 2. FreTS Component 处理（可学习频域变换，替代 T3Time 的固定 FFT）
        # FreTS Component 内部：FFT → 可学习变换 → IFFT
        fre_processed = self.frets_branch(fre_input, pred_len=pred_len)  # [B*N, L, C]
        
        # 3. 频域编码器（与 T3Time 一致）
        freq_enc_out = self.freq_encoder(fre_processed)  # [B*N, L, C]
        
        # 4. 频域池化（与 T3Time 一致）
        freq_enc_out = self.freq_pool(freq_enc_out)  # [B*N, C]
        
        # 5. Reshape
        freq_enc_out = freq_enc_out.reshape(B, N, self.channel)  # [B, N, C]
        
        return freq_enc_out

    def forward(self, input_data, input_data_mark, embeddings):
        """
        前向传播（完全按照 T3Time 的流程）
        """
        input_data = input_data.float()
        if input_data_mark is not None:
            input_data_mark = input_data_mark.float()
        embeddings = embeddings.float()
        embeddings = embeddings.squeeze(-1)  # [B, E, N]
        embeddings = embeddings.permute(0, 2, 1)  # [B, N, E]

        #------ RevIN
        input_data = self.normalize_layers(input_data, 'norm')
        input_data = input_data.permute(0, 2, 1)  # [B, N, L]

        #------ Frequency Encoding（使用 FreTS Component）
        freq_enc_out = self.frequency_domain_processing(input_data, pred_len=self.pred_len)  # [B, N, C]
        
        #------ Time Series Encoding（与 T3Time 一致）
        input_data = self.length_to_feature(input_data)  # [B, N, C]
        enc_out = self.ts_encoder(input_data)  # [B, N, C]
        enc_out = enc_out.permute(0, 2, 1)  # [B, C, N]

        #------ Rich Horizon Gate（与 T3Time 完全一致）
        gate = self.rich_horizon_gate(enc_out, self.pred_len)  # [B, C, 1]
        enc_out = gate * freq_enc_out.permute(0, 2, 1) + (1 - gate) * enc_out  # [B, C, N]
        
        #------ Prompt encoding（与 T3Time 一致）
        embeddings = self.prompt_encoder(embeddings)  # [B, N, E]
        embeddings = embeddings.permute(0, 2, 1)  # [B, E, N]

        #------ Aggregating Multiple CMA Heads（与 T3Time 一致）
        cma_outputs = []
        for cma_head in self.cma_heads:
            head_out = cma_head(enc_out, embeddings, embeddings)  # [B, C, N]
            cma_outputs.append(head_out)

        fused = self.adaptive_dynamic_heads_cma(cma_outputs)  # [B, C, N]

        #------ Residual Fusion（与 T3Time 一致）
        B, C, N = fused.shape
        alpha = self.residual_alpha.view(1, C, 1)
        cross_out = alpha * fused + (1 - alpha) * enc_out  # [B, C, N]
        cross_out = cross_out.permute(0, 2, 1)  # [B, N, C]

        #------ Decoder（与 T3Time 一致）
        dec_out = self.decoder(cross_out, cross_out)  # [B, N, C]

        #------ Projection（与 T3Time 一致）
        dec_out = self.c_to_length(dec_out)  # [B, N, L]
        dec_out = dec_out.permute(0, 2, 1)  # [B, L, N]

        #------ Denorm（与 T3Time 一致）
        dec_out = self.normalize_layers(dec_out, 'denorm')

        return dec_out

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
