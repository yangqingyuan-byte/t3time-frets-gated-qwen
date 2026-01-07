"""
T3Time with Wavelet Transform, Qwen3-0.6B, Gated Attention, and Dynamic Statistical Injection.
在 Gated Wavelet Qwen 模型的基础上，引入方案二：轻量化动态统计特征注入。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np
from layers.StandardNorm import Normalize
from layers.Cross_Modal_Align import CrossModal


class GatedTransformerEncoderLayer(nn.Module):
    """
    根据 NIPS 2025 Best Paper 改进的 Transformer Encoder Layer
    在 SDPA (Scaled Dot-Product Attention) 输出后添加 Sigmoid 门控
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu, layer_norm_eps=1e-5):
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
        gate = torch.sigmoid(self.gate_proj(nx))
        attn_output = attn_output * gate
        x = x + self.dropout1(attn_output)
        nx = self.norm2(x)
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(nx))))
        x = x + self.dropout2(ff_output)
        return x


class StatisticalFeatureExtractor(nn.Module):
    """
    轻量化统计特征提取器 (方案二)
    提取输入序列的均值、标准差、峰峰值、一阶差分均值等动态特征
    """
    def __init__(self, d_llm):
        super().__init__()
        self.d_llm = d_llm
        # 我们提取 4 种统计量: mean, std, (max-min), diff_mean
        self.num_stats = 4
        self.proj = nn.Sequential(
            nn.Linear(self.num_stats, d_llm // 4),
            nn.ReLU(),
            nn.Linear(d_llm // 4, d_llm)
        )
        self.layer_norm = nn.LayerNorm(d_llm)

    def forward(self, x):
        # x shape: [B, N, L] (Batch, Nodes, Length)
        # 计算统计量
        mean = x.mean(dim=-1, keepdim=True)          # [B, N, 1]
        std = x.std(dim=-1, keepdim=True)            # [B, N, 1]
        max_val = x.max(dim=-1, keepdim=True)[0]     # [B, N, 1]
        min_val = x.min(dim=-1, keepdim=True)[0]     # [B, N, 1]
        peak_to_peak = max_val - min_val             # [B, N, 1]
        diff_mean = torch.diff(x, dim=-1).abs().mean(dim=-1, keepdim=True) # [B, N, 1]
        
        # 拼接特征
        stats = torch.cat([mean, std, peak_to_peak, diff_mean], dim=-1) # [B, N, 4]
        
        # 投影到 LLM 维度
        dynamic_emb = self.proj(stats) # [B, N, d_llm]
        return self.layer_norm(dynamic_emb)


class WaveletTransform(nn.Module):
    def __init__(self, wavelet='db4', mode='symmetric', max_level=None):
        super().__init__()
        self.wavelet = wavelet
        self.mode = mode
        self.max_level = max_level
        try:
            wavelet_obj = pywt.Wavelet(wavelet)
            self.dec_lo = torch.tensor(wavelet_obj.dec_lo, dtype=torch.float32)
            self.dec_hi = torch.tensor(wavelet_obj.dec_hi, dtype=torch.float32)
        except:
            self.dec_lo = torch.tensor([-0.0106, 0.0329, 0.0308, -0.1870, -0.0280, 0.6309, 0.7148, 0.2304], dtype=torch.float32)
            self.dec_hi = torch.tensor([-0.2304, 0.7148, -0.6309, -0.0280, 0.1870, 0.0308, -0.0329, -0.0106], dtype=torch.float32)
    
    def dwt_1d(self, x):
        B, N, L = x.shape
        device = x.device
        dec_lo = self.dec_lo.to(device)
        dec_hi = self.dec_hi.to(device)
        pad_len = len(dec_lo) - 1
        x_padded = F.pad(x, (pad_len, pad_len), mode='circular')
        cA = F.conv1d(x_padded.view(B*N, 1, -1), dec_lo.view(1, 1, -1), stride=2)
        cD = F.conv1d(x_padded.view(B*N, 1, -1), dec_hi.view(1, 1, -1), stride=2)
        cA = cA.view(B, N, -1)
        cD = cD.view(B, N, -1)
        return cA, cD
    
    def wavedec(self, x, level=None):
        if level is None:
            level = self.max_level if self.max_level else pywt.dwt_max_level(x.shape[-1], self.wavelet)
        coeffs = []
        current = x
        for _ in range(level):
            cA, cD = self.dwt_1d(current)
            coeffs.insert(0, cD)
            current = cA
        coeffs.insert(0, cA)
        return coeffs
        
    def forward(self, x):
        max_level = self.max_level if self.max_level else pywt.dwt_max_level(x.shape[-1], self.wavelet)
        coeffs = self.wavedec(x, level=max_level)
        return coeffs


class WaveletAttentionPooling(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )
    
    def forward(self, wavelet_features):
        attn_weights_list = []
        for feat in wavelet_features:
            attn_logits = self.attention(feat)
            attn_weights = F.softmax(attn_logits, dim=1)
            attn_weights_list.append(attn_weights)
        
        pooled_list = []
        for feat, attn in zip(wavelet_features, attn_weights_list):
            pooled = (feat * attn).sum(dim=1)
            pooled_list.append(pooled)
        
        if len(pooled_list) > 1:
            pooled = torch.stack(pooled_list, dim=1)
            pooled = pooled.mean(dim=1)
        else:
            pooled = pooled_list[0]
        return pooled


class CrossAttentionFusion(nn.Module):
    def __init__(self, d_model, nhead=8, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, time_features, freq_features):
        fused, _ = self.cross_attn(query=time_features, key=freq_features, value=freq_features)
        fused = self.norm(fused + time_features)
        return fused


class TriModalWaveletGatedStatQwen(nn.Module):
    """
    方案二改进版：在原有模型基础上注入动态统计特征 (Stat Injection)
    """
    def __init__(
        self,
        device="cuda",
        channel=32,
        num_nodes=7,
        seq_len=96,
        pred_len=96,
        dropout_n=0.1,
        d_llm=1024,
        e_layer=1,
        d_layer=1,
        d_ff=32,
        head=8,
        wavelet='db4',
        use_cross_attention=True
    ):
        super().__init__()
        
        self.device = device
        self.channel = channel
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.dropout_n = dropout_n
        self.d_llm = d_llm
        self.e_layer = e_layer
        self.d_layer = d_layer
        self.d_ff = d_ff
        self.head = head
        self.use_cross_attention = use_cross_attention
        
        # 归一化层
        self.normalize_layers = Normalize(self.num_nodes, affine=False).to(self.device)
        
        # 时域特征提取
        self.length_to_feature = nn.Linear(self.seq_len, self.channel).to(self.device)
        
        # 时域编码器
        self.ts_encoder = nn.ModuleList([
            GatedTransformerEncoderLayer(d_model=self.channel, nhead=self.head, dropout=self.dropout_n)
            for _ in range(self.e_layer)
        ]).to(self.device)
        
        # 小波变换模块
        self.wavelet_transform = WaveletTransform(wavelet=wavelet).to(self.device)
        
        max_level = pywt.dwt_max_level(seq_len, wavelet) if seq_len > 0 else 3
        self.wavelet_proj_layers = nn.ModuleList([
            nn.Linear(1, self.channel).to(self.device)
            for _ in range(max_level + 1)
        ])
        
        self.wavelet_encoder = GatedTransformerEncoderLayer(
            d_model=self.channel, nhead=self.head, dropout=self.dropout_n
        ).to(self.device)
        
        self.wavelet_pool = WaveletAttentionPooling(self.channel).to(self.device)
        
        # 融合模块
        if self.use_cross_attention:
            self.fusion = CrossAttentionFusion(d_model=self.channel, nhead=self.head, dropout=self.dropout_n).to(self.device)
        else:
            self.fusion_gate = nn.Sequential(
                nn.Linear(self.channel * 2, self.channel),
                nn.ReLU(),
                nn.Linear(self.channel, self.channel),
                nn.Sigmoid()
            ).to(self.device)
        
        # --- 方案二核心: 统计特征提取与注入 ---
        self.stat_extractor = StatisticalFeatureExtractor(d_llm=self.d_llm).to(self.device)
        self.prompt_stat_fusion = nn.Sequential(
            nn.Linear(self.d_llm * 2, self.d_llm),
            nn.ReLU(),
            nn.Linear(self.d_llm, self.d_llm)
        ).to(self.device)
        # ------------------------------------

        self.prompt_encoder = nn.ModuleList([
            GatedTransformerEncoderLayer(d_model=self.d_llm, nhead=self.head, dropout=self.dropout_n)
            for _ in range(self.e_layer)
        ]).to(self.device)
        
        self.num_cma_heads = 4
        self.cma_heads = nn.ModuleList([
            CrossModal(
                d_model=self.num_nodes, n_heads=1, d_ff=self.d_ff, norm='LayerNorm',
                attn_dropout=self.dropout_n, dropout=self.dropout_n, pre_norm=True,
                activation="gelu", res_attention=True, n_layers=1, store_attn=False
            ).to(self.device)
            for _ in range(self.num_cma_heads)
        ])
        
        from models.T3Time import AdaptiveDynamicHeadsCMA
        self.adaptive_dynamic_heads_cma = AdaptiveDynamicHeadsCMA(
            num_heads=self.num_cma_heads, num_nodes=self.num_nodes, channel=self.channel, device=self.device
        )
        
        self.residual_alpha = nn.Parameter(torch.ones(self.channel) * 0.5).to(self.device)
        
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.channel, nhead=self.head, batch_first=True, norm_first=True, dropout=self.dropout_n
        ).to(self.device)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=self.d_layer).to(self.device)
        
        self.c_to_length = nn.Linear(self.channel, self.pred_len, bias=True).to(self.device)
    
    def wavelet_domain_processing(self, input_data):
        B, N, L = input_data.shape
        wavelet_coeffs = self.wavelet_transform(input_data)
        wavelet_features_list = []
        for level_idx, coeffs in enumerate(wavelet_coeffs):
            B_level, N_level, L_level = coeffs.shape
            if level_idx >= len(self.wavelet_proj_layers):
                for _ in range(level_idx - len(self.wavelet_proj_layers) + 1):
                    self.wavelet_proj_layers.append(nn.Linear(1, self.channel).to(self.device))
            coeffs_reshaped = coeffs.unsqueeze(-1).reshape(B * N, L_level, 1)
            proj_layer = self.wavelet_proj_layers[level_idx]
            wavelet_tokens = proj_layer(coeffs_reshaped)
            wavelet_encoded = self.wavelet_encoder(wavelet_tokens)
            wavelet_features_list.append(wavelet_encoded)
        
        pooled = self.wavelet_pool(wavelet_features_list)
        wavelet_features = pooled.reshape(B, N, self.channel)
        return wavelet_features
    
    def forward(self, input_data, input_data_mark, embeddings):
        input_data = input_data.float()
        input_data_mark = input_data_mark.float()
        embeddings = embeddings.float() # [B, d_llm, N]
        
        # 归一化与维度调整
        input_data_norm = self.normalize_layers(input_data, 'norm') # [B, L, N]
        input_data_norm = input_data_norm.permute(0, 2, 1)           # [B, N, L]
        
        # 1. 提取动态统计特征 (方案二)
        dynamic_stat_emb = self.stat_extractor(input_data_norm)     # [B, N, d_llm]
        
        # 2. 准备静态 LLM 嵌入
        static_llm_emb = embeddings.squeeze(-1).permute(0, 2, 1)   # [B, N, d_llm]
        
        # 3. 动态注入融合
        # 将静态语义与动态统计特征拼接并融合
        combined_prompt_emb = torch.cat([static_llm_emb, dynamic_stat_emb], dim=-1)
        fused_prompt_emb = self.prompt_stat_fusion(combined_prompt_emb) # [B, N, d_llm]
        
        # 4. 后续正常处理
        wavelet_features = self.wavelet_domain_processing(input_data_norm)
        time_features = self.length_to_feature(input_data_norm)
        
        time_encoded = time_features
        for layer in self.ts_encoder:
            time_encoded = layer(time_encoded)
        
        if self.use_cross_attention:
            fused_features = self.fusion(time_encoded, wavelet_features)
        else:
            concat_features = torch.cat([time_encoded, wavelet_features], dim=-1)
            gate = self.fusion_gate(concat_features)
            fused_features = gate * wavelet_features + (1 - gate) * time_encoded
        
        fused_features = fused_features.permute(0, 2, 1) # [B, C, N]
        
        # Prompt 编码
        for layer in self.prompt_encoder:
            fused_prompt_emb = layer(fused_prompt_emb)
        fused_prompt_emb = fused_prompt_emb.permute(0, 2, 1) # [B, d_llm, N]
        
        cma_outputs = []
        for cma_head in self.cma_heads:
            head_out = cma_head(fused_features, fused_prompt_emb, fused_prompt_emb)
            cma_outputs.append(head_out)
        
        fused = self.adaptive_dynamic_heads_cma(cma_outputs)
        B, C, N = fused.shape
        alpha = self.residual_alpha.view(1, C, 1)
        cross_out = alpha * fused + (1 - alpha) * fused_features
        cross_out = cross_out.permute(0, 2, 1)
        
        dec_out = self.decoder(cross_out, cross_out)
        dec_out = self.c_to_length(dec_out).permute(0, 2, 1)
        dec_out = self.normalize_layers(dec_out, 'denorm')
        return dec_out
    
    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

