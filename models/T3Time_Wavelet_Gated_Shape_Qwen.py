"""
T3Time with Wavelet Transform, Qwen3-0.6B, Gated Attention, and Shape-Aware Loss.
核心改进：在训练中引入形态损失 (Shape Loss)，提升对波峰和波谷的刻画能力。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np
from layers.StandardNorm import Normalize
from layers.Cross_Modal_Align import CrossModal

class GatedTransformerEncoderLayer(nn.Module):
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

class WaveletAttentionPooling(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )
    
    def forward(self, wavelet_features):
        pooled_list = []
        for feat in wavelet_features:
            attn_weights = F.softmax(self.attention(feat), dim=1)
            pooled = (feat * attn_weights).sum(dim=1)
            pooled_list.append(pooled)
        return torch.stack(pooled_list, dim=1).mean(dim=1)

class TriModalWaveletGatedShapeQwen(nn.Module):
    def __init__(
        self, device="cuda", channel=32, num_nodes=7, seq_len=96, pred_len=96,
        dropout_n=0.1, d_llm=1024, e_layer=1, d_layer=1, d_ff=32, head=8, wavelet='db4'
    ):
        super().__init__()
        self.device, self.channel, self.num_nodes, self.seq_len, self.pred_len = device, channel, num_nodes, seq_len, pred_len
        self.dropout_n, self.d_llm, self.e_layer, self.d_layer, self.d_ff, self.head = dropout_n, d_llm, e_layer, d_layer, d_ff, head
        
        self.normalize_layers = Normalize(self.num_nodes, affine=False).to(self.device)
        self.length_to_feature = nn.Linear(self.seq_len, self.channel).to(self.device)
        self.ts_encoder = nn.ModuleList([GatedTransformerEncoderLayer(self.channel, self.head, dropout=self.dropout_n) for _ in range(self.e_layer)]).to(self.device)
        
        self.wavelet_transform = WaveletTransform(wavelet=wavelet).to(self.device)
        self.wavelet_proj = nn.ModuleList([nn.Linear(1, self.channel).to(self.device) for _ in range(10)]) # 预留足够层数
        self.wavelet_encoder = GatedTransformerEncoderLayer(self.channel, self.head, dropout=self.dropout_n).to(self.device)
        self.wavelet_pool = WaveletAttentionPooling(self.channel).to(self.device)
        
        self.cross_attn_fusion = nn.MultiheadAttention(self.channel, self.head, dropout=self.dropout_n, batch_first=True).to(self.device)
        self.fusion_norm = nn.LayerNorm(self.channel).to(self.device)

        self.prompt_encoder = nn.ModuleList([GatedTransformerEncoderLayer(self.d_llm, self.head, dropout=self.dropout_n) for _ in range(self.e_layer)]).to(self.device)
        
        from layers.Cross_Modal_Align import CrossModal
        self.num_cma_heads = 4
        self.cma_heads = nn.ModuleList([
            CrossModal(
                d_model=self.num_nodes, n_heads=1, d_ff=self.d_ff, norm='LayerNorm',
                attn_dropout=self.dropout_n, dropout=self.dropout_n, pre_norm=True,
                activation="gelu", res_attention=True, n_layers=1
            ).to(self.device)
            for _ in range(self.num_cma_heads)
        ])
        
        from models.T3Time import AdaptiveDynamicHeadsCMA
        self.adaptive_dynamic_heads_cma = AdaptiveDynamicHeadsCMA(self.num_cma_heads, self.num_nodes, self.channel, self.device)
        self.residual_alpha = nn.Parameter(torch.ones(self.channel) * 0.5).to(self.device)
        
        self.decoder_layer = nn.TransformerDecoderLayer(self.channel, self.head, batch_first=True, norm_first=True, dropout=self.dropout_n).to(self.device)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=self.d_layer).to(self.device)
        self.c_to_length = nn.Linear(self.channel, self.pred_len).to(self.device)

    def forward(self, input_data, input_data_mark, embeddings):
        input_data = input_data.float()
        embeddings = embeddings.float().squeeze(-1).permute(0, 2, 1)
        
        input_data_norm = self.normalize_layers(input_data, 'norm')
        input_data_norm = input_data_norm.permute(0, 2, 1) # [B, N, L]
        
        # 小波域
        wavelet_coeffs = self.wavelet_transform(input_data_norm)
        wavelet_feats = []
        for i, node in enumerate(wavelet_coeffs):
            B, N, L_part = node.shape
            tokens = self.wavelet_proj[min(i, 9)](node.unsqueeze(-1).reshape(B*N, L_part, 1))
            wavelet_feats.append(self.wavelet_encoder(tokens))
        wavelet_features = self.wavelet_pool(wavelet_feats).reshape(B, N, self.channel)
        
        # 时域
        time_encoded = self.length_to_feature(input_data_norm)
        for layer in self.ts_encoder: time_encoded = layer(time_encoded)
        
        # 融合
        fused_attn, _ = self.cross_attn_fusion(time_encoded, wavelet_features, wavelet_features)
        fused_features = self.fusion_norm(fused_attn + time_encoded).permute(0, 2, 1) # [B, C, N]
        
        # Prompt
        prompt_feat = embeddings
        for layer in self.prompt_encoder: prompt_feat = layer(prompt_feat)
        prompt_feat = prompt_feat.permute(0, 2, 1) # [B, d_llm, N]
        
        # CMA
        cma_outs = [h(fused_features, prompt_feat, prompt_feat) for h in self.cma_heads]
        fused_cma = self.adaptive_dynamic_heads_cma(cma_outs)
        alpha = self.residual_alpha.view(1, -1, 1)
        cross_out = (alpha * fused_cma + (1 - alpha) * fused_features).permute(0, 2, 1)
        
        # 解码
        dec_out = self.decoder(cross_out, cross_out)
        dec_out = self.c_to_length(dec_out).permute(0, 2, 1)
        return self.normalize_layers(dec_out, 'denorm')

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

