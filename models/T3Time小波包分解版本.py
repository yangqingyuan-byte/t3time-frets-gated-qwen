"""
T3Time with Wavelet Packet Decomposition (WPD), Qwen3-0.6B, and Gated Attention.
核心改进：采用小波包分解（Wavelet Packet Decomposition），对时序信号进行更精细的频段切分。
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

class WaveletPacketTransform(nn.Module):
    """
    小波包分解 (WPD) 模块：对低频和高频部分同时进行递归分解
    """
    def __init__(self, wavelet='db4', level=2):
        super().__init__()
        self.wavelet = wavelet
        self.level = level
        wavelet_obj = pywt.Wavelet(wavelet)
        self.register_buffer('dec_lo', torch.tensor(wavelet_obj.dec_lo, dtype=torch.float32).view(1, 1, -1))
        self.register_buffer('dec_hi', torch.tensor(wavelet_obj.dec_hi, dtype=torch.float32).view(1, 1, -1))
    
    def _dwt_step(self, x):
        B, N, L = x.shape
        pad_len = self.dec_lo.shape[-1] - 1
        # 使用 reshape 替代 view 以处理不连续张量
        x_padded = F.pad(x.reshape(B*N, 1, -1), (pad_len, pad_len), mode='circular')
        cA = F.conv1d(x_padded, self.dec_lo, stride=2)
        cD = F.conv1d(x_padded, self.dec_hi, stride=2)
        return cA.reshape(B, N, -1), cD.reshape(B, N, -1)
    
    def forward(self, x):
        # 初始输入: [cA]
        nodes = [x]
        # 递归分解 level 次，生成 2^level 个叶子节点
        for _ in range(self.level):
            next_nodes = []
            for node in nodes:
                cA, cD = self._dwt_step(node)
                next_nodes.append(cA)
                next_nodes.append(cD)
            nodes = next_nodes
        return nodes # 返回叶子节点列表

class WaveletPacketAttentionPooling(nn.Module):
    def __init__(self, embed_dim, num_nodes_wp):
        super().__init__()
        self.node_weights = nn.Parameter(torch.ones(num_nodes_wp))
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )
    
    def forward(self, wp_features):
        # wp_features: List of [B*N, L_part, C]
        pooled_list = []
        for i, feat in enumerate(wp_features):
            attn_logits = self.attention(feat) # [B*N, L_part, 1]
            attn_weights = F.softmax(attn_logits, dim=1)
            pooled = (feat * attn_weights).sum(dim=1) # [B*N, C]
            # 应用可学习的节点权重
            pooled_list.append(pooled * self.node_weights[i])
        
        # 将所有频段池化结果融合
        final_pooled = torch.stack(pooled_list, dim=1).mean(dim=1) # [B*N, C]
        return final_pooled

class TriModalWaveletPacketGatedQwen(nn.Module):
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
        wp_level=2 # 小波包分解层数，2层产生4个频段
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
        self.wp_level = wp_level
        
        self.normalize_layers = Normalize(self.num_nodes, affine=False).to(self.device)
        self.length_to_feature = nn.Linear(self.seq_len, self.channel).to(self.device)
        
        self.ts_encoder = nn.ModuleList([
            GatedTransformerEncoderLayer(d_model=self.channel, nhead=self.head, dropout=self.dropout_n)
            for _ in range(self.e_layer)
        ]).to(self.device)
        
        # 小波包分解
        self.wp_transform = WaveletPacketTransform(wavelet=wavelet, level=wp_level).to(self.device)
        num_wp_nodes = 2 ** wp_level
        
        # 每个频段独立投影
        self.wp_proj_layers = nn.ModuleList([
            nn.Linear(1, self.channel).to(self.device)
            for _ in range(num_wp_nodes)
        ])
        
        self.wp_encoder = GatedTransformerEncoderLayer(
            d_model=self.channel, nhead=self.head, dropout=self.dropout_n
        ).to(self.device)
        
        self.wp_pool = WaveletPacketAttentionPooling(self.channel, num_wp_nodes).to(self.device)
        
        # 频率位置编码（区分不同频段）
        self.freq_pos_embed = nn.Parameter(torch.zeros(1, num_wp_nodes, self.channel)).to(self.device)
        
        self.cross_attn_fusion = nn.MultiheadAttention(self.channel, self.head, dropout=self.dropout_n, batch_first=True).to(self.device)
        self.fusion_norm = nn.LayerNorm(self.channel).to(self.device)

        self.prompt_encoder = nn.ModuleList([
            GatedTransformerEncoderLayer(d_model=self.d_llm, nhead=self.head, dropout=self.dropout_n)
            for _ in range(self.e_layer)
        ]).to(self.device)
        
        from layers.Cross_Modal_Align import CrossModal
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

    def wp_domain_processing(self, x):
        B, N, L = x.shape
        # x: [B, N, L]
        wp_nodes = self.wp_transform(x) # List of 2^level tensors [B, N, L_part]
        
        wp_features_list = []
        for i, node in enumerate(wp_nodes):
            L_part = node.shape[-1]
            node_reshaped = node.unsqueeze(-1).reshape(B * N, L_part, 1) # [B*N, L_part, 1]
            proj = self.wp_proj_layers[i]
            tokens = proj(node_reshaped) # [B*N, L_part, C]
            
            # 添加频段特定的偏差（简易位置编码）
            tokens = tokens + self.freq_pos_embed[:, i, :].unsqueeze(1)
            
            # 通过带门控的编码器
            encoded = self.wp_encoder(tokens)
            wp_features_list.append(encoded)
        
        # 池化
        pooled = self.wp_pool(wp_features_list) # [B*N, C]
        return pooled.reshape(B, N, self.channel)

    def forward(self, input_data, input_data_mark, embeddings):
        input_data = input_data.float()
        embeddings = embeddings.float().squeeze(-1).permute(0, 2, 1)
        
        input_data_norm = self.normalize_layers(input_data, 'norm') # [B, L, N]
        input_data_norm = input_data_norm.permute(0, 2, 1)           # [B, N, L]
        
        # 1. 小波包处理
        wp_features = self.wp_domain_processing(input_data_norm) # [B, N, C]
        
        # 2. 时域处理
        time_features = self.length_to_feature(input_data_norm)
        time_encoded = time_features
        for layer in self.ts_encoder:
            time_encoded = layer(time_encoded)
        
        # 3. 融合
        fused_attn, _ = self.cross_attn_fusion(time_encoded, wp_features, wp_features)
        fused_features = self.fusion_norm(fused_attn + time_encoded)
        fused_features = fused_features.permute(0, 2, 1) # [B, C, N]
        
        # 4. Prompt 处理
        prompt_feat = embeddings
        for layer in self.prompt_encoder:
            prompt_feat = layer(prompt_feat)
        prompt_feat = prompt_feat.permute(0, 2, 1) # [B, d_llm, N]
        
        # 5. 跨模态对齐
        cma_outputs = []
        for cma_head in self.cma_heads:
            head_out = cma_head(fused_features, prompt_feat, prompt_feat)
            cma_outputs.append(head_out)
        
        fused_cma = self.adaptive_dynamic_heads_cma(cma_outputs)
        alpha = self.residual_alpha.view(1, -1, 1)
        cross_out = alpha * fused_cma + (1 - alpha) * fused_features
        cross_out = cross_out.permute(0, 2, 1)
        
        # 6. 解码
        dec_out = self.decoder(cross_out, cross_out)
        dec_out = self.c_to_length(dec_out).permute(0, 2, 1)
        dec_out = self.normalize_layers(dec_out, 'denorm')
        return dec_out

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

