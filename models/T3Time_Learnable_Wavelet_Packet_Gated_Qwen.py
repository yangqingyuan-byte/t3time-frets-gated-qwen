"""
T3Time with Learnable Wavelet Packet Decomposition (LWPD), Qwen3-0.6B, and Gated Attention.
核心改进：将固定的小波包分解替换为基于提升方案（Lifting Scheme）的可学习小波包分解。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
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

class LearnableLiftingStep(nn.Module):
    """
    基于提升方案的可学习小波步 (Split, Predict, Update)
    """
    def __init__(self):
        super().__init__()
        # 预测器：用偶数序列预测奇数序列的残差
        self.predict = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(4, 1, kernel_size=3, padding=1)
        )
        # 更新器：用残差更新偶数序列以保持统计特性
        self.update = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(4, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # x: [B*N, 1, L]
        # 1. Split (甚至/奇数分裂)
        x_even = x[:, :, 0::2]
        x_odd = x[:, :, 1::2]
        
        # 确保长度一致
        min_len = min(x_even.shape[-1], x_odd.shape[-1])
        x_even = x_even[:, :, :min_len]
        x_odd = x_odd[:, :, :min_len]

        # 2. Predict (生成细节分量 d)
        # d = x_odd - P(x_even)
        d = x_odd - self.predict(x_even)
        
        # 3. Update (生成近似分量 a)
        # a = x_even + U(d)
        a = x_even + self.update(d)
        
        return a, d

class LearnableWaveletPacketTransform(nn.Module):
    """
    可学习的小波包分解：对所有分支递归应用提升步
    """
    def __init__(self, level=2):
        super().__init__()
        self.level = level
        # 为每一层和每一个节点创建独立的提升步，或者共享。
        # 这里我们每一层使用相同的提升算子，但在不同分支递归。
        self.lifting_layers = nn.ModuleList([
            LearnableLiftingStep() for _ in range(level)
        ])
    
    def forward(self, x):
        B, N, L = x.shape
        # x: [B, N, L] -> [B*N, 1, L]
        x_reshaped = x.reshape(B * N, 1, L)
        
        nodes = [x_reshaped]
        for l in range(self.level):
            next_nodes = []
            lifting_step = self.lifting_layers[l]
            for node in nodes:
                a, d = lifting_step(node)
                next_nodes.append(a)
                next_nodes.append(d)
            nodes = next_nodes
        
        # nodes 是 2^level 个张量，每个形状 [B*N, 1, L_part]
        return nodes

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
            pooled_list.append(pooled * self.node_weights[i])
        
        final_pooled = torch.stack(pooled_list, dim=1).mean(dim=1) # [B*N, C]
        return final_pooled

class TriModalLearnableWaveletPacketGatedQwen(nn.Module):
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
        wp_level=2
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
        
        # 可学习的小波包分解
        self.wp_transform = LearnableWaveletPacketTransform(level=wp_level).to(self.device)
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
        wp_nodes = self.wp_transform(x) # List of 2^level tensors [B*N, 1, L_part]
        
        wp_features_list = []
        for i, node in enumerate(wp_nodes):
            # node: [B*N, 1, L_part]
            tokens = node.transpose(1, 2) # [B*N, L_part, 1]
            proj = self.wp_proj_layers[i]
            tokens = proj(tokens) # [B*N, L_part, C]
            
            # 位置编码
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
        
        # 1. 可学习小波包处理
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
