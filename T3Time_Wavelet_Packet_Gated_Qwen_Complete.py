"""
T3Time_Wavelet_Packet_Gated_Qwen 完整独立模型文件
包含所有必要的依赖组件，可直接导入使用
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np


# ==================== 依赖层：StandardNorm ====================
class Normalize(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=False, subtract_last=False, non_norm=False):
        super(Normalize, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.non_norm:
            return x
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.non_norm:
            return x
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


# ==================== 依赖层：TS_Pos_Enc ====================
class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)

def get_activation_fn(activation):
    if callable(activation): return activation()
    elif activation.lower() == "relu": return nn.ReLU()
    elif activation.lower() == "gelu": return nn.GELU()
    raise ValueError(f'{activation} is not available. You can use "relu", "gelu", or a callable')


# ==================== 依赖层：Cross_Modal_Align ====================
from typing import Callable, Optional

class _ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q, k, v, prev=None, key_padding_mask=None, attn_mask=None):
        attn_scores = torch.matmul(q, k) * self.scale
        if prev is not None: attn_scores = attn_scores + prev
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask
        if key_padding_mask is not None:
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        output = torch.matmul(attn_weights, v)
        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights

class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v
        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa)
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))

    def forward(self, Q, K=None, V=None, prev=None, key_padding_mask=None, attn_mask=None):
        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v)
        output = self.to_out(output)
        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights

class TSTEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='LayerNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.LayerNorm(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.LayerNorm(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)
        self.pre_norm = pre_norm
        self.store_attn = store_attn

    def forward(self, q, k, v, prev=None, key_padding_mask=None, attn_mask=None):
        if self.pre_norm:
            q = self.norm_attn(q)
            k = self.norm_attn(k)
            v = self.norm_attn(v)
        if self.res_attention:
            q2, attn, scores = self.self_attn(q, k, v, prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            q2, attn = self.self_attn(q, k, v, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        if self.store_attn:
            self.attn = attn
        q = q + self.dropout_attn(q2)
        if not self.pre_norm:
            q = self.norm_attn(q)
        if self.pre_norm:
            q = self.norm_ffn(q)
        q2 = self.ff(q)
        q = q + self.dropout_ffn(q2)
        if not self.pre_norm:
            q = self.norm_ffn(q)
        if self.res_attention:
            return q, scores
        else:
            return q

class CrossModal(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, d_ff=None, 
                        norm='LayerNorm', attn_dropout=0., dropout=0., activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()
        self.layers = nn.ModuleList([TSTEncoderLayer( d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                      attn_dropout=attn_dropout, dropout=dropout,
                                                      activation=activation, res_attention=res_attention,
                                                      pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, q, k, v, key_padding_mask=None, attn_mask=None):
        scores = None
        if self.res_attention:
            for mod in self.layers: output, scores = mod(q,k,v,  prev=scores, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output
        else:
            for mod in self.layers: output = mod(q,k,v, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output


# ==================== 从 T3Time 继承的组件：AdaptiveDynamicHeadsCMA ====================
class AdaptiveDynamicHeadsCMA(nn.Module):
    """
    Small network to compute per-head weights from the concatenated heads
    Input shape: [B, N, H*C]      Output shape: [B, N, H]
    """
    def __init__(self, num_heads, num_nodes, channel, device):
        super().__init__()
        self.num_heads = num_heads
        self.num_nodes = num_nodes
        self.channel = channel
        self.device = device
        
        self.gate_mlp = nn.Sequential(
            nn.Linear(num_heads * channel, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_heads)
        ).to(device)

    def forward(self, cma_outputs):
        B, C, N = cma_outputs[0].shape
        H = self.num_heads
        combined = torch.cat(cma_outputs, dim=1)
        combined_permute = combined.permute(0, 2, 1)
        gates = self.gate_mlp(combined_permute)
        gates = F.softmax(gates, dim=-1)
        combined_heads = combined.view(B, H, C, N).permute(0, 1, 3, 2)
        gates = gates.permute(0, 2, 1).unsqueeze(-1)
        weighted_heads = combined_heads * gates
        weighted_heads = weighted_heads.permute(0, 1, 3, 2)
        fused = weighted_heads.sum(dim=1)
        return fused


# ==================== T3Time_Wavelet_Packet_Gated_Qwen 模型核心组件 ====================
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


# ==================== T3Time_Wavelet_Packet_Gated_Qwen 主模型 ====================
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
        
        self.num_cma_heads = 4
        self.cma_heads = nn.ModuleList([
            CrossModal(
                d_model=self.num_nodes, n_heads=1, d_ff=self.d_ff, norm='LayerNorm',
                attn_dropout=self.dropout_n, dropout=self.dropout_n, pre_norm=True,
                activation="gelu", res_attention=True, n_layers=1, store_attn=False
            ).to(self.device)
            for _ in range(self.num_cma_heads)
        ])
        
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

