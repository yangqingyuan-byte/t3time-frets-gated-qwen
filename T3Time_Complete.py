"""
T3Time 完整独立模型文件
包含所有必要的依赖组件，可直接导入使用
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ==================== 依赖层：StandardNorm ====================
class Normalize(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=False, subtract_last=False, non_norm=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
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
        # initialize RevIN params: (C,)
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
import numpy as np

class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module"""
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


# ==================== T3Time 模型核心组件 ====================
class GatedTransformerEncoderLayer(nn.Module):
    """
    带 SDPA 输出门控的 Encoder Layer（受 Gated Attention 论文启发）
    - 在多头自注意力输出后加一层 element-wise sigmoid gate：
        attn_out = MHA(...)
        gate     = sigmoid(W_g(attn_out))
        attn_out = attn_out * gate
      然后再走残差 + FFN。
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.gate_proj = nn.Linear(d_model, d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.ffn_dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        residual = src
        src_norm = self.norm1(src)
        attn_out, _ = self.self_attn(src_norm, src_norm, src_norm)
        gate = torch.sigmoid(self.gate_proj(attn_out))
        attn_out = attn_out * gate
        src = residual + self.attn_dropout(attn_out)
        residual = src
        src_norm = self.norm2(src)
        ffn_out = self.linear2(
            self.dropout(self.activation(self.linear1(src_norm)))
        )
        src = residual + self.ffn_dropout(ffn_out)
        return src

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

class RichHorizonGate(nn.Module):
    """
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
        B, C, N = embedding.size()
        pooled_embed = embedding.mean(dim=2)
        horizon_tensor = torch.full((B, 1), float(horizon) / 1000.0, device=embedding.device)
        gating_input = torch.cat([pooled_embed, horizon_tensor], dim=1)
        gate = self.gate_mlp(gating_input).unsqueeze(-1)
        return gate
    
class FrequencyAttentionPooling(nn.Module):
    """
    Learnable, attention-weighted pooling over frequency bins
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.freq_attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )

    def forward(self, freq_enc_out):
        attn_logits  = self.freq_attention(freq_enc_out)
        attn_weights = F.softmax(attn_logits, dim=1)
        pooled_freq  = (freq_enc_out * attn_weights).sum(dim=1)
        return pooled_freq


# ==================== T3Time 主模型 ====================
class TriModal(nn.Module):
    def __init__(
        self,
        device = "cuda:7",
        channel = 32,
        num_nodes = 7,
        seq_len = 96,
        pred_len = 96,
        dropout_n = 0.1,
        d_llm = 768,
        e_layer = 1,
        d_layer = 1,
        d_ff=32,
        head =8
    ):
        super().__init__()

        self.device = device
        self.channel = channel
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.dropout_n= dropout_n
        self.d_llm = d_llm
        self.e_layer = e_layer
        self.d_layer = d_layer
        self.d_ff = d_ff
        self.head = head

        self.normalize_layers = Normalize(self.num_nodes, affine=False).to(self.device)
        self.length_to_feature = nn.Linear(self.seq_len, self.channel).to(self.device)
        self.num_cma_heads = 4 

        # Time Series Encoder（改为带 SDPA 门控的 Encoder Layer）
        self.ts_encoder = GatedTransformerEncoderLayer(
            d_model=self.channel,
            nhead=self.head,
            dim_feedforward=4 * self.channel,
            dropout=self.dropout_n,
        ).to(self.device)

        # Prompt Encoder（同样使用 Gated Attention）
        self.prompt_encoder = GatedTransformerEncoderLayer(
            d_model=self.d_llm,
            nhead=self.head,
            dim_feedforward=4 * self.d_llm,
            dropout=self.dropout_n,
        ).to(self.device)
        
        # Spectral Encoder（FFT 频域编码使用 Gated Attention）
        self.Lf = seq_len // 2 + 1   
        self.freq_token_proj = nn.Linear(1, self.channel).to(self.device)
        self.freq_encoder = GatedTransformerEncoderLayer(
            d_model=self.channel,
            nhead=self.head,
            dim_feedforward=4 * self.channel,
            dropout=self.dropout_n,
        ).to(self.device)

        self.freq_pool = FrequencyAttentionPooling(self.channel).to(self.device)
        self.rich_horizon_gate = RichHorizonGate(self.channel).to(self.device)

        # multi head CMA
        self.cma_heads = nn.ModuleList([
            CrossModal(d_model= self.num_nodes, n_heads= 1, d_ff=self.d_ff, norm='LayerNorm', attn_dropout=self.dropout_n, 
                                dropout=self.dropout_n, pre_norm=True, activation="gelu", res_attention=True, n_layers=1, 
                                store_attn=False).to(self.device)
            for _ in range(self.num_cma_heads)
        ])

        # Aggregate multi heads
        self.adaptive_dynamic_heads_cma = AdaptiveDynamicHeadsCMA(num_heads=self.num_cma_heads, num_nodes=self.num_nodes, channel=self.channel, device=self.device)
   
        # Residual connection 
        self.residual_alpha = nn.Parameter(torch.ones(self.channel) * 0.5).to(self.device)  

        # Transformer decoder
        self.decoder_layer = nn.TransformerDecoderLayer(d_model = self.channel, nhead = self.head, batch_first=True, norm_first = True, dropout = self.dropout_n).to(self.device)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers = self.d_layer).to(self.device)

        # Projection
        self.c_to_length = nn.Linear(self.channel, self.pred_len, bias=True).to(self.device)

    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def frequency_domain_processing(self, input_data):
        freq_complex    = torch.fft.rfft(input_data, dim=-1)
        freq_mag        = torch.abs(freq_complex)
        B, N, Lf        = freq_mag.shape
        
        freq_tokens = freq_mag.unsqueeze(-1)
        freq_tokens = freq_tokens.reshape(B*N, Lf, 1)
        freq_tokens = self.freq_token_proj(freq_tokens)

        freq_enc_out = self.freq_encoder(freq_tokens)
        freq_enc_out = self.freq_pool(freq_enc_out)

        freq_enc_out = freq_enc_out.reshape(B, N, self.channel)
        return freq_enc_out
    

    def forward(self, input_data, input_data_mark, embeddings):
        input_data = input_data.float()
        input_data_mark = input_data_mark.float()
        embeddings = embeddings.float()
        embeddings = embeddings.squeeze(-1)
        embeddings = embeddings.permute(0,2,1)

        #------ RevIN
        input_data = self.normalize_layers(input_data, 'norm')
        input_data = input_data.permute(0,2,1)

        #------ Frequency Encoding
        freq_enc_out = self.frequency_domain_processing(input_data)
        input_data = self.length_to_feature(input_data)

        #------ Time Series Encoding
        enc_out = self.ts_encoder(input_data)
        enc_out = enc_out.permute(0,2,1)
  
        #------ Rich Horizon Gate
        gate = self.rich_horizon_gate(enc_out, self.pred_len)
        enc_out = gate * freq_enc_out.permute(0,2,1) + (1 - gate) * enc_out
        
        #------ Prompt encoding 
        embeddings = self.prompt_encoder(embeddings)
        embeddings = embeddings.permute(0,2,1)

        #------ Aggregating Multiple CMA Heads
        cma_outputs = []
        for cma_head in self.cma_heads:
            head_out = cma_head(enc_out, embeddings, embeddings)
            cma_outputs.append(head_out)

        fused = self.adaptive_dynamic_heads_cma(cma_outputs)

        #------ Residual Fusion 
        B, C, N = fused.shape
        alpha = self.residual_alpha.view(1, C, 1)      
        cross_out = alpha * fused + (1 - alpha) * enc_out
        cross_out = cross_out.permute(0, 2, 1)

        #------ Decoder
        dec_out = self.decoder(cross_out, cross_out)

        #------ Projection
        dec_out = self.c_to_length(dec_out)
        dec_out = dec_out.permute(0,2,1)

        #------ Denorm
        dec_out = self.normalize_layers(dec_out, 'denorm')

        return dec_out

