"""
T3Time_FreEformer_Gated_Qwen
将 T3Time_FreTS_Gated_Qwen 中的 FreTS Component（可学习频域MLP）
替换为 FreEformer 的频域注意力机制
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.StandardNorm import Normalize
from layers.Cross_Modal_Align import CrossModal
# 使用 FreEformer 的组件
import sys
import os
freeformer_components_path = os.path.join(os.path.dirname(__file__), 'frEformer组件')
if freeformer_components_path not in sys.path:
    sys.path.insert(0, freeformer_components_path)

# 临时处理缺失的工具函数（如果不存在）
try:
    from utils.tools import hier_half_token_weight, create_sin_pos_embed
except ImportError:
    def hier_half_token_weight(x):
        if x is None:
            return None
        return x[:, ::2] if x.shape[1] > 1 else x
    
    def create_sin_pos_embed(*args, **kwargs):
        return None

try:
    from utils.CKA import CudaCKA
except ImportError:
    CudaCKA = None

try:
    from utils.tools import create_sub_diagonal_matrix, plot_mat
except ImportError:
    def create_sub_diagonal_matrix(*args, **kwargs):
        return None
    
    def plot_mat(*args, **kwargs):
        pass

# 现在导入组件
from Transformer_EncDec import Encoder_ori, EncoderLayer
from SelfAttention_Family import AttentionLayer, FullAttention_ablation

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

class FreEformerComponent(nn.Module):
    """
    FreEformer Component: 使用 Transformer 注意力机制的频域处理
    基于 FreEformer 的频域注意力机制，但适配 T3Time 的输入输出格式
    """
    def __init__(self, channel, seq_len, pred_len, embed_size=16, e_layer=1, 
                 d_model=None, d_ff=None, n_heads=8, dropout=0.1, device="cuda",
                 attn_enhance=None, attn_softmax_flag=True, attn_weight_plus=False, 
                 attn_outside_softmax=False):
        super().__init__()
        self.channel = channel
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.embed_size = embed_size
        self.device = device
        
        # 强制使用 channel 作为 d_model（为了兼容性和稳定性）
        # 忽略传入的 d_model 和 d_ff 参数，避免维度不匹配问题
        d_model = channel
        d_ff = channel * 4
        
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.attn_enhance = attn_enhance
        self.attn_softmax_flag = attn_softmax_flag
        self.attn_weight_plus = attn_weight_plus
        self.attn_outside_softmax = attn_outside_softmax
        
        # 计算频域点数
        self.valid_fre_points = int((self.seq_len + 1) / 2 + 0.5)  # 输入频域点数
        self.valid_fre_points2 = int((self.pred_len + 1) / 2 + 0.5)  # 输出频域点数
        
        # Embedding（用于 token embedding）
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size)).to(device)
        
        # 频域 Transformer Encoder（实部）- 使用 Encoder_ori
        self.encoder_fre_real = Encoder_ori(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention_ablation(
                            False, 1, attention_dropout=dropout,
                            output_attention=False, token_num=channel,
                            SF_mode=attn_enhance, softmax_flag=attn_softmax_flag,
                            weight_plus=attn_weight_plus, outside_softmax=attn_outside_softmax,
                            plot_mat_flag=False, plot_grad_flag=False,
                            save_folder='./'  # 提供默认路径
                        ),
                        d_model, n_heads
                    ),
                    d_model, d_ff, dropout=dropout, activation='gelu'
                ) for _ in range(e_layer)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
            one_output=True,
            CKA_flag=False
        ).to(device)
        
        # 频域 Transformer Encoder（虚部）- 使用 Encoder_ori
        self.encoder_fre_imag = Encoder_ori(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention_ablation(
                            False, 1, attention_dropout=dropout,
                            output_attention=False, token_num=channel,
                            SF_mode=attn_enhance, softmax_flag=attn_softmax_flag,
                            weight_plus=attn_weight_plus, outside_softmax=attn_outside_softmax,
                            plot_mat_flag=False, plot_grad_flag=False,
                            save_folder='./'  # 提供默认路径
                        ),
                        d_model, n_heads
                    ),
                    d_model, d_ff, dropout=dropout, activation='gelu'
                ) for _ in range(e_layer)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
            one_output=True,
            CKA_flag=False
        ).to(device)
        
        # 频域变换（实部）- 需要单独处理，因为 Encoder_ori 需要 3D 输入
        self.fre_proj_in_real = nn.Linear(self.valid_fre_points * self.embed_size, self.d_model).to(device)
        self.fre_proj_out_real = nn.Linear(self.d_model, self.valid_fre_points2).to(device)
        
        # 频域变换（虚部）
        self.fre_proj_in_imag = nn.Linear(self.valid_fre_points * self.embed_size, self.d_model).to(device)
        self.fre_proj_out_imag = nn.Linear(self.d_model, self.valid_fre_points2).to(device)
        
        self.dropout = nn.Dropout(dropout).to(device)
    
    def tokenEmb(self, x, embeddings):
        """
        Token embedding（从 FreEformer）
        x: [B*N, L, C] -> [B*N, L, C, D]
        """
        B_N, L, C = x.shape
        if self.embed_size <= 1:
            return x.unsqueeze(-1)
        # [B*N, L, C, 1] * [1, D] -> [B*N, L, C, D]
        x = x.unsqueeze(-1)
        return x * embeddings
    
    def forward(self, x):
        """
        FreEformer 频域处理
        Args:
            x: [B*N, L, C] 输入特征
        Returns:
            out: [B*N, L, C] 输出特征（保持相同长度）
        """
        B_N, L, C = x.shape
        assert L == self.seq_len, f"输入长度 {L} 与 seq_len {self.seq_len} 不匹配"
        
        # 1. Token embedding: [B*N, L, C] -> [B*N, L, C, D]
        x_emb = self.tokenEmb(x, self.embeddings)  # [B*N, L, C, D]
        
        # 2. 重新组织维度以适配 FreEformer: [B*N, L, C, D] -> [B*N, C, L, D]
        x_reshaped = x_emb.permute(0, 2, 1, 3)  # [B*N, C, L, D]
        
        # 3. 将 (L, D) 展平为单一维度: [B*N, C, L, D] -> [B*N, C, L*D]
        # 但为了 FFT，我们需要在 L 维度上进行，所以保持 [B*N, C, L, D]
        # 实际上，FreEformer 在 L 维度上做 FFT，所以我们需要 [B*N, C, L]
        # 简化：对每个 channel 分别处理，或者将 D 维度合并
        
        # 方案：将 [B*N, C, L, D] 视为 [B*N*C, L, D]，然后在 L 维度上做 FFT
        B_N_C, L2, D = x_reshaped.reshape(B_N * C, L, self.embed_size).shape
        
        # 4. FFT: 在 L 维度上进行 FFT -> [B*N*C, fre_points]
        x_fre = torch.fft.rfft(
            x_reshaped.reshape(B_N * C, L, self.embed_size).mean(dim=-1),  # [B*N*C, L] (对embed_size维度求平均)
            dim=1, norm='ortho'
        )  # [B*N*C, fre_points]
        assert x_fre.shape[1] == self.valid_fre_points
        
        # 5. 分离实部和虚部
        y_real, y_imag = x_fre.real, x_fre.imag  # 都是 [B*N*C, fre_points]
        
        # 6. 扩展维度以匹配 Transformer 输入要求
        # Transformer 需要 [B*N*C, fre_points*embed_size] 的输入
        y_real_expanded = y_real.unsqueeze(-1).expand(-1, -1, self.embed_size).reshape(
            B_N * C, self.valid_fre_points * self.embed_size
        )  # [B*N*C, fre_points*embed_size]
        y_imag_expanded = y_imag.unsqueeze(-1).expand(-1, -1, self.embed_size).reshape(
            B_N * C, self.valid_fre_points * self.embed_size
        )  # [B*N*C, fre_points*embed_size]
        
        # 7. Transformer 处理（实部和虚部分别处理）
        # 投影到 d_model: [B*N*C, fre_points*embed_size] -> [B*N*C, d_model]
        y_real_proj = self.fre_proj_in_real(y_real_expanded)  # [B*N*C, d_model]
        y_imag_proj = self.fre_proj_in_imag(y_imag_expanded)  # [B*N*C, d_model]
        
        # Encoder_ori 需要 3D 输入 [B, L, D]，所以添加序列维度
        # 将每个样本视为一个长度为1的序列: [B*N*C, d_model] -> [B*N*C, 1, d_model]
        y_real_enc_input = y_real_proj.unsqueeze(1)  # [B*N*C, 1, d_model]
        y_imag_enc_input = y_imag_proj.unsqueeze(1)  # [B*N*C, 1, d_model]
        
        # Encoder_ori 处理（one_output=True，直接返回输出，不返回attns）
        y_real_enc = self.encoder_fre_real(y_real_enc_input)  # [B*N*C, 1, d_model]
        y_imag_enc = self.encoder_fre_imag(y_imag_enc_input)  # [B*N*C, 1, d_model]
        
        # 移除序列维度: [B*N*C, 1, d_model] -> [B*N*C, d_model]
        y_real_enc = y_real_enc.squeeze(1)  # [B*N*C, d_model]
        y_imag_enc = y_imag_enc.squeeze(1)  # [B*N*C, d_model]
        
        # 投影到输出维度: [B*N*C, d_model] -> [B*N*C, fre_points2]
        y_real_trans = self.fre_proj_out_real(y_real_enc)  # [B*N*C, fre_points2]
        y_imag_trans = self.fre_proj_out_imag(y_imag_enc)  # [B*N*C, fre_points2]
        
        # 8. 重新组合为复数
        y = torch.complex(y_real_trans, y_imag_trans)  # [B*N*C, fre_points2]
        
        # 9. IFFT 转回时域: [B*N*C, fre_points2] -> [B*N*C, pred_len]
        x_out = torch.fft.irfft(y, n=self.pred_len, dim=1, norm='ortho')  # [B*N*C, pred_len]
        
        # 10. 重新组织维度: [B*N*C, pred_len] -> [B*N, C, pred_len] -> [B*N, pred_len, C]
        x_out = x_out.reshape(B_N, C, self.pred_len).permute(0, 2, 1)  # [B*N, pred_len, C]
        
        # 11. 如果 pred_len != seq_len，需要调整到 seq_len
        if self.pred_len != self.seq_len:
            # 使用插值调整到 seq_len
            x_out = F.interpolate(
                x_out.permute(0, 2, 1),  # [B*N, C, pred_len]
                size=self.seq_len,
                mode='linear',
                align_corners=False
            ).permute(0, 2, 1)  # [B*N, seq_len, C]
        
        return self.dropout(x_out)

class AttentionPooling(nn.Module):
    """注意力池化"""
    def __init__(self, embed_dim):
        super().__init__()
        self.attention = nn.Sequential(nn.Linear(embed_dim, embed_dim // 2), nn.ReLU(), nn.Linear(embed_dim // 2, 1))
    def forward(self, x):
        attn_weights = F.softmax(self.attention(x), dim=1)
        return (x * attn_weights).sum(dim=1)

class TriModalFreEformerGatedQwen(nn.Module):
    """
    T3Time_FreEformer_Gated_Qwen
    将 T3Time_FreTS_Gated_Qwen 中的 FreTS Component 替换为 FreEformer 的频域注意力机制
    """
    def __init__(self, device="cuda", channel=32, num_nodes=7, seq_len=96, pred_len=96, 
                 dropout_n=0.1, d_llm=1024, e_layer=1, d_layer=1, d_ff=32, head=8,
                 embed_size=16, fre_e_layer=1, d_model=None, fre_d_ff=None,
                 attn_enhance=None, attn_softmax_flag=True, attn_weight_plus=False, 
                 attn_outside_softmax=False):
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
        
        # 频域分支：使用 FreEformer Component
        self.fre_projection = nn.Linear(1, self.channel).to(self.device)
        # 为了兼容性，强制使用 channel 作为 d_model（忽略传入的 d_model 和 d_ff）
        self.freeformer_branch = FreEformerComponent(
            channel=self.channel,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            embed_size=embed_size,
            e_layer=fre_e_layer,
            d_model=None,  # 使用 None，让内部自动设置为 channel
            d_ff=None,     # 使用 None，让内部自动设置为 channel * 4
            n_heads=head,
            dropout=dropout_n,
            device=self.device,
            attn_enhance=attn_enhance,
            attn_softmax_flag=attn_softmax_flag,
            attn_weight_plus=attn_weight_plus,
            attn_outside_softmax=attn_outside_softmax
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
        
        # 频域处理：使用 FreEformer Component
        fre_input = self.fre_projection(x_perm.reshape(B*N, L, 1))
        fre_processed = self.freeformer_branch(fre_input)
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
