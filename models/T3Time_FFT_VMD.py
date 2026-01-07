"""
T3Time 的 FFT + VMD 并行融合版本
保留原有的 FFT 频域分支，新增 VMD 作为辅助特征，两者并行融合
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.StandardNorm import Normalize
from layers.Cross_Modal_Align import CrossModal
from models.T3Time_VMD import VMDEncoder

# 从 T3Time.py 中导入需要的组件
from models.T3Time import (
    AdaptiveDynamicHeadsCMA,
    RichHorizonGate,
    FrequencyAttentionPooling,
)


class TriModalFFTVMD(nn.Module):
    """
    三模态模型：时域 + FFT频域 + VMD辅助特征
    VMD 作为辅助特征，与 FFT 并行使用，而不是替换 FFT
    """
    def __init__(
        self,
        device="cuda",
        channel=32,
        num_nodes=7,
        seq_len=96,
        pred_len=96,
        dropout_n=0.1,
        d_llm=768,
        e_layer=1,
        d_layer=1,
        d_ff=32,
        head=8,
        num_cma_heads=4,
        vmd_modes=4,  # VMD 模态数 K
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
        self.num_cma_heads = num_cma_heads
        self.vmd_modes = vmd_modes

        # 归一化
        self.normalize_layers = Normalize(self.num_nodes, affine=False).to(self.device)

        # ========== 时域分支（与原模型一致）==========
        self.length_to_feature = nn.Linear(self.seq_len, self.channel).to(self.device)
        self.ts_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.channel,
            nhead=self.head,
            batch_first=True,
            norm_first=True,
            dropout=self.dropout_n,
        ).to(self.device)
        self.ts_encoder = nn.TransformerEncoder(
            self.ts_encoder_layer, num_layers=self.e_layer
        ).to(self.device)

        # ========== FFT 频域分支（与原模型一致）==========
        self.Lf = seq_len // 2 + 1
        self.freq_token_proj = nn.Linear(1, self.channel).to(self.device)
        self.freq_attn_layer = nn.TransformerEncoderLayer(
            d_model=self.channel,
            nhead=self.head,
            batch_first=True,
            norm_first=True,
            dropout=self.dropout_n,
        ).to(self.device)
        self.freq_encoder = nn.TransformerEncoder(
            self.freq_attn_layer, num_layers=1
        ).to(self.device)
        self.freq_pool = FrequencyAttentionPooling(self.channel).to(self.device)

        # ========== VMD 辅助分支（新增）==========
        self.vmd_encoder = VMDEncoder(
            channel=self.channel,
            num_heads=self.head,
            vmd_modes=self.vmd_modes,
            dropout=self.dropout_n,
        ).to(self.device)

        # ========== 三路融合：时域 + FFT + VMD ==========
        # 归一化 FFT 和 VMD 分支，使它们在同一量级
        self.freq_norm = nn.LayerNorm(self.channel).to(self.device)
        self.vmd_norm = nn.LayerNorm(self.channel).to(self.device)
        
        # 融合 FFT 和 VMD 的频域特征
        self.freq_vmd_fusion = nn.Sequential(
            nn.Linear(self.channel * 2, self.channel),
            nn.ReLU(),
            nn.Dropout(self.dropout_n),
            nn.Linear(self.channel, self.channel),
        ).to(self.device)

        # 门控融合（时域 vs 融合后的频域）
        self.rich_horizon_gate = RichHorizonGate(self.channel).to(self.device)

        # ========== Prompt 编码（与原模型一致）==========
        self.prompt_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_llm,
            nhead=self.head,
            batch_first=True,
            norm_first=True,
            dropout=self.dropout_n,
        ).to(self.device)
        self.prompt_encoder = nn.TransformerEncoder(
            self.prompt_encoder_layer, num_layers=self.e_layer
        ).to(self.device)

        # ========== 多头 CMA（与原模型一致）==========
        self.cma_heads = nn.ModuleList(
            [
                CrossModal(
                    d_model=self.num_nodes,
                    n_heads=1,
                    d_ff=self.d_ff,
                    norm="LayerNorm",
                    attn_dropout=self.dropout_n,
                    dropout=self.dropout_n,
                    pre_norm=True,
                    activation="gelu",
                    res_attention=True,
                    n_layers=1,
                    store_attn=False,
                ).to(self.device)
                for _ in range(self.num_cma_heads)
            ]
        )
        self.adaptive_dynamic_heads_cma = AdaptiveDynamicHeadsCMA(
            num_heads=self.num_cma_heads,
            num_nodes=self.num_nodes,
            channel=self.channel,
            device=self.device,
        )

        # ========== 残差融合（与原模型一致）==========
        self.residual_alpha = nn.Parameter(
            torch.ones(self.channel, device=self.device) * 0.5
        ).to(self.device)

        # ========== 解码器（与原模型一致）==========
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.channel,
            nhead=self.head,
            batch_first=True,
            norm_first=True,
            dropout=self.dropout_n,
        ).to(self.device)
        self.decoder = nn.TransformerDecoder(
            self.decoder_layer, num_layers=self.d_layer
        ).to(self.device)

        # ========== 投影到预测长度（与原模型一致）==========
        self.c_to_length = nn.Linear(self.channel, self.pred_len, bias=True).to(
            self.device
        )

    def frequency_domain_processing(self, input_data):
        """
        FFT 频域处理（与原模型一致）
        """
        freq_complex = torch.fft.rfft(input_data, dim=-1)  # [B, N, Lf]
        freq_mag = torch.abs(freq_complex)
        B, N, Lf = freq_mag.shape

        freq_tokens = freq_mag.unsqueeze(-1)  # [B, N, Lf, 1]
        freq_tokens = freq_tokens.reshape(B * N, Lf, 1)  # [B*N, Lf, 1]
        freq_tokens = self.freq_token_proj(freq_tokens)  # [B*N, Lf, C]

        freq_enc_out = self.freq_encoder(freq_tokens)  # [B*N, Lf, C]
        freq_enc_out = self.freq_pool(freq_enc_out)  # [B*N, C]

        freq_enc_out = freq_enc_out.reshape(B, N, self.channel)  # [B, N, C]
        return freq_enc_out

    def forward(self, input_data, input_data_mark, embeddings, x_modes):
        """
        Args:
            input_data: [B, L, N] 时域输入
            input_data_mark: [B, L, ...] 时间标记
            embeddings: [B, E, N] 预训练 embeddings
            x_modes: [B, L, N, K] 预先计算的 VMD 模态分量
        """
        input_data = input_data.float()
        input_data_mark = input_data_mark.float()
        embeddings = embeddings.float()
        x_modes = x_modes.float()

        # 处理 embeddings 形状
        embeddings = embeddings.squeeze(-1)  # [B, E, N] (兼容部分加载方式)
        embeddings = embeddings.permute(0, 2, 1)  # [B, N, E]

        # RevIN
        input_data = self.normalize_layers(input_data, "norm")  # [B, L, N]
        input_data = input_data.permute(0, 2, 1)  # [B, N, L]

        # ========== 时域编码 ==========
        time_feat = self.length_to_feature(input_data)  # [B, N, C]
        time_enc = self.ts_encoder(time_feat)  # [B, N, C]
        time_enc = time_enc.permute(0, 2, 1)  # [B, C, N]

        # ========== VMD 频域分支（仅保留 VMD，去掉 FFT 分支）==========
        x_modes = x_modes.permute(0, 2, 1, 3)  # [B, N, L, K]
        vmd_out = self.vmd_encoder(x_modes)  # [B, N, C]
        vmd_enc = vmd_out.permute(0, 2, 1)  # [B, C, N]

        # ========== 归一化 VMD 分支，使其数值稳定 ==========
        # 对每个节点（N 维度）分别做 LayerNorm
        vmd_enc = vmd_enc.permute(0, 2, 1)  # [B, N, C] for LayerNorm
        vmd_enc = self.vmd_norm(vmd_enc)
        vmd_enc = vmd_enc.permute(0, 2, 1)  # [B, C, N]

        # ========== 频域特征（仅使用 VMD 编码结果）==========
        freq_vmd_fused = vmd_enc  # [B, C, N]

        # 保存中间结果用于调试（仅在训练时）
        if self.training:
            # 这里的 _debug_freq_enc 仅作为占位，表示“原 FFT 分支输出”现在被移除
            # 为了兼容已有调试脚本，仍保留相同形状的张量（全零）
            self._debug_freq_enc = torch.zeros_like(vmd_enc).detach()
            self._debug_vmd_enc = vmd_enc.detach()
            self._debug_freq_vmd_fused = freq_vmd_fused.detach()

        # ========== 门控融合：时域 vs 融合后的频域 ==========
        gate = self.rich_horizon_gate(time_enc, self.pred_len)  # [B, C, 1]
        enc_out = gate * freq_vmd_fused + (1 - gate) * time_enc  # [B, C, N]

        # ========== Prompt 编码 ==========
        embeddings = self.prompt_encoder(embeddings)  # [B, N, E]
        embeddings = embeddings.permute(0, 2, 1)  # [B, E, N]

        # ========== 多模态交叉对齐 (CMA) ==========
        cma_outputs = []
        for cma_head in self.cma_heads:
            cma_out = cma_head(enc_out, embeddings, embeddings)  # [B, C, N]
            cma_outputs.append(cma_out)

        fused = self.adaptive_dynamic_heads_cma(cma_outputs)  # [B, C, N]

        # ========== 残差融合 ==========
        B, C, N = fused.shape
        alpha = self.residual_alpha.view(1, C, 1)
        cross_out = alpha * fused + (1 - alpha) * enc_out  # [B, C, N]
        cross_out = cross_out.permute(0, 2, 1)  # [B, N, C]

        # ========== 解码 ==========
        dec_out = self.decoder(cross_out, cross_out)  # [B, N, C]

        # ========== 投影到输出 ==========
        dec_out = self.c_to_length(dec_out)  # [B, N, pred_len]
        dec_out = dec_out.permute(0, 2, 1)  # [B, pred_len, N]

        # ========== Denorm ==========
        dec_out = self.normalize_layers(dec_out, "denorm")

        return dec_out

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def param_num(self) -> int:
        return sum(p.numel() for p in self.parameters())


__all__ = ["TriModalFFTVMD"]

