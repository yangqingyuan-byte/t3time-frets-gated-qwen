import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.StandardNorm import Normalize
from layers.Cross_Modal_Align import CrossModal

# 复用已有模块，避免修改原模型文件
from models.T3Time import AdaptiveDynamicHeadsCMA, RichHorizonGate


class GatedTransformerEncoderLayer(nn.Module):
    """
    带 SDPA 输出门控的 TransformerEncoderLayer（受 Gated Attention 论文启发）
    - 在 Multi-Head Attention 输出后，增加一层 element-wise sigmoid 门控：
        attn_out = MHA(...)
        gate     = sigmoid(W_g(attn_out))
        attn_out = attn_out * gate
      再走残差 + FFN。
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

        # Self-Attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)

        # SDPA 输出上的 element-wise 门控
        self.gate_proj = nn.Linear(d_model, d_model)

        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.ffn_dropout = nn.Dropout(dropout)

        # Norm（pre-norm）
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src: [B, L, C]
        Returns:
            out: [B, L, C]
        """
        # ------ Self-Attention + Gating ------
        residual = src
        src_norm = self.norm1(src)

        attn_out, _ = self.self_attn(src_norm, src_norm, src_norm)  # [B, L, C]

        # Gated SDPA 输出：element-wise sigmoid
        gate = torch.sigmoid(self.gate_proj(attn_out))  # [B, L, C]
        attn_out = attn_out * gate

        src = residual + self.attn_dropout(attn_out)

        # ------ FFN ------
        residual = src
        src_norm = self.norm2(src)

        ffn_out = self.linear2(
            self.dropout(self.activation(self.linear1(src_norm)))
        )  # [B, L, C]
        src = residual + self.ffn_dropout(ffn_out)

        return src


class VMDEncoder(nn.Module):
    """
    VMD 模态编码器（分层设计）：
    输入 x_modes 形状 [B, N, L, K]，其中 K 是 VMD 分解得到的模态数。
    流程：
      1）模态内时间编码：对每个 IMF 单独在时间维 L 上做轻量卷积编码；
      2）模态间关系建模：在模态维 K 上做 Self-Attention/Transformer；
      3）沿模态维 K 做注意力池化，得到 [B, N, L, C]；
      4）在时间维 L 上用 Transformer + Attention Pooling 聚合为 [B, N, C]。
    """

    def __init__(
        self,
        channel: int,
        num_heads: int,
        vmd_modes: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.channel = channel
        self.vmd_modes = vmd_modes

        # 1）模态内时间编码：对每个模态的时间序列做 1D 卷积编码
        # 输入 [B*N*K, 1, L] → 输出 [B*N*K, C, L]
        self.time_conv = nn.Conv1d(
            in_channels=1,
            out_channels=channel,
            kernel_size=3,
            padding=1,
        )

        # 2）模态间关系建模：在模态维 K 上做 TransformerEncoder
        # 输入视为序列长度 K、特征维 C
        # 需要保证 nhead 可以整除 d_model=channel，因此从 min(num_heads, vmd_modes) 开始向下寻找一个可整除的头数
        mode_nhead = 1
        for h in range(min(num_heads, vmd_modes), 0, -1):
            if channel % h == 0:
                mode_nhead = h
                break

        self.mode_attn_layer = nn.TransformerEncoderLayer(
            d_model=channel,
            nhead=mode_nhead,
            batch_first=True,
            norm_first=True,
            dropout=dropout,
        )
        self.mode_encoder = nn.TransformerEncoder(
            self.mode_attn_layer,
            num_layers=1,
        )

        # 沿模态维 K 的注意力池化
        self.mode_pool = nn.Sequential(
            nn.Linear(channel, channel // 2),
            nn.ReLU(),
            nn.Linear(channel // 2, 1),
        )

        # 3）时间维上的 Transformer 编码（替换为带 SDPA 门控的版本）
        self.encoder = GatedTransformerEncoderLayer(
            d_model=channel,
            nhead=num_heads,
            dim_feedforward=4 * channel,
            dropout=dropout,
        )

        # 时间维 Attention Pooling
        self.attn_pool = nn.Sequential(
            nn.Linear(channel, channel // 2),
            nn.ReLU(),
            nn.Linear(channel // 2, 1),
        )

    def forward(self, x_modes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_modes: [B, N, L, K] 预先离线计算好的 VMD 分量
        Returns:
            vmd_out: [B, N, C]
        """
        B, N, L, K = x_modes.shape

        # ---------- 1）模态内时间编码 ----------
        # [B, N, L, K] → [B, N, K, L]
        x = x_modes.permute(0, 1, 3, 2)  # [B, N, K, L]
        x = x.reshape(B * N * K, 1, L)   # [B*N*K, 1, L]

        # 时间卷积编码
        x = self.time_conv(x)            # [B*N*K, C, L]
        x = x.permute(0, 2, 1)           # [B*N*K, L, C]

        # reshape 回 [B, N, K, L, C]
        x = x.reshape(B, N, K, L, self.channel)

        # ---------- 2）模态间关系建模 ----------
        # 视 K 为序列长度，C 为特征维；在每个时间步、每个节点上对 K 个模态做 Self-Attention
        x = x.permute(0, 1, 3, 2, 4)          # [B, N, L, K, C]
        x = x.reshape(B * N * L, K, self.channel)  # [B*N*L, K, C]

        # 在模态维 K 上的 TransformerEncoder
        x = self.mode_encoder(x)              # [B*N*L, K, C]

        # 沿 K 做注意力池化，得到每个时间步的模态聚合表示
        mode_logits = self.mode_pool(x)       # [B*N*L, K, 1]
        mode_weights = F.softmax(mode_logits, dim=1)  # [B*N*L, K, 1]
        x = (x * mode_weights).sum(dim=1)     # [B*N*L, C]

        # reshape 回 [B*N, L, C]
        x = x.reshape(B * N, L, self.channel)  # [B*N, L, C]

        # ---------- 3）时间维 Transformer + Attention Pooling ----------
        x = self.encoder(x)                    # [B*N, L, C]
        time_logits = self.attn_pool(x)        # [B*N, L, 1]
        time_weights = F.softmax(time_logits, dim=1)  # [B*N, L, 1]
        pooled = (x * time_weights).sum(dim=1)        # [B*N, C]

        # 还原 B、N 维
        vmd_out = pooled.reshape(B, N, self.channel)  # [B, N, C]
        return vmd_out


class TriModalVMD(nn.Module):
    """
    基于 VMD 的三模态模型变体。
    不覆盖原有模型文件，通过新建类实现 VMD 分支替换频域分支。
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
        vmd_modes=4,           # VMD 模态数 K
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

        # 时域投影与编码
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

        # VMD 分支编码器
        self.vmd_encoder = VMDEncoder(
            channel=self.channel,
            num_heads=self.head,
            vmd_modes=self.vmd_modes,
            dropout=self.dropout_n,
        ).to(self.device)

        # 门控融合
        self.rich_horizon_gate = RichHorizonGate(self.channel).to(self.device)

        # Prompt 编码器
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

        # 多头 CMA
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

        # VMD-时域残差融合强度参数（softplus 保证非负），初始略大一些以放大 VMD 作用
        self._vmd_residual_alpha = nn.Parameter(
            torch.ones(self.channel, device=self.device) * 0.3
        )

        # 专用于 VMD 分支的门控：让 gate 同时看到时域与 VMD 特征
        # 输入维度: concat([time_enc, vmd_c], dim=1) → [B, 2C, N]
        self.vmd_gate_mlp = nn.Sequential(
            nn.Conv1d(in_channels=2 * self.channel, out_channels=self.channel, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.channel, out_channels=self.channel, kernel_size=1),
            nn.Sigmoid(),
        )

        # 时域-模态跨注意力：让时域特征作为 Query，从 VMD 特征中按节点维 N 进行精细对齐
        self.time_vmd_attn = nn.MultiheadAttention(
            embed_dim=self.channel,
            num_heads=self.head,
            batch_first=True,
            dropout=self.dropout_n,
        )

        # 解码器
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

        # 投影到预测长度
        self.c_to_length = nn.Linear(self.channel, self.pred_len, bias=True).to(
            self.device
        )

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

        # VMD 分支（期望 x_modes 与 input_data 对齐： [B, L, N, K]）
        x_modes = x_modes.permute(0, 2, 1, 3)  # [B, N, L, K]
        vmd_out = self.vmd_encoder(x_modes)  # [B, N, C]

        # 时域编码
        time_feat = self.length_to_feature(input_data)  # [B, N, C]
        time_enc = self.ts_encoder(time_feat)  # [B, N, C]
        time_enc = time_enc.permute(0, 2, 1)  # [B, C, N]

        # 门控 + 跨注意力残差融合（VMD vs Time）：
        # 1) 利用时域+VMD 计算逐通道逐节点的 gate；
        # 2) 以时域为 Query、VMD 为 Key/Value 做跨注意力，对每个节点进行精细对齐；
        # 3) 使用 softplus(alpha) 控制 VMD 对时域的残差修正强度。
        vmd_c = vmd_out.permute(0, 2, 1)  # [B, C, N]
        B, C, N = time_enc.shape

        # 计算 gate: [B, 2C, N] -> [B, C, N]
        gate_input = torch.cat([time_enc, vmd_c], dim=1)  # [B, 2C, N]
        gate = self.vmd_gate_mlp(gate_input)             # [B, C, N] in [0,1]

        # softplus 约束 alpha >= 0
        alpha = F.softplus(self._vmd_residual_alpha).view(1, 1, C)  # [1, 1, C]

        # 节点维 N 上的跨注意力：时域为 Query，VMD 为 Key/Value
        time_seq = time_enc.permute(0, 2, 1)  # [B, N, C]
        vmd_seq = vmd_c.permute(0, 2, 1)      # [B, N, C]
        gate_seq = gate.permute(0, 2, 1)      # [B, N, C]

        attn_out, _ = self.time_vmd_attn(
            query=time_seq,
            key=vmd_seq,
            value=vmd_seq,
        )  # [B, N, C]

        # 残差修正：time_seq + alpha * gate * attn_out
        fused_seq = time_seq + alpha * gate_seq * attn_out  # [B, N, C]
        fused = fused_seq.permute(0, 2, 1)                  # [B, C, N]

        # Prompt 编码
        embeddings = self.prompt_encoder(embeddings)  # [B, N, E]
        embeddings = embeddings.permute(0, 2, 1)  # [B, E, N]

        # 多模态交叉对齐 (CMA) —— 调用方式与原 TriModal 保持一致
        cma_outputs = []
        for cma_head in self.cma_heads:
            cma_out = cma_head(fused, embeddings, embeddings)  # [B, C, N]
            cma_outputs.append(cma_out)

        # 自适应聚合多头
        fused = self.adaptive_dynamic_heads_cma(cma_outputs)  # [B, C, N]

        # 解码：与原模型解码流程保持一致，使用 [B, N, C] 作为 tgt 和 memory
        fused_perm = fused.permute(0, 2, 1)  # [B, N, C]
        decoded = self.decoder(tgt=fused_perm, memory=fused_perm)  # [B, N, C]

        # 投影到输出，得到 [B, pred_len, N]
        output = self.c_to_length(decoded)          # [B, N, pred_len]
        output = output.permute(0, 2, 1)            # [B, pred_len, N]
        return output

    # -------- 与原 TriModal 接口保持一致的参数统计方法 --------
    def count_trainable_params(self) -> int:
        """返回可训练参数的数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def param_num(self) -> int:
        """返回全部参数的数量"""
        return sum(p.numel() for p in self.parameters())


__all__ = ["TriModalVMD", "VMDEncoder"]

