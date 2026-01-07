"""
简单的三层 MLP baseline，用于对比 VMD/FFT 等复杂模型的效果
输入: [B, L, N] (L=seq_len, N=num_nodes)
输出: [B, pred_len, N]
"""
import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    def __init__(
        self,
        seq_len: int = 96,
        pred_len: int = 96,
        num_nodes: int = 7,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim

        # 输入: [B, L, N] -> flatten -> [B, L*N]
        input_dim = seq_len * num_nodes

        # 三层 MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_len * num_nodes),
        )

    def forward(self, x, x_mark=None, embeddings=None):
        """
        Args:
            x: [B, L, N] 输入序列
            x_mark: 时间标记（本模型不使用，但保持接口兼容）
            embeddings: LLM embeddings（本模型不使用，但保持接口兼容）
        Returns:
            output: [B, pred_len, N]
        """
        B, L, N = x.shape
        # Flatten: [B, L, N] -> [B, L*N]
        x_flat = x.reshape(B, -1)
        # MLP: [B, L*N] -> [B, pred_len*N]
        output_flat = self.mlp(x_flat)
        # Reshape: [B, pred_len*N] -> [B, pred_len, N]
        output = output_flat.reshape(B, self.pred_len, N)
        return output

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def param_num(self) -> int:
        return sum(p.numel() for p in self.parameters())


__all__ = ["SimpleMLP"]

