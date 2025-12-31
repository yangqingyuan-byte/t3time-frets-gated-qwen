import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
from .T3Time_Wavelet import CrossModal, GatedTransformerEncoderLayer

class AdaptiveThreshold(nn.Module):
    """根据输入特征动态生成去噪阈值"""
    def __init__(self, channel):
        super().__init__()
        self.threshold_net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channel, channel // 4),
            nn.GELU(),
            nn.Linear(channel // 4, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        # x: [B, C, L]
        t = self.threshold_net(x.permute(0, 2, 1)).unsqueeze(-1) # [B, 1, 1]
        # 软阈值去噪: sign(x) * max(0, |x| - threshold)
        return torch.sign(x) * torch.relu(torch.abs(x) - t)

class MultiScalePromptInteraction(nn.Module):
    """LLM Prompt 与小波各层级进行深度交互"""
    def __init__(self, channel, d_llm, n_levels):
        super().__init__()
        self.alignments = nn.ModuleList([
            nn.Linear(d_llm, channel) for _ in range(n_levels)
        ])
        self.gates = nn.ModuleList([
            nn.Sequential(nn.Linear(channel * 2, 1), nn.Sigmoid()) for _ in range(n_levels)
        ])

    def forward(self, levels_list, prompt_emb):
        # levels_list: List of [B, L_i, C]
        # prompt_emb: [B, 1, d_llm]
        fused_levels = []
        for i, feat in enumerate(levels_list):
            p = self.alignments[i](prompt_emb) # [B, 1, C]
            # 扩展 p 到 feat 的长度
            p_ext = p.expand(-1, feat.size(1), -1)
            # 门控交互
            g = self.gates[i](torch.cat([feat, p_ext], dim=-1))
            fused_levels.append(feat * g)
        return fused_levels

class TriModalWaveletGatedShapeProQwen(nn.Module):
    def __init__(self, device, channel=64, d_ff=None, num_nodes=7, seq_len=96, pred_len=96, 
                 dropout_n=0.2, d_llm=1024, e_layer=1, wavelet='db4'):
        super().__init__()
        self.device = device
        self.channel = channel
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_ff = d_ff or 4 * channel
        self.wavelet = wavelet
        
        # 1. 时间域路径
        self.ts_embed = nn.Linear(num_nodes, channel)
        self.ts_encoder = nn.ModuleList([
            GatedTransformerEncoderLayer(channel, nhead=8, dim_feedforward=self.d_ff, dropout=dropout_n)
            for _ in range(e_layer)
        ])

        # 2. 小波域路径
        self.wavelet_embed = nn.Linear(num_nodes, channel)
        self.denoiser = AdaptiveThreshold(channel)
        
        # 计算小波层级
        dummy_input = torch.randn(1, seq_len)
        coeffs = pywt.wavedec(dummy_input.numpy(), self.wavelet)
        self.num_levels = len(coeffs)
        
        self.ms_interaction = MultiScalePromptInteraction(channel, d_llm, self.num_levels)
        
        # 小波层级编码器
        self.wavelet_encoder = GatedTransformerEncoderLayer(channel, nhead=8, dim_feedforward=self.d_ff, dropout=dropout_n)

        # 3. Prompt 路径
        self.prompt_encoder = nn.ModuleList([
            GatedTransformerEncoderLayer(d_llm, nhead=8, dim_feedforward=d_llm*2, dropout=dropout_n)
            for _ in range(e_layer)
        ])

        # 4. 融合与投影
        self.cma = CrossModal(d_model=channel, n_heads=1, d_k=channel, d_v=channel, d_ff=self.d_ff)
        self.out_proj = nn.Linear(channel, num_nodes)
        self.proj_to_pred = nn.Linear(seq_len, pred_len)

    def forward(self, x, x_mark, embeddings):
        B, L, N = x.shape
        # 数据标准化
        means = x.mean(1, keepdim=True)
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x /= stdev

        # --- Prompt 处理 ---
        prompt_feat = embeddings # [B, 1, d_llm]
        for layer in self.prompt_encoder:
            prompt_feat = layer(prompt_feat)

        # --- 时间域编码 ---
        ts_feat = self.ts_embed(x)
        for layer in self.ts_encoder:
            ts_feat = layer(ts_feat)

        # --- 小波域编码 (Pro 版) ---
        x_p = x.permute(0, 2, 1).reshape(B*N, L).cpu().numpy()
        coeffs = pywt.wavedec(x_p, self.wavelet)
        
        level_feats = []
        for c in coeffs:
            c_t = torch.tensor(c, dtype=torch.float32).to(self.device)
            # 这里的 c_t 形状是 [B*N, L_i]
            # 还原形状为 [B, N, L_i] -> [B, L_i, N]
            c_t = c_t.view(B, N, -1).permute(0, 2, 1)
            c_feat = self.wavelet_embed(c_t)
            # 自适应去噪
            c_feat = self.denoiser(c_feat.permute(0, 2, 1)).permute(0, 2, 1)
            level_feats.append(c_feat)

        # 重点：Prompt 与各尺度交互
        fused_levels = self.ms_interaction(level_feats, prompt_feat)
        
        # 将各尺度插值对齐到原长度并聚合
        aggregated_wavelet = 0
        for feat in fused_levels:
            feat_interp = F.interpolate(feat.permute(0, 2, 1), size=L, mode='linear', align_corners=False).permute(0, 2, 1)
            aggregated_wavelet += feat_interp
        
        wavelet_encoded = self.wavelet_encoder(aggregated_wavelet)

        # --- 跨模态融合 ---
        fused = self.cma(ts_feat, wavelet_encoded, wavelet_encoded)
        
        # --- 投影输出 ---
        out = self.out_proj(fused).permute(0, 2, 1)
        out = self.proj_to_pred(out).permute(0, 2, 1)
        
        # 反标准化
        out = out * stdev + means
        return out

