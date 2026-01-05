"""
T3Time Learnable Wavelet Packet Gated Pro Qwen (V30 - Freq-Dropout SOTA)
基于 V25 (冠军版本) 的改进：
1. 频带随机丢弃 (Frequency Dropout): 在训练时随机 mask 掉某个频带，
   强迫模型不依赖单一频率成分，增强鲁棒性。
2. 结构基底: 严格保持 V25 的所有配置 (Prior Init, Pre-Norm, affine=False).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.StandardNorm import Normalize

class GatedTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.gate_proj = nn.Linear(d_model, d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        nx = self.norm1(src)
        attn_output, _ = self.self_attn(nx, nx, nx)
        gate = torch.sigmoid(self.gate_proj(nx))
        src = src + self.dropout1(attn_output * gate)
        nx = self.norm2(src)
        ff_output = self.linear2(self.dropout(F.relu(self.linear1(nx))))
        src = src + self.dropout2(ff_output)
        return src

class LearnableSoftThreshold(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.threshold = nn.Parameter(torch.full((1, channel, 1), 0.1))

    def forward(self, x):
        thr = torch.abs(self.threshold)
        return torch.sign(x) * torch.relu(torch.abs(x) - thr)

class ResidualLiftingStep(nn.Module):
    def __init__(self, channel=1):
        super().__init__()
        # V18: Kernel=3, LeakyReLU
        self.predict = nn.Sequential(
            nn.Conv1d(channel, 8, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(8, channel, kernel_size=3, padding=1)
        )
        self.update = nn.Sequential(
            nn.Conv1d(channel, 8, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(8, channel, kernel_size=3, padding=1)
        )
        self.threshold_a = LearnableSoftThreshold(channel)
        self.threshold_d = LearnableSoftThreshold(channel)

    def forward(self, x):
        x_even = x[:, :, 0::2]
        x_odd = x[:, :, 1::2]
        min_len = min(x_even.shape[-1], x_odd.shape[-1])
        x_even, x_odd = x_even[:, :, :min_len], x_odd[:, :, :min_len]
        d = x_odd - self.predict(x_even)
        d = self.threshold_d(d)
        a = x_even + self.update(d)
        a = self.threshold_a(a)
        return a, d

class TriModalLearnableWaveletPacketGatedProQwen(nn.Module):
    def __init__(self, device="cuda", channel=32, num_nodes=7, seq_len=96, pred_len=96, 
                 dropout_n=0.1, d_llm=1024, e_layer=1, d_layer=1, head=8, wp_level=2):
        super().__init__()
        self.device, self.channel, self.num_nodes, self.seq_len, self.pred_len = device, channel, num_nodes, seq_len, pred_len
        self.wp_level = wp_level
        
        # V25: affine=False
        self.normalize_layers = Normalize(num_nodes, affine=False).to(device)
        self.length_to_feature = nn.Linear(seq_len, channel).to(device)
        
        # V25: FFN=4*C (V12配置)
        self.ts_encoder = nn.ModuleList([
            GatedTransformerEncoderLayer(channel, head, dropout=dropout_n) 
            for _ in range(e_layer)
        ]).to(device)
        
        self.lifting_steps = nn.ModuleList([ResidualLiftingStep() for _ in range(wp_level)]).to(device)
        num_wp_nodes = 2 ** wp_level
        
        self.wp_proj = nn.ModuleList([nn.Linear(1, channel) for _ in range(num_wp_nodes)]).to(device)
        self.freq_pos_embed = nn.Parameter(torch.zeros(1, num_wp_nodes, channel)).to(device)
        nn.init.trunc_normal_(self.freq_pos_embed, std=0.02)

        self.wp_encoder = GatedTransformerEncoderLayer(channel, head, dropout=dropout_n).to(device)
        self.attn_pool_query = nn.Parameter(torch.randn(1, 1, channel)).to(device)
        
        self.cf_interaction = nn.MultiheadAttention(channel, head, dropout=dropout_n, batch_first=True).to(device)
        self.cf_norm = nn.LayerNorm(channel).to(device)

        # V25: Prior Init
        prior_weights = torch.zeros(num_wp_nodes, channel)
        prior_weights[0, :] = 1.0  
        prior_weights[1:, :] = -1.0 
        self.band_weights = nn.Parameter(prior_weights).to(device)

        self.fusion_gate = nn.Sequential(
            nn.Linear(channel * 2 + 1, channel // 2),
            nn.ReLU(),
            nn.Linear(channel // 2, channel),
            nn.Sigmoid()
        ).to(device)
        
        self.prompt_encoder = nn.ModuleList([
            GatedTransformerEncoderLayer(d_llm, head, dropout=dropout_n) 
            for _ in range(e_layer)
        ]).to(device)
        
        from layers.Cross_Modal_Align import CrossModal
        self.cma = CrossModal(d_model=num_nodes, n_heads=1, d_ff=32, dropout=dropout_n).to(device)
        
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=channel, nhead=head, batch_first=True, dropout=dropout_n),
            num_layers=d_layer
        ).to(device)
        self.out_proj = nn.Linear(channel, pred_len).to(device)

    def wp_processing(self, x):
        B, N, L = x.shape
        x_reshaped = x.reshape(B * N, 1, L)
        
        nodes = [x_reshaped]
        for l in range(self.wp_level):
            next_nodes = []
            for node in nodes:
                a, d = self.lifting_steps[l](node)
                next_nodes.append(a); next_nodes.append(d)
            nodes = next_nodes
        
        node_feats = []
        for i, node in enumerate(nodes):
            feat = self.wp_proj[i](node.transpose(1, 2))
            feat = feat + self.freq_pos_embed[:, i, :].unsqueeze(1)
            feat = self.wp_encoder(feat)
            scores = torch.matmul(feat, self.attn_pool_query.transpose(1, 2))
            weights = F.softmax(scores, dim=1)
            node_feats.append((feat * weights).sum(dim=1))
        
        wp_stack = torch.stack(node_feats, dim=1) # [B*N, 4, C]
        
        # 【核心改进 V30】 频带 Dropout (Frequency Dropout)
        # 仅在训练时，以 10% 的概率随机 mask 掉某个频带
        if self.training:
            # 生成 mask: [B*N, 4, 1]
            # Bernouli 分布: 0.9 概率为 1 (保留), 0.1 概率为 0 (丢弃)
            mask = torch.bernoulli(torch.full((B*N, 4, 1), 0.9, device=self.device))
            # 缩放以保持期望值不变 (Inverted Dropout)
            mask = mask / 0.9 
            wp_stack = wp_stack * mask

        # Pre-Norm (V25)
        wp_norm = self.cf_norm(wp_stack)
        wp_inter, _ = self.cf_interaction(wp_norm, wp_norm, wp_norm)
        wp_out = wp_stack + wp_inter
        
        w = torch.sigmoid(self.band_weights) 
        wp_final = (wp_out * w.unsqueeze(0)).sum(dim=1)
        
        return wp_final.reshape(B, N, self.channel)
    
    def get_sparsity_loss(self):
        loss = 0
        for m in self.modules():
            if isinstance(m, LearnableSoftThreshold):
                loss += torch.abs(m.threshold).mean()
        return loss

    def forward(self, x, x_mark, emb):
        emb = emb.float().squeeze(-1)
        x_norm = self.normalize_layers(x.float(), 'norm').permute(0, 2, 1)
        
        wp_feat = self.wp_processing(x_norm)
        ts_feat = self.length_to_feature(x_norm)
        for layer in self.ts_encoder: ts_feat = layer(ts_feat)
        
        B, N, C = ts_feat.shape
        horizon_info = torch.full((B, N, 1), self.pred_len / 100.0, device=self.device)
        gate_input = torch.cat([ts_feat, wp_feat, horizon_info], dim=-1)
        gate = self.fusion_gate(gate_input)
        fused = ts_feat + gate * wp_feat
        
        p_feat = emb.permute(0, 2, 1)
        for layer in self.prompt_encoder: p_feat = layer(p_feat)
        p_feat = p_feat.permute(0, 2, 1)
        
        fused_cma = self.cma(fused.permute(0, 2, 1), p_feat, p_feat)
        dec_in = fused_cma.permute(0, 2, 1)
        dec_out = self.decoder(dec_in, dec_in)
        out = self.out_proj(dec_out).permute(0, 2, 1)
        return self.normalize_layers(out, 'denorm')
