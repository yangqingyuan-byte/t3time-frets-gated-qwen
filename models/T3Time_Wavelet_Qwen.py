"""
T3Time with Wavelet Transform and Qwen3-0.6B
基于小波变换和Qwen3-0.6B的时频域融合模型
使用小波变换替代FFT，提供时频局部化能力
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np
from layers.StandardNorm import Normalize
from layers.Cross_Modal_Align import CrossModal


class WaveletTransform(nn.Module):
    """
    小波变换模块
    使用离散小波变换(DWT)进行时频分析
    优化版本：支持批处理和GPU加速
    """
    def __init__(self, wavelet='db4', mode='symmetric', max_level=None):
        super().__init__()
        self.wavelet = wavelet
        self.mode = mode
        self.max_level = max_level
        
        # 获取小波滤波器系数
        try:
            wavelet_obj = pywt.Wavelet(wavelet)
            self.dec_lo = torch.tensor(wavelet_obj.dec_lo, dtype=torch.float32)
            self.dec_hi = torch.tensor(wavelet_obj.dec_hi, dtype=torch.float32)
        except:
            # 默认使用db4
            self.dec_lo = torch.tensor([-0.0106, 0.0329, 0.0308, -0.1870, -0.0280, 0.6309, 0.7148, 0.2304], dtype=torch.float32)
            self.dec_hi = torch.tensor([-0.2304, 0.7148, -0.6309, -0.0280, 0.1870, 0.0308, -0.0329, -0.0106], dtype=torch.float32)
    
    def dwt_1d(self, x):
        """
        一维离散小波变换（PyTorch实现）
        Args:
            x: [B, N, L] 输入信号
        Returns:
            cA: [B, N, L//2] 近似系数
            cD: [B, N, L//2] 细节系数
        """
        B, N, L = x.shape
        device = x.device
        
        # 将滤波器移到设备上
        dec_lo = self.dec_lo.to(device)
        dec_hi = self.dec_hi.to(device)
        
        # 卷积操作
        # 需要padding以保持长度
        pad_len = len(dec_lo) - 1
        x_padded = F.pad(x, (pad_len, pad_len), mode='circular')
        
        # 下采样卷积
        cA = F.conv1d(x_padded.view(B*N, 1, -1), dec_lo.view(1, 1, -1), stride=2)
        cD = F.conv1d(x_padded.view(B*N, 1, -1), dec_hi.view(1, 1, -1), stride=2)
        
        # Reshape回原始维度
        cA = cA.view(B, N, -1)
        cD = cD.view(B, N, -1)
        
        return cA, cD
    
    def wavedec(self, x, level=None):
        """
        多级小波分解
        Args:
            x: [B, N, L] 输入信号
            level: 分解层数
        Returns:
            coeffs: 系数列表，[cA_n, cD_n, cD_{n-1}, ..., cD_1]
        """
        if level is None:
            level = self.max_level if self.max_level else pywt.dwt_max_level(x.shape[-1], self.wavelet)
        
        coeffs = []
        current = x
        
        for _ in range(level):
            cA, cD = self.dwt_1d(current)
            coeffs.insert(0, cD)  # 细节系数
            current = cA  # 继续分解近似系数
        
        coeffs.insert(0, cA)  # 最底层的近似系数
        
        return coeffs
        
    def forward(self, x):
        """
        Args:
            x: [B, N, L] 输入时间序列
        Returns:
            coeffs: 小波系数列表，每个元素为 [B, N, L_i]
        """
        # 计算最大分解层数
        max_level = self.max_level if self.max_level else pywt.dwt_max_level(x.shape[-1], self.wavelet)
        
        # 进行多级小波分解
        coeffs = self.wavedec(x, level=max_level)
        
        return coeffs


class WaveletAttentionPooling(nn.Module):
    """
    小波系数的注意力加权池化
    对不同分解层的小波系数进行可学习的注意力加权
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )
    
    def forward(self, wavelet_features):
        """
        Args:
            wavelet_features: List of [B*N, L_i, C] 不同层的小波特征
        Returns:
            pooled: [B*N, C] 池化后的特征
        """
        # 对每一层计算注意力权重
        attn_weights_list = []
        for feat in wavelet_features:
            attn_logits = self.attention(feat)  # [B*N, L_i, 1]
            attn_weights = F.softmax(attn_logits, dim=1)  # [B*N, L_i, 1]
            attn_weights_list.append(attn_weights)
        
        # 加权求和
        pooled_list = []
        for feat, attn in zip(wavelet_features, attn_weights_list):
            pooled = (feat * attn).sum(dim=1)  # [B*N, C]
            pooled_list.append(pooled)
        
        # 对不同层进行加权融合
        if len(pooled_list) > 1:
            # 使用简单的平均或可学习的权重
            pooled = torch.stack(pooled_list, dim=1)  # [B*N, num_levels, C]
            # 对层维度进行平均池化
            pooled = pooled.mean(dim=1)  # [B*N, C]
        else:
            pooled = pooled_list[0]
        
        return pooled


class CrossAttentionFusion(nn.Module):
    """
    交叉注意力融合模块
    使用时域特征作为Query，频域特征作为Key/Value
    实现细粒度的时频对齐
    """
    def __init__(self, d_model, nhead=8, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, time_features, freq_features):
        """
        Args:
            time_features: [B, N, C] 时域特征
            freq_features: [B, N, C] 频域特征
        Returns:
            fused: [B, N, C] 融合后的特征
        """
        # 时域作为Query，频域作为Key/Value
        fused, _ = self.cross_attn(
            query=time_features,
            key=freq_features,
            value=freq_features
        )
        fused = self.norm(fused + time_features)  # 残差连接
        return fused


class TriModalWaveletQwen(nn.Module):
    """
    基于小波变换的三模态模型 (Qwen3-0.6B 版本)
    使用小波变换替代FFT，提供时频局部化能力
    """
    def __init__(
        self,
        device="cuda",
        channel=32,
        num_nodes=7,
        seq_len=96,
        pred_len=96,
        dropout_n=0.1,
        d_llm=1024, # Qwen3-0.6B default
        e_layer=1,
        d_layer=1,
        d_ff=32,
        head=8,
        wavelet='db4',
        use_cross_attention=True
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
        self.use_cross_attention = use_cross_attention
        
        # 归一化层
        self.normalize_layers = Normalize(self.num_nodes, affine=False).to(self.device)
        
        # 时域特征提取
        self.length_to_feature = nn.Linear(self.seq_len, self.channel).to(self.device)
        
        # 时域编码器
        self.ts_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.channel,
            nhead=self.head,
            batch_first=True,
            norm_first=True,
            dropout=self.dropout_n
        ).to(self.device)
        self.ts_encoder = nn.TransformerEncoder(
            self.ts_encoder_layer,
            num_layers=self.e_layer
        ).to(self.device)
        
        # 小波变换模块
        self.wavelet_transform = WaveletTransform(wavelet=wavelet).to(self.device)
        
        # 小波系数投影层（将不同层的小波系数投影到统一维度）
        # 由于小波分解后各层长度不同，需要分别处理
        # 预计算最大分解层数
        max_level = pywt.dwt_max_level(seq_len, wavelet) if seq_len > 0 else 3
        self.wavelet_proj_layers = nn.ModuleList([
            nn.Linear(1, self.channel).to(self.device)
            for _ in range(max_level + 1)  # +1 for approximation coefficient
        ])
        
        # 小波特征编码器
        self.wavelet_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.channel,
            nhead=self.head,
            batch_first=True,
            norm_first=True,
            dropout=self.dropout_n
        ).to(self.device)
        self.wavelet_encoder = nn.TransformerEncoder(
            self.wavelet_encoder_layer,
            num_layers=1
        ).to(self.device)
        
        # 小波注意力池化
        self.wavelet_pool = WaveletAttentionPooling(self.channel).to(self.device)
        
        # 融合模块
        if self.use_cross_attention:
            self.fusion = CrossAttentionFusion(
                d_model=self.channel,
                nhead=self.head,
                dropout=self.dropout_n
            ).to(self.device)
        else:
            # 使用门控融合作为备选
            self.fusion_gate = nn.Sequential(
                nn.Linear(self.channel * 2, self.channel),
                nn.ReLU(),
                nn.Linear(self.channel, self.channel),
                nn.Sigmoid()
            ).to(self.device)
        
        # Prompt编码器
        self.prompt_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_llm,
            nhead=self.head,
            batch_first=True,
            norm_first=True,
            dropout=self.dropout_n
        ).to(self.device)
        self.prompt_encoder = nn.TransformerEncoder(
            self.prompt_encoder_layer,
            num_layers=self.e_layer
        ).to(self.device)
        
        # 多头交叉模态对齐
        self.num_cma_heads = 4
        self.cma_heads = nn.ModuleList([
            CrossModal(
                d_model=self.num_nodes,
                n_heads=1,
                d_ff=self.d_ff,
                norm='LayerNorm',
                attn_dropout=self.dropout_n,
                dropout=self.dropout_n,
                pre_norm=True,
                activation="gelu",
                res_attention=True,
                n_layers=1,
                store_attn=False
            ).to(self.device)
            for _ in range(self.num_cma_heads)
        ])
        
        # 自适应多头聚合
        from models.T3Time import AdaptiveDynamicHeadsCMA
        self.adaptive_dynamic_heads_cma = AdaptiveDynamicHeadsCMA(
            num_heads=self.num_cma_heads,
            num_nodes=self.num_nodes,
            channel=self.channel,
            device=self.device
        )
        
        # 残差连接权重
        self.residual_alpha = nn.Parameter(torch.ones(self.channel) * 0.5).to(self.device)
        
        # 解码器
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.channel,
            nhead=self.head,
            batch_first=True,
            norm_first=True,
            dropout=self.dropout_n
        ).to(self.device)
        self.decoder = nn.TransformerDecoder(
            self.decoder_layer,
            num_layers=self.d_layer
        ).to(self.device)
        
        # 输出投影
        self.c_to_length = nn.Linear(self.channel, self.pred_len, bias=True).to(self.device)
    
    def wavelet_domain_processing(self, input_data):
        """
        小波域处理
        Args:
            input_data: [B, N, L] 输入时间序列
        Returns:
            wavelet_features: [B, N, C] 小波域特征
        """
        B, N, L = input_data.shape
        
        # 进行小波变换
        wavelet_coeffs = self.wavelet_transform(input_data)
        
        # 处理每一层的小波系数
        wavelet_features_list = []
        for level_idx, coeffs in enumerate(wavelet_coeffs):
            # coeffs: [B, N, L_level]
            B_level, N_level, L_level = coeffs.shape
            
            # 确保投影层存在
            if level_idx >= len(self.wavelet_proj_layers):
                # 动态扩展投影层列表
                for _ in range(level_idx - len(self.wavelet_proj_layers) + 1):
                    self.wavelet_proj_layers.append(nn.Linear(1, self.channel).to(self.device))
            
            # 将小波系数reshape为tokens
            coeffs_reshaped = coeffs.unsqueeze(-1)  # [B, N, L_level, 1]
            coeffs_reshaped = coeffs_reshaped.reshape(B * N, L_level, 1)  # [B*N, L_level, 1]
            
            # 投影到特征空间
            proj_layer = self.wavelet_proj_layers[level_idx]
            wavelet_tokens = proj_layer(coeffs_reshaped)  # [B*N, L_level, C]
            
            # Transformer编码
            wavelet_encoded = self.wavelet_encoder(wavelet_tokens)  # [B*N, L_level, C]
            
            wavelet_features_list.append(wavelet_encoded)
        
        # 注意力池化
        pooled = self.wavelet_pool(wavelet_features_list)  # [B*N, C]
        
        # Reshape回原始维度
        wavelet_features = pooled.reshape(B, N, self.channel)  # [B, N, C]
        
        return wavelet_features
    
    def forward(self, input_data, input_data_mark, embeddings):
        """
        Args:
            input_data: [B, L, N] 输入时间序列
            input_data_mark: [B, L, ...] 时间标记
            embeddings: [B, E, N] 预训练embeddings
        Returns:
            output: [B, L, N] 预测结果
        """
        input_data = input_data.float()
        input_data_mark = input_data_mark.float()
        embeddings = embeddings.float()
        embeddings = embeddings.squeeze(-1)  # [B, E, N]
        embeddings = embeddings.permute(0, 2, 1)  # [B, N, E]
        
        # RevIN归一化
        input_data = self.normalize_layers(input_data, 'norm')
        input_data = input_data.permute(0, 2, 1)  # [B, N, L]
        
        # 小波域处理
        wavelet_features = self.wavelet_domain_processing(input_data)  # [B, N, C]
        
        # 时域特征提取
        time_features = self.length_to_feature(input_data)  # [B, N, C]
        
        # 时域编码
        time_encoded = self.ts_encoder(time_features)  # [B, N, C]
        
        # 时频融合
        if self.use_cross_attention:
            # 使用交叉注意力融合
            fused_features = self.fusion(time_encoded, wavelet_features)  # [B, N, C]
        else:
            # 使用门控融合
            concat_features = torch.cat([time_encoded, wavelet_features], dim=-1)  # [B, N, 2*C]
            gate = self.fusion_gate(concat_features)  # [B, N, C]
            fused_features = gate * wavelet_features + (1 - gate) * time_encoded  # [B, N, C]
        
        fused_features = fused_features.permute(0, 2, 1)  # [B, C, N]
        
        # Prompt编码
        embeddings = self.prompt_encoder(embeddings)  # [B, N, E]
        embeddings = embeddings.permute(0, 2, 1)  # [B, E, N]
        
        # 多模态交叉对齐
        cma_outputs = []
        for cma_head in self.cma_heads:
            head_out = cma_head(fused_features, embeddings, embeddings)  # [B, C, N]
            cma_outputs.append(head_out)
        
        fused = self.adaptive_dynamic_heads_cma(cma_outputs)  # [B, C, N]
        
        # 残差融合
        B, C, N = fused.shape
        alpha = self.residual_alpha.view(1, C, 1)
        cross_out = alpha * fused + (1 - alpha) * fused_features  # [B, C, N]
        cross_out = cross_out.permute(0, 2, 1)  # [B, N, C]
        
        # 解码器
        dec_out = self.decoder(cross_out, cross_out)  # [B, N, C]
        
        # 输出投影
        dec_out = self.c_to_length(dec_out)  # [B, N, L]
        dec_out = dec_out.permute(0, 2, 1)  # [B, L, N]
        
        # 反归一化
        dec_out = self.normalize_layers(dec_out, 'denorm')
        
        return dec_out
    
    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])
    
    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

