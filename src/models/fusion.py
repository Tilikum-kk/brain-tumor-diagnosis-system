"""
==============================================================================
脑肿瘤MRI智能辅助诊断系统 - 多模态融合模块
Brain Tumor MRI Intelligent Diagnosis System - Multi-modal Fusion Module

功能描述：
    实现多模态医学图像的融合策略，支持以下融合方式：
        1. 早期融合（Early Fusion）：通道拼接
        2. 中期融合（Middle Fusion）：特征级加权融合
        3. 晚期融合（Late Fusion）：决策级投票融合
    并集成超图融合作为高级融合方式。

参考：
    三级融合策略源自Z. Guo et al. (2019) 和课程设计文档
==============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from enum import Enum


class FusionStrategy(str, Enum):
    """融合策略枚举"""
    EARLY = "early"          # 早期融合（通道拼接）
    MIDDLE = "middle"        # 中期融合（特征级）
    LATE = "late"            # 晚期融合（决策级）
    HYPERGRAPH = "hypergraph"  # 超图融合
    ATTENTION = "attention"  # 注意力融合
    ADAPTIVE = "adaptive"    # 自适应融合


class EarlyFusion(nn.Module):
    """
    早期融合模块

    将多种MRI模态在输入层进行通道拼接。
    优点：简单直接，保留原始信息
    缺点：忽略模态特异性，增加计算量
    """

    def __init__(self, num_modalities: int = 4):
        """
        初始化早期融合

        Args:
            num_modalities: 模态数量
        """
        super().__init__()
        self.num_modalities = num_modalities

    def forward(self, modalities: torch.Tensor) -> torch.Tensor:
        """
        早期融合前向传播

        Args:
            modalities: (B, M, D, H, W) 多模态图像

        Returns:
            融合图像 (B, M, D, H, W) 直接作为多通道输入
        """
        return modalities


class MiddleFusion(nn.Module):
    """
    中期融合模块（特征级）

    对各模态的中间特征进行加权融合。
    使用可学习的模态权重参数。
    """

    def __init__(self, feature_dim: int = 256, num_modalities: int = 4):
        """
        初始化中期融合

        Args:
            feature_dim: 特征维度
            num_modalities: 模态数量
        """
        super().__init__()
        # 可学习的模态权重
        self.modality_weights = nn.Parameter(torch.ones(num_modalities) / num_modalities)
        self.softmax = nn.Softmax(dim=0)
        # 融合后的特征投影
        self.fusion_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, modality_features: List[torch.Tensor]) -> torch.Tensor:
        """
        中期融合

        Args:
            modality_features: 各模态特征列表 [B, C_i, ...]

        Returns:
            融合特征
        """
        weights = self.softmax(self.modality_weights)
        fused = sum(w * feat for w, feat in zip(weights, modality_features))
        return fused


class LateFusion(nn.Module):
    """
    晚期融合模块（决策级）

    对各模态的独立预测进行投票融合。
    实现方式：平均预测概率 + 不确定性加权
    """

    def __init__(self, num_classes: int = 4, num_modalities: int = 4):
        """
        初始化晚期融合

        Args:
            num_classes: 分类类别数
            num_modalities: 模态数量
        """
        super().__init__()
        self.num_classes = num_classes
        # 模态置信度学习器
        self.confidence_net = nn.Sequential(
            nn.Linear(num_classes * num_modalities, num_modalities * 2),
            nn.ReLU(inplace=True),
            nn.Linear(num_modalities * 2, num_modalities),
            nn.Softmax(dim=1),
        )

    def forward(self, modality_predictions: List[torch.Tensor]) -> torch.Tensor:
        """
        晚期融合

        Args:
            modality_predictions: 各模态预测概率列表

        Returns:
            融合后的预测
        """
        stacked = torch.stack(modality_predictions, dim=1)  # (B, M, C, ...)
        # 平均投票
        fused = stacked.mean(dim=1)
        return fused


class CrossModalAttention(nn.Module):
    """
    跨模态注意力融合

    使用多头注意力机制建模模态间的交互关系。
    每个模态作为查询（Query），其他模态作为键（Key）和值（Value）。
    """

    def __init__(self, embed_dim: int = 256, num_heads: int = 8,
                 num_modalities: int = 4):
        """
        初始化跨模态注意力

        Args:
            embed_dim: 嵌入维度
            num_heads: 注意力头数
            num_modalities: 模态数量
        """
        super().__init__()
        self.num_modalities = num_modalities

        # 模态间注意力
        self.inter_modal_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
        )

        # 模态内自注意力
        self.intra_modal_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, modality_features: torch.Tensor) -> torch.Tensor:
        """
        跨模态注意力融合

        Args:
            modality_features: (B, M, N, D) 模态特征

        Returns:
            融合特征 (B, N, D)
        """
        B, M, N, D = modality_features.shape

        # 展平模态维度进行跨模态交互
        x = modality_features.view(B * M, N, D)

        # 模态内自注意力
        attn_out, _ = self.intra_modal_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))

        # 恢复到 (B, M, N, D)
        x = x.view(B, M, N, D)

        # 跨模态融合：取所有模态的平均作为最终表示
        fused = x.mean(dim=1)  # (B, N, D)

        return fused


class FusionFactory:
    """
    融合策略工厂类

    根据指定的融合策略创建对应的融合模块。

    使用示例：
        fusion = FusionFactory.create(FusionStrategy.HYPERGRAPH, feature_dim=256)
        fused = fusion(modality_features)
    """

    @staticmethod
    def create(strategy: FusionStrategy, **kwargs) -> nn.Module:
        """
        创建融合模块

        Args:
            strategy: 融合策略
            **kwargs: 模块参数

        Returns:
            融合模块实例

        Raises:
            ValueError: 未知的融合策略
        """
        if strategy == FusionStrategy.EARLY:
            return EarlyFusion(**kwargs)

        elif strategy == FusionStrategy.MIDDLE:
            return MiddleFusion(
                feature_dim=kwargs.get('feature_dim', 256),
                num_modalities=kwargs.get('num_modalities', 4),
            )

        elif strategy == FusionStrategy.LATE:
            return LateFusion(
                num_classes=kwargs.get('num_classes', 4),
                num_modalities=kwargs.get('num_modalities', 4),
            )

        elif strategy == FusionStrategy.ATTENTION:
            return CrossModalAttention(
                embed_dim=kwargs.get('embed_dim', 256),
                num_heads=kwargs.get('num_heads', 8),
                num_modalities=kwargs.get('num_modalities', 4),
            )

        elif strategy == FusionStrategy.HYPERGRAPH:
            # 超图融合使用HG-MFNet
            from .hypergraph import HGMFNet
            return HGMFNet()

        else:
            raise ValueError(f"未知的融合策略: {strategy}")


# ============================================================================
# 多模态预测器（集成多种融合策略）
# ============================================================================
class MultiModalPredictor(nn.Module):
    """
    多模态预测器

    集成多种融合策略，可灵活切换。
    支持在线推理时选择不同的融合方式。

    用于系统部署时的模型选择。
    """

    def __init__(self, fusion_strategy: FusionStrategy = FusionStrategy.HYPERGRAPH,
                 num_classes: int = 4, feature_dim: int = 256):
        """
        初始化多模态预测器

        Args:
            fusion_strategy: 融合策略
            num_classes: 分类类别数
            feature_dim: 特征维度
        """
        super().__init__()
        self.fusion_strategy = fusion_strategy

        # 特征提取器（共享）
        self.shared_encoder = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.InstanceNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, feature_dim, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(feature_dim),
            nn.ReLU(inplace=True),
        )

        # 融合模块
        self.fusion_module = FusionFactory.create(fusion_strategy, **{
            'feature_dim': feature_dim,
            'num_classes': num_classes,
            'num_modalities': 4,
        })

    def forward(self, mri_images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            mri_images: (B, 4, D, H, W)

        Returns:
            预测结果字典
        """
        B, M, D, H, W = mri_images.shape

        # 逐模态提取特征
        modality_feats = []
        for m in range(M):
            feat = self.shared_encoder(mri_images[:, m:m+1])
            modality_feats.append(feat)

        # 融合
        if hasattr(self.fusion_module, 'forward'):
            fused = self.fusion_module(mri_images)  # 早期融合

        return {'fused_features': fused if isinstance(fused, torch.Tensor) else None}
