"""
==============================================================================
脑肿瘤MRI智能辅助诊断系统 - 3D U-Net基线模型
Brain Tumor MRI Intelligent Diagnosis System - 3D U-Net Baseline

功能描述：
    实现3D U-Net作为脑肿瘤分割的基线模型。
    支持单模态和多模态（通道拼接）输入。
    用于与超图融合模型（HG-MFNet）进行性能对比。

架构说明：
    - 编码器：4层3D卷积下采样，通道数 [32, 64, 128, 256]
    - 瓶颈层：3D卷积，通道数 512
    - 解码器：4层3D转置卷积上采样 + 跳跃连接
    - 输出层：1×1×1 卷积 → 4类分割

参考：
    [1] Çiçek et al., "3D U-Net: Learning Dense Volumetric Segmentation
        from Sparse Annotation", MICCAI, 2016.
==============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class DoubleConv3D(nn.Module):
    """
    双卷积块（3D版本）

    结构：Conv3D → BN → ReLU → Conv3D → BN → ReLU
    这是U-Net的基本构建块。
    """

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        """
        初始化双卷积块

        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            mid_channels: 中间通道数（默认为out_channels）
        """
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.double_conv(x)


class Down3D(nn.Module):
    """
    下采样模块

    结构：MaxPool3d(2) → DoubleConv3D
    用于编码器的空间分辨率降低。
    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        初始化下采样模块

        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
        """
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.maxpool_conv(x)


class Up3D(nn.Module):
    """
    上采样模块

    结构：ConvTranspose3d (或 Upsample + Conv) → DoubleConv3D
    与跳跃连接特征拼接后处理。
    """

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        """
        初始化上采样模块

        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            bilinear: 是否使用双线性插值上采样
        """
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv3D(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        上采样 + 跳跃连接

        Args:
            x1: 来自深层的特征图
            x2: 来自编码器的跳跃连接特征图

        Returns:
            上采样后的特征图
        """
        x1 = self.up(x1)

        # 处理尺寸不匹配
        diff_d = x2.size(2) - x1.size(2)
        diff_h = x2.size(3) - x1.size(3)
        diff_w = x2.size(4) - x1.size(4)

        x1 = F.pad(x1, [
            diff_w // 2, diff_w - diff_w // 2,
            diff_h // 2, diff_h - diff_h // 2,
            diff_d // 2, diff_d - diff_d // 2,
        ])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet3D(nn.Module):
    """
    3D U-Net模型

    用于脑肿瘤MRI图像分割的基线模型。

    输入：(B, C, D, H, W) 其中C为模态数（1或4）
    输出：(B, num_classes, D, H, W) 分割预测
    """

    def __init__(self, in_channels: int = 4, num_classes: int = 4,
                 base_channels: int = 32, bilinear: bool = True):
        """
        初始化3D U-Net

        Args:
            in_channels: 输入通道数（=MRI模态数）
            num_classes: 分割类别数
            base_channels: 基础通道数
            bilinear: 是否使用双线性上采样
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        # 编码器
        self.inc = DoubleConv3D(in_channels, base_channels)
        self.down1 = Down3D(base_channels, base_channels * 2)
        self.down2 = Down3D(base_channels * 2, base_channels * 4)
        self.down3 = Down3D(base_channels * 4, base_channels * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down3D(base_channels * 8, base_channels * 16 // factor)

        # 解码器
        self.up1 = Up3D(base_channels * 16, base_channels * 8 // factor, bilinear)
        self.up2 = Up3D(base_channels * 8, base_channels * 4 // factor, bilinear)
        self.up3 = Up3D(base_channels * 4, base_channels * 2 // factor, bilinear)
        self.up4 = Up3D(base_channels * 2, base_channels, bilinear)

        # 输出层
        self.outc = nn.Conv3d(base_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        3D U-Net前向传播

        Args:
            x: 输入图像 (B, C, D, H, W)

        Returns:
            分割logits (B, num_classes, D, H, W)
        """
        # 编码路径
        x1 = self.inc(x)       # (B, 32, D, H, W)
        x2 = self.down1(x1)    # (B, 64, D/2, H/2, W/2)
        x3 = self.down2(x2)    # (B, 128, D/4, H/4, W/4)
        x4 = self.down3(x3)    # (B, 256, D/8, H/8, W/8)
        x5 = self.down4(x4)    # (B, 512, D/16, H/16, W/16)

        # 解码路径 + 跳跃连接
        x = self.up1(x5, x4)   # (B, 256, D/8, H/8, W/8)
        x = self.up2(x, x3)    # (B, 128, D/4, H/4, W/4)
        x = self.up3(x, x2)    # (B, 64, D/2, H/2, W/2)
        x = self.up4(x, x1)    # (B, 32, D, H, W)

        logits = self.outc(x)  # (B, num_classes, D, H, W)
        return logits


class AttentionUNet3D(UNet3D):
    """
    注意力3D U-Net

    在跳跃连接中加入注意力门控机制，增强对病灶区域的关注。
    注意力门控帮助模型聚焦于肿瘤区域，抑制背景噪声。

    改进：
        - 在跳跃连接前添加注意力门控
        - 利用深层特征（全局上下文）指导浅层特征（局部细节）
    """

    def __init__(self, in_channels: int = 4, num_classes: int = 4):
        """初始化注意力U-Net"""
        super().__init__(in_channels, num_classes)

        # 注意力门控模块
        base_ch = 32
        self.att_gate4 = AttentionGate3D(base_ch * 8, base_ch * 16, base_ch * 8)
        self.att_gate3 = AttentionGate3D(base_ch * 4, base_ch * 8, base_ch * 4)
        self.att_gate2 = AttentionGate3D(base_ch * 2, base_ch * 4, base_ch * 2)
        self.att_gate1 = AttentionGate3D(base_ch, base_ch * 2, base_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """带注意力的前向传播"""
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # 应用注意力门控
        a4 = self.att_gate4(x4, x5)
        x = self.up1(x5, a4)
        a3 = self.att_gate3(x3, x)
        x = self.up2(x, a3)
        a2 = self.att_gate2(x2, x)
        x = self.up3(x, a2)
        a1 = self.att_gate1(x1, x)
        x = self.up4(x, a1)

        return self.outc(x)


class AttentionGate3D(nn.Module):
    """
    三维注意力门控

    实现对跳跃连接特征的注意力加权。
    使用加法注意力（Additive Attention）机制。
    """

    def __init__(self, F_g: int, F_l: int, F_int: int):
        """
        初始化注意力门控

        Args:
            F_g: 门控信号（深层特征）通道数
            F_l: 输入特征（浅层特征）通道数
            F_int: 中间特征通道数
        """
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, bias=True),
            nn.InstanceNorm3d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, bias=True),
            nn.InstanceNorm3d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, bias=True),
            nn.InstanceNorm3d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        注意力门控前向传播

        Args:
            x: 浅层特征 (B, F_l, D, H, W)
            g: 门控信号（深层特征上采样后）(B, F_g, D, H, W)

        Returns:
            注意力加权的浅层特征
        """
        # 尺寸对齐
        if g.size()[2:] != x.size()[2:]:
            g = F.interpolate(g, size=x.size()[2:], mode='trilinear', align_corners=False)

        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


# ============================================================================
# 辅助函数 - Utility Functions
# ============================================================================
def create_unet3d(in_channels: int = 4, num_classes: int = 4,
                  use_attention: bool = False) -> nn.Module:
    """
    工厂函数：创建3D U-Net模型

    Args:
        in_channels: 输入通道数
        num_classes: 分割类别数
        use_attention: 是否使用注意力机制

    Returns:
        模型实例
    """
    if use_attention:
        return AttentionUNet3D(in_channels, num_classes)
    return UNet3D(in_channels, num_classes)


if __name__ == "__main__":
    # 测试代码
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 测试基本U-Net
    model = UNet3D(in_channels=4, num_classes=4).to(device)
    dummy_input = torch.randn(2, 4, 128, 128, 128).to(device)
    output = model(dummy_input)
    print(f"3D U-Net 输入: {dummy_input.shape}")
    print(f"3D U-Net 输出: {output.shape}")
    print(f"3D U-Net 参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 测试注意力U-Net
    att_model = AttentionUNet3D(in_channels=4, num_classes=4).to(device)
    att_output = att_model(dummy_input)
    print(f"\nAttention U-Net 输出: {att_output.shape}")
    print(f"Attention U-Net 参数量: {sum(p.numel() for p in att_model.parameters()) / 1e6:.2f}M")
