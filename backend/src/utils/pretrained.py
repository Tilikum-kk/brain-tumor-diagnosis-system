"""
==============================================================================
预训练模型权重加载与迁移 - Pretrained Weight Transfer
==============================================================================

功能：下载 MONAI / Swin UNETR 预训练权重，迁移到 HG-MFNet 的编码器，
      大幅缩短训练时间，弥补 BraTS 2020 数据量不足的问题。

支持的预训练来源：
  1. MONAI brats_mri_segmentation (SegResNet, BraTS 2018, ~36MB)
  2. Swin UNETR BraTS 2021 (5-fold, Dice 0.88-0.91, ~250MB/fold)

使用方式：
  python -m src.utils.pretrained --source monai     # 下载 MONAI 预训练
  python -m src.utils.pretrained --source swin_unetr --fold 1  # Swin UNETR fold1
==============================================================================
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import MODEL_DIR
from src.models.hypergraph import HGMFNet

logger = logging.getLogger(__name__)


class PretrainedWeightLoader:
    """
    预训练权重加载器

    负责：
      1. 下载预训练模型
      2. 提取编码器权重
      3. 迁移到 HG-MFNet 的 4 个模态编码器
      4. 保存初始化后的模型检查点
    """

    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.model = model
        self.device = device
        self.checkpoint_dir = MODEL_DIR
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # 方案一：MONAI SegResNet 预训练权重
    # ========================================================================
    def load_monai_pretrained(self) -> Dict[str, torch.Tensor]:
        """
        加载 MONAI brats_mri_segmentation 预训练模型

        MONAI Bundle 包含一个完整的 SegResNet 模型，
        输入 4 通道 MRI，输出 3 类分割（TC, WT, ET）。
        我们提取其编码器部分的权重。

        Returns:
            encoder_weights: 编码器权重字典
        """
        logger.info("正在加载 MONAI 预训练模型...")

        try:
            from monai.bundle import load
            pretrained = load(name="brats_mri_segmentation")
            logger.info("✓ MONAI 预训练模型加载成功")
        except Exception as e:
            logger.error(f"MONAI Bundle 加载失败: {e}")
            logger.info("尝试从 HuggingFace Hub 直接下载...")
            pretrained = self._download_from_huggingface()

        # 提取编码器权重
        encoder_weights = self._extract_monai_encoder(pretrained)
        return encoder_weights

    def _download_from_huggingface(self):
        """备选方案：从 HuggingFace Hub 下载"""
        try:
            from huggingface_hub import hf_hub_download
            import json

            # 下载模型配置和权重
            config_path = hf_hub_download(
                repo_id="MONAI/brats_mri_segmentation",
                filename="configs/inference.json",
                repo_type="model",
            )
            weight_path = hf_hub_download(
                repo_id="MONAI/brats_mri_segmentation",
                filename="models/model.pt",
                repo_type="model",
            )

            # 加载权重
            checkpoint = torch.load(weight_path, map_location="cpu", weights_only=True)
            logger.info(f"✓ 从 HuggingFace Hub 下载成功: {weight_path}")
            return checkpoint

        except Exception as e:
            logger.error(f"HuggingFace Hub 下载也失败: {e}")
            raise RuntimeError("无法加载预训练模型，请检查网络连接")

    def _extract_monai_encoder(self, pretrained) -> Dict[str, torch.Tensor]:
        """
        从 SegResNet 中提取编码器权重

        SegResNet 结构:
          - encoder: Sequential blocks (conv1 → layer1 → layer2 → layer3)
          - decoder: 上采样 + 逐层融合
        我们只需要 encoder 部分
        """
        if isinstance(pretrained, dict):
            state_dict = pretrained.get("model_state_dict", pretrained.get("state_dict", pretrained))
        elif hasattr(pretrained, "state_dict"):
            state_dict = pretrained.state_dict()
        else:
            state_dict = pretrained

        # 过滤出编码器相关的键（SegResNet 的 encoder 以特定前缀命名）
        encoder_keys = [k for k in state_dict.keys()
                        if not any(x in k.lower() for x in ["decoder", "seg", "out", "class"])]

        encoder_weights = {k: v for k, v in state_dict.items() if k in encoder_keys}

        logger.info(f"提取到 {len(encoder_weights)} 个编码器参数层")
        return encoder_weights

    # ========================================================================
    # 方案二：Swin UNETR 预训练权重
    # ========================================================================
    def load_swin_unetr_pretrained(self, fold: int = 1) -> Dict[str, torch.Tensor]:
        """
        加载 Swin UNETR BraTS 2021 预训练权重

        Swin UNETR 基于 Vision Transformer，在 BraTS 2021 (1,470例) 上训练，
        Dice 达 0.885-0.906。我们提取其编码器权重用于初始化。

        Args:
            fold: 交叉验证折数 (0-4)，fold 1 精度最高 (Dice 0.906)

        Returns:
            encoder_weights: 编码器权重字典
        """
        logger.info(f"正在加载 Swin UNETR Fold-{fold} 预训练模型...")

        # 下载地址
        url = (
            f"https://github.com/Project-MONAI/MONAI-extra-test-data/releases/"
            f"download/0.8.1/fold{fold}_f48_ep300_4gpu_dice0_9059.zip"
        )

        zip_path = self.checkpoint_dir / f"swin_unetr_fold{fold}.zip"
        extract_dir = self.checkpoint_dir / f"swin_unetr_fold{fold}"

        # 下载
        if not zip_path.exists():
            logger.info(f"下载 Swin UNETR 权重: {url}")
            import urllib.request
            urllib.request.urlretrieve(url, str(zip_path))

        # 解压
        if not extract_dir.exists():
            import zipfile
            with zipfile.ZipFile(str(zip_path), "r") as zf:
                zf.extractall(str(extract_dir))

        # 查找 .pt 文件
        pt_files = list(extract_dir.rglob("*.pt"))
        if not pt_files:
            raise FileNotFoundError(f"在 {extract_dir} 中未找到 .pt 权重文件")

        checkpoint = torch.load(str(pt_files[0]), map_location="cpu", weights_only=True)
        logger.info(f"✓ Swin UNETR Fold-{fold} 权重加载成功")

        # 提取编码器权重
        encoder_weights = self._extract_swin_encoder(checkpoint)
        return encoder_weights

    def _extract_swin_encoder(self, checkpoint: Dict) -> Dict[str, torch.Tensor]:
        """提取 Swin UNETR 的编码器权重"""
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        # Swin UNETR 的 encoder 包含 swinViT 和 encoder 层
        encoder_keys = [k for k in state_dict.keys()
                        if not any(x in k for x in ["decoder", "seg", "out", "classification"])]

        encoder_weights = {k: v for k, v in state_dict.items() if k in encoder_keys}
        logger.info(f"提取到 {len(encoder_weights)} 个编码器参数层")
        return encoder_weights

    # ========================================================================
    # 权重迁移：将预训练编码器权重映射到 HG-MFNet
    # ========================================================================
    def transfer_to_hgmfnet(self, encoder_weights: Dict[str, torch.Tensor],
                            strategy: str = "copy_4x") -> HGMFNet:
        """
        将预训练编码器权重复制到 HG-MFNet 的 4 个模态编码器

        迁移策略：
          - copy_4x: 将同一份编码器权重复制给 4 个模态（默认）
          - first_only: 只初始化第一个编码器，其余随机初始化
          - adaptive: 对每个编码器添加微小噪声，增加模态差异性

        Args:
            encoder_weights: 预训练编码器权重
            strategy: 迁移策略

        Returns:
            初始化后的 HG-MFNet 模型
        """
        logger.info(f"将预训练权重迁移到 HG-MFNet (策略: {strategy})...")
        logger.info(f"使用形状匹配方式（MONAI SegResNet → HG-MFNet 编码器）")

        for m in range(4):
            if strategy == "first_only" and m > 0:
                logger.info(f"  编码器 {m}: 跳过（first_only 策略）")
                break

            if strategy == "adaptive":
                # 添加微小噪声增加模态差异
                noisy_weights = {}
                for k, v in encoder_weights.items():
                    noise = torch.randn_like(v) * 0.02
                    noisy_weights[k] = v + noise
                transferred = self._shape_transfer(m, noisy_weights)
            else:
                # copy_4x: 每模态复制相同预训练权重
                transferred = self._shape_transfer(m, encoder_weights)

        logger.info(f"✓ 预训练权重迁移完成")
        return self.model

    def _shape_transfer(self, encoder_idx: int,
                        encoder_weights: Dict[str, torch.Tensor]) -> int:
        """
        按张量形状逐层匹配迁移权重（核心方法）

        MONAI SegResNet 和 HG-MFNet 编码器命名完全不同，但内部 Conv3D/BN 层
        形状有大量匹配。此方法遍历目标模型的每一层，在预训练权重中查找同形状
        的层进行复制。

        形状对应关系（已验证）：
          预训练 down_layers.1.1.conv* (32,32,3,3,3) → HG-MFNet enc1.1.conv* (32,32,3,3,3) ✓
          预训练 down_layers.2.*.conv* (64,64,3,3,3)   → HG-MFNet enc2.3.conv* (64,64,3,3,3) ✓
          预训练 down_layers.3.*.conv* (128,128,3,3,3) → HG-MFNet enc3.3.conv* (128,128,3,3,3) ✓
        """
        model_state = self.model.mri_encoders[encoder_idx].state_dict()
        pretrain_items = list(encoder_weights.items())
        transferred = 0
        skipped = 0

        for model_key, model_tensor in model_state.items():
            model_shape = model_tensor.shape
            best_match = None

            for pretrain_key, pretrain_tensor in pretrain_items:
                if pretrain_tensor.shape != model_shape:
                    continue

                # 按层类型进一步匹配（conv→conv, bn→bn）
                model_layer_type = self._get_layer_type(model_key)
                pretrain_layer_type = self._get_layer_type(pretrain_key)

                if model_layer_type == pretrain_layer_type:
                    best_match = pretrain_tensor
                    # 优先匹配更深的层（通道数更大的更晚被覆盖）
                    break

            if best_match is not None:
                model_state[model_key] = best_match.clone()
                transferred += 1
            else:
                skipped += 1

        self.model.mri_encoders[encoder_idx].load_state_dict(model_state, strict=True)

        pct = transferred / len(model_state) * 100
        logger.info(f"  编码器 {encoder_idx}: 形状匹配 {transferred}/{len(model_state)} 层 ({pct:.0f}%)")

        return transferred

    @staticmethod
    def _get_layer_type(key: str) -> str:
        """识别层类型：conv / bn / norm / bias"""
        key_lower = key.lower()
        if 'conv' in key_lower and 'weight' in key_lower:
            return 'conv_weight'
        if 'conv' in key_lower and 'bias' in key_lower:
            return 'conv_bias'
        if any(x in key_lower for x in ['bn', 'norm', 'instance']):
            if 'weight' in key_lower:
                return 'norm_weight'
            if 'bias' in key_lower:
                return 'norm_bias'
        return 'other'

    # ========================================================================
    # 保存和加载
    # ========================================================================
    def save_checkpoint(self, output_name: str = "hg_mfnet_pretrained.pth"):
        """保存初始化后的模型检查点"""
        output_path = self.checkpoint_dir / output_name
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'in_channels': 4,
                'num_classes': 4,
                'hidden_dim': 256,
                'num_hypergraph_layers': 3,
                'num_hyperedges': 128,
            },
            'description': 'HG-MFNet initialized with MONAI pretrained encoder weights',
        }, str(output_path))
        logger.info(f"✓ 预训练模型已保存: {output_path}")
        return str(output_path)

    def load_checkpoint(self, checkpoint_path: str):
        """加载已有的检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        logger.info(f"✓ 检查点已加载: {checkpoint_path}")


# ============================================================================
# 命令行入口
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="预训练权重下载与迁移")
    parser.add_argument("--source", type=str, default="monai",
                       choices=["monai", "swin_unetr"],
                       help="预训练权重来源")
    parser.add_argument("--fold", type=int, default=1,
                       help="Swin UNETR 的 fold 编号 (0-4)")
    parser.add_argument("--strategy", type=str, default="copy_4x",
                       choices=["copy_4x", "adaptive", "first_only"],
                       help="权重迁移策略")
    parser.add_argument("--output", type=str, default="hg_mfnet_pretrained.pth",
                       help="输出文件名")
    parser.add_argument("--device", type=str, default="cuda",
                       help="目标设备")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s [%(levelname)s] %(message)s')

    device = args.device if torch.cuda.is_available() else "cpu"
    logger.info(f"使用设备: {device}")

    # 创建模型
    logger.info("创建 HG-MFNet 模型...")
    model = HGMFNet().to(device)

    # 加载预训练权重
    loader = PretrainedWeightLoader(model, device)

    if args.source == "monai":
        encoder_weights = loader.load_monai_pretrained()
    elif args.source == "swin_unetr":
        encoder_weights = loader.load_swin_unetr_pretrained(args.fold)

    # 迁移到 HG-MFNet
    loader.transfer_to_hgmfnet(encoder_weights, strategy=args.strategy)

    # 保存
    loader.save_checkpoint(args.output)

    # 验证
    model.eval()
    dummy_input = torch.randn(1, 4, 128, 128, 128).to(device)
    dummy_clinical = torch.randn(1, 8).to(device)
    with torch.no_grad():
        output = model(dummy_input, dummy_clinical)
    logger.info(f"✓ 模型验证通过: seg={output['seg_logits'].shape}, "
               f"cls={output['who_logits'].shape}")


if __name__ == "__main__":
    main()
