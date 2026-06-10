"""
==============================================================================
训练可视化和验证对比图生成
==============================================================================

功能：
  1. 训练结束后生成 Loss/Dice 曲线图
  2. 验证时生成原图/标签/预测 对比图
  3. 推理时生成多模态叠加分割图
==============================================================================
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

# 配置中文字体（Windows系统首选SimHei黑体）
_chinese_fonts = [f.name for f in fm.fontManager.ttflist
                  if f.name in ['SimHei', 'Microsoft YaHei', 'STXihei', 'FangSong']]
if _chinese_fonts:
    plt.rcParams['font.sans-serif'] = [_chinese_fonts[0]] + plt.rcParams['font.sans-serif']
    plt.rcParams['axes.unicode_minus'] = False


class TrainingVisualizer:
    """训练过程可视化器"""

    @staticmethod
    def plot_training_curves(history: Dict[str, List[float]], save_dir: str):
        """
        生成训练曲线图

        Args:
            history: 训练历史 {'train_loss':[], 'val_loss':[], 'val_dice_wt':[], ...}
            save_dir: 保存目录
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        epochs = range(1, len(history.get('train_loss', [])) + 1)

        # 1. Loss 曲线
        ax = axes[0, 0]
        if history.get('train_loss'):
            ax.plot(epochs, history['train_loss'], color='#1677ff', linewidth=2, label='训练 Loss')
        if history.get('val_loss'):
            ax.plot(epochs, history['val_loss'], color='#ff4d4f', linewidth=2, label='验证 Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('训练/验证 Loss 曲线')
        ax.legend()
        ax.grid(alpha=0.3)

        # 2. WT Dice
        ax = axes[0, 1]
        if history.get('val_dice_wt'):
            ax.plot(epochs, history['val_dice_wt'], color='#52c41a', linewidth=2, marker='o', markersize=3)
            ax.axhline(y=max(history['val_dice_wt']), color='#ff4d4f', linestyle='--', alpha=0.5,
                      label=f'最佳={max(history["val_dice_wt"]):.4f}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Dice')
        ax.set_title('全肿瘤 (WT) Dice 曲线')
        ax.legend()
        ax.grid(alpha=0.3)

        # 3. 多类别 Dice
        ax = axes[1, 0]
        colors = {'val_dice_et': '#ff4d4f', 'val_dice_tc': '#1677ff', 'val_dice_ed': '#faad14'}
        labels = {'val_dice_et': '增强肿瘤(ET)', 'val_dice_tc': '坏死核心(TC)', 'val_dice_ed': '水肿(ED)'}
        for key, color in colors.items():
            if history.get(key):
                ax.plot(epochs, history[key], color=color, linewidth=1.5, alpha=0.8, label=labels[key])
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Dice')
        ax.set_title('各类别 Dice 曲线')
        ax.legend()
        ax.grid(alpha=0.3)

        # 4. LR 曲线
        ax = axes[1, 1]
        if history.get('lr'):
            ax.plot(epochs, history['lr'], color='#722ed1', linewidth=2)
            ax.set_yscale('log')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('学习率变化')
        ax.grid(alpha=0.3)

        plt.suptitle('HG-MFNet 训练曲线', fontsize=16, fontweight='bold', y=1.01)
        plt.tight_layout()

        path = save_dir / 'training_curves.png'
        fig.savefig(str(path), dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        return str(path)


class ValidationVisualizer:
    """验证对比图生成器"""

    # 类别颜色映射
    CLASS_COLORS = {
        0: [0, 0, 0, 0],        # 背景 - 透明
        1: [1, 0, 0, 0.7],      # 坏死核心 - 红色
        2: [0, 0, 1, 0.7],      # 水肿 - 蓝色
        3: [1, 1, 0, 0.7],      # 增强肿瘤 - 黄色
    }

    @staticmethod
    def save_comparison(
        mri_slice: np.ndarray,
        ground_truth: np.ndarray,
        prediction: np.ndarray,
        save_path: str,
        modality_name: str = "FLAIR",
        dice_metrics: Optional[Dict[str, float]] = None,
        slice_idx: Optional[int] = None,
    ):
        """
        保存原图/差异/专家标注/AI预测四图对比

        Args:
            mri_slice: MRI 2D切片 (H, W)，或3D (D, H, W)
            ground_truth: 标签 (H, W) 或 (D, H, W)
            prediction: 预测 (H, W) 或 (D, H, W)
            save_path: 保存路径
            modality_name: MRI模态名
            dice_metrics: 各类别Dice指标 {'WT':0.85, 'TC':0.72, 'ED':0.80, 'ET':0.65}
            slice_idx: 切片索引，3D时取中间层
        """
        # 3D → 自动选肿瘤面积最大的切片（而非固定中间层）
        if ground_truth.ndim == 3:
            tumor_per_slice = (ground_truth > 0).sum(axis=(1, 2))
            if tumor_per_slice.max() > 0:
                slice_idx = int(np.argmax(tumor_per_slice))
            else:
                slice_idx = ground_truth.shape[0] // 2
        else:
            slice_idx = None

        if mri_slice.ndim == 3:
            mri_slice = mri_slice[slice_idx]
        if ground_truth.ndim == 3:
            ground_truth = ground_truth[slice_idx]
        if prediction.ndim == 3:
            prediction = prediction[slice_idx]

        from matplotlib.patches import Patch

        fig = plt.figure(figsize=(20, 10.5))
        gs = fig.add_gridspec(2, 3, hspace=0.15, wspace=0.05, height_ratios=[1, 1])

        # ==== Row 1: Original Image | Expert Label | Prediction ====
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.imshow(mri_slice, cmap='gray')
        ax0.set_title('Original Image', fontsize=13, fontweight='bold')
        ax0.axis('off')

        ax1 = fig.add_subplot(gs[0, 1])
        ax1.imshow(mri_slice, cmap='gray')
        ax1.imshow(_build_rgba_mask(ground_truth), alpha=0.6)
        ax1.set_title('Expert Label', fontsize=13, fontweight='bold')
        ax1.legend(handles=[
            Patch(facecolor='#ff0000', alpha=0.6, label='Necrotic Core'),
            Patch(facecolor='#0000ff', alpha=0.6, label='Edema'),
            Patch(facecolor='#ffff00', alpha=0.6, label='Enhancing Tumor'),
        ], loc='lower left', fontsize=8, framealpha=0.9, edgecolor='gray')
        ax1.axis('off')

        ax2 = fig.add_subplot(gs[0, 2])
        ax2.imshow(mri_slice, cmap='gray')
        ax2.imshow(_build_rgba_mask(prediction), alpha=0.6)
        ax2.set_title('Prediction', fontsize=13, fontweight='bold')
        ax2.axis('off')

        # ==== Row 2: Comparison (col 0) | Info Panel (col 1-2) ====
        diff_map = np.zeros((*mri_slice.shape, 3))
        for c in [1, 2, 3]:
            gt_c = ground_truth == c
            pred_c = prediction == c
            diff_map[gt_c & pred_c, 1] = 1.0
            diff_map[gt_c & ~pred_c, 0] = 1.0
            diff_map[~gt_c & pred_c, 2] = 1.0

        ax3 = fig.add_subplot(gs[1, 0])
        ax3.imshow(mri_slice, cmap='gray', alpha=0.5)
        ax3.imshow(diff_map, alpha=0.7)
        ax3.set_title('Comparison', fontsize=13, fontweight='bold')
        ax3.legend(handles=[
            Patch(facecolor='#00ff00', alpha=0.7, label='Correct (TP)'),
            Patch(facecolor='#ff0000', alpha=0.7, label='Missed (FN)'),
            Patch(facecolor='#0000ff', alpha=0.7, label='False (FP)'),
        ], loc='lower left', fontsize=8, framealpha=0.9, edgecolor='gray')
        ax3.axis('off')

        # Info panel
        ax4 = fig.add_subplot(gs[1, 1:])
        ax4.axis('off')

        lines = []
        lines.append("COMPARISON COLOR LEGEND")
        lines.append("-" * 36)
        lines.append("  Green  = Correct prediction (TP)")
        lines.append("  Red    = Missed by AI (FN)")
        lines.append("  Blue   = False alarm (FP)")
        lines.append("")
        lines.append("EXPERT LABEL COLORS")
        lines.append("-" * 36)
        lines.append("  Red    = Necrotic Core (TC)")
        lines.append("  Blue   = Edema (ED)")
        lines.append("  Yellow = Enhancing Tumor (ET)")
        lines.append("")
        lines.append("DICE SCORES")
        lines.append("-" * 36)
        if dice_metrics:
            lines.append(f"  WT  (Whole Tumor)       = {dice_metrics.get('WT', 0):.4f}")
            lines.append(f"  TC  (Necrotic Core)     = {dice_metrics.get('TC', 0):.4f}")
            lines.append(f"  ED  (Edema)             = {dice_metrics.get('ED', 0):.4f}")
            lines.append(f"  ET  (Enhancing Tumor)   = {dice_metrics.get('ET', 0):.4f}")
            avg = np.mean([dice_metrics.get(k, 0) for k in ['WT', 'TC', 'ED', 'ET']])
            lines.append(f"  " + "-" * 32)
            lines.append(f"  Mean Dice               = {avg:.4f}")
        else:
            lines.append("  (calculating...)")

        info_text = '\n'.join(lines)
        ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes,
                fontsize=11, family='monospace', verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='#f8f9fa',
                         alpha=0.95, edgecolor='#cccccc', pad=2.0))

        epoch_str = f"  Epoch {dice_metrics.get('epoch', '')}" if dice_metrics and dice_metrics.get('epoch') else ''
        plt.suptitle(f'Tumor Segmentation Comparison - {modality_name}{epoch_str}',
                    fontsize=15, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        return save_path

    @staticmethod
    def save_multi_modal_overview(
        images: Dict[str, np.ndarray],   # {'t1': array, 't1ce': ..., 't2': ..., 'flair': ...}
        prediction: np.ndarray,
        save_path: str,
    ):
        """
        四模态 + 分割结果全景图

        Args:
            images: 四种MRI模态的3D数组
            prediction: 预测分割3D数组
            save_path: 保存路径
        """
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        modality_names = {'t1': 'T1 加权', 't1ce': 'T1 增强', 't2': 'T2 加权', 'flair': 'FLAIR'}

        # 取中间切片
        first_img = list(images.values())[0]
        mid_z = first_img.shape[0] // 2 if first_img.ndim == 3 else first_img.shape[2] // 2

        for i, (mod, name) in enumerate(modality_names.items()):
            if mod in images:
                img = images[mod]
                if img.ndim == 4:
                    img = img[0]
                if img.ndim == 3:
                    img = img[mid_z] if img.shape[0] == mid_z * 2 else img[:, :, mid_z]
                axes[i].imshow(img, cmap='gray')
            axes[i].set_title(name, fontsize=11)
            axes[i].axis('off')

        # 最后一个：分割结果
        pred_slice = prediction
        if pred_slice.ndim == 3:
            pred_slice = pred_slice[mid_z if pred_slice.shape[0] > pred_slice.shape[-1] else pred_slice.shape[-1] // 2]
        axes[4].imshow(pred_slice, cmap='jet', vmin=0, vmax=3)
        axes[4].set_title('AI 分割结果', fontsize=11)
        axes[4].axis('off')

        plt.suptitle('多模态 MRI + 肿瘤分割全景', fontsize=14, fontweight='bold')
        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        return save_path


def _build_rgba_mask(mask: np.ndarray) -> np.ndarray:
    """构建 RGBA 分割掩码叠加层"""
    rgba = np.zeros((*mask.shape, 4))
    class_colors = {
        1: [1, 0, 0, 0.6],    # 红色 - 坏死核心
        2: [0, 0, 1, 0.6],    # 蓝色 - 水肿
        3: [1, 1, 0, 0.6],    # 黄色 - 增强肿瘤
    }
    for c, color in class_colors.items():
        rgba[mask == c] = color
    return rgba
