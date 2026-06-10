"""
==============================================================================
HG-MFNet 训练脚本 v3 — 完整优化版
==============================================================================

优化清单：
  1. ✅ 多目录合并（--data_dir + --extra_data）
  2. ✅ 独立验证集（--val_dir，优先于随机切分）
  3. ✅ Warmup + ReduceLROnPlateau + 梯度裁剪
  4. ✅ 128³ + 梯度检查点（8GB适配）
  5. ✅ 多类别 Dice 监控 + 最佳模型保存
  6. ✅ 训练日志 + TensorBoard 兼容
  7. ✅ argparse 完整命令行（所有超参数可调）

用法：
  python -m src.train --data_dir "dataset/BraTS2020" \
                      --extra_data "dataset/BraTS2021" \
                      --val_dir "dataset/BraTS_validationData" \
                      --pretrained checkpoints/hg_mfnet_pretrained.pth \
                      --epochs 100 --batch_size 1
==============================================================================
"""

import os, sys, argparse, logging, json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, ConcatDataset
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import model_config, MODEL_DIR
from src.models.hypergraph import HGMFNet, TotalLoss
from src.utils.preprocessing import MedicalImagePreprocessor, BraTSDataset
from src.utils.visualization import TrainingVisualizer, ValidationVisualizer

logger = logging.getLogger(__name__)

# 类别名称（用于日志）
LABEL_NAMES = {1: '坏死核心', 2: '水肿', 3: '增强肿瘤'}


# ============================================================================
# 训练器 v3
# ============================================================================
class Trainer:
    def __init__(self, model: nn.Module, config: Dict, device: str = "cuda"):
        self.model = model.to(device)
        self.config = config
        self.device = device

        # 损失
        self.criterion = TotalLoss(
            seg_weight=config.get('seg_weight', 1.0),
            cls_weight=config.get('cls_weight', 0.5),
            reg_weight=config.get('reg_weight', 0.1),
        )

        # 优化器 — AdamW
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.get('lr', 2e-4),
            weight_decay=config.get('weight_decay', 1e-5),
        )

        # 学习率调度器
        lr_mode = config.get('lr_scheduler', 'plateau')
        if lr_mode == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=config.get('cosine_T0', 20), T_mult=2,
                eta_min=config.get('min_lr', 1e-6),
            )
        else:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', factor=0.5,
                patience=config.get('lr_patience', 8), min_lr=1e-6, verbose=True,
            )

        self.lr_mode = lr_mode
        self.warmup_epochs = config.get('warmup_epochs', 5)
        self.base_lr = config.get('lr', 2e-4)
        self.grad_clip = config.get('grad_clip', 1.0)

        # 混合精度
        self.use_amp = config.get('use_amp', False)
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp and device == 'cuda' else None

        # 日志目录
        self.log_dir = Path(config.get('log_dir', 'logs'))
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 训练状态
        self.best_dice = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        self.history = {'train_loss': [], 'val_loss': [], 'val_dice_wt': [],
                        'val_dice_et': [], 'val_dice_tc': [], 'val_dice_ed': [], 'lr': []}
        self.val_samples_saved = False  # 是否已保存验证对比图

    # ================================================================
    def _apply_warmup(self, epoch: int):
        """Warmup：前 warmup_epochs 轮线性增长学习率"""
        if epoch <= self.warmup_epochs:
            lr = self.base_lr * epoch / self.warmup_epochs
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr

    # ================================================================
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}", leave=False)
        for batch in pbar:
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            clinical = batch.get('clinical', None)
            if clinical is not None:
                clinical = clinical.to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(images, clinical)
                    total, loss_dict = self.criterion(outputs, labels)
                self.scaler.scale(total).backward()
                self.scaler.unscale_(self.optimizer)  # 梯度裁剪前unscale
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images, clinical)
                total, loss_dict = self.criterion(outputs, labels)
                total.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            total_loss += total.item()
            pbar.set_postfix({
                'loss': f"{total.item():.4f}",
                'dice': f"{1 - loss_dict['loss_seg']:.3f}"
            })

        return total_loss / len(train_loader)

    # ================================================================
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        # 每类 Dice 收集
        dice_per_class = {c: [] for c in [1, 2, 3]}
        dice_wt = []

        for batch in tqdm(val_loader, desc="验证", leave=False):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            clinical = batch.get('clinical', None)
            if clinical is not None:
                clinical = clinical.to(self.device)

            outputs = self.model(images, clinical)
            total, loss_dict = self.criterion(outputs, labels)
            total_loss += total.item()

            seg_pred = torch.argmax(outputs['seg_logits'], dim=1)

            for b in range(images.size(0)):
                # 全肿瘤 WT
                pw = (seg_pred[b] > 0).float()
                gw = (labels[b] > 0).float()
                inter = (pw * gw).sum()
                union = pw.sum() + gw.sum()
                dice_wt.append(float((2 * inter + 1e-5) / (union + 1e-5)))

                # 逐类
                for c in [1, 2, 3]:
                    pc = (seg_pred[b] == c).float()
                    gc = (labels[b] == c).float()
                    inter = (pc * gc).sum()
                    union = pc.sum() + gc.sum()
                    dice_per_class[c].append(float((2 * inter + 1e-5) / (union + 1e-5)))

        return {
            'loss': total_loss / len(val_loader),
            'dice_wt': np.mean(dice_wt) if dice_wt else 0.0,
            'dice_et': np.mean(dice_per_class[3]) if dice_per_class[3] else 0.0,
            'dice_tc': np.mean(dice_per_class[1]) if dice_per_class[1] else 0.0,
            'dice_ed': np.mean(dice_per_class[2]) if dice_per_class[2] else 0.0,
        }

    # ================================================================
    def train(self, train_loader, val_loader, start_epoch: int = 1):
        epochs = self.config.get('epochs', 100)
        patience = self.config.get('patience', 20)

        logger.info(f"\n{'='*60}")
        logger.info(f"训练配置 v3 (容错版)")
        logger.info(f"  训练集: {len(train_loader.dataset)}例 | 验证集: {len(val_loader.dataset)}例")
        logger.info(f"  设备: {self.device} | AMP: {self.use_amp}")
        logger.info(f"  LR: {self.base_lr:.1e} | Warmup: {self.warmup_epochs}轮 | "
                    f"Clipping: {self.grad_clip}")
        logger.info(f"  Scheduler: {'Cosine' if self.lr_mode == 'cosine' else 'ReduceLROnPlateau'}")
        logger.info(f"  早停 patience: {patience}")
        logger.info(f"  断点保护: 每10轮自动保存 + Ctrl+C安全退出")
        logger.info(f"{'='*60}\n")

        last_epoch = start_epoch - 1
        try:
            for epoch in range(start_epoch, epochs + 1):
                last_epoch = epoch
                self._apply_warmup(epoch)

                train_loss = self.train_epoch(train_loader, epoch)
                val = self.validate(val_loader)

                # 更新学习率
                old_lr = self.optimizer.param_groups[0]['lr']
                if self.lr_mode == 'cosine':
                    self.scheduler.step(epoch)
                else:
                    self.scheduler.step(val['dice_wt'])
                new_lr = self.optimizer.param_groups[0]['lr']

                # 记录
                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val['loss'])
                self.history['val_dice_wt'].append(val['dice_wt'])
                self.history['val_dice_et'].append(val['dice_et'])
                self.history['val_dice_tc'].append(val['dice_tc'])
                self.history['val_dice_ed'].append(val['dice_ed'])
                self.history['lr'].append(new_lr)

                # 每10轮保存验证对比图
                if epoch % 10 == 0 or epoch == 1:
                    self._save_val_sample(val_loader, epoch)

                logger.info(
                    f"E {epoch:3d}/{epochs} | Loss {train_loss:.4f}/{val['loss']:.4f} | "
                    f"WT:{val['dice_wt']:.4f} TC:{val['dice_tc']:.4f} "
                    f"ED:{val['dice_ed']:.4f} ET:{val['dice_et']:.4f} | "
                    f"LR:{new_lr:.1e}" + (" ⬇" if new_lr < old_lr else "")
                )

                mean_dice = (val['dice_wt'] + val['dice_et'] + val['dice_tc'] + val['dice_ed']) / 4

                if mean_dice > self.best_dice:
                    self.best_dice = mean_dice
                    self.best_epoch = epoch
                    self.patience_counter = 0
                    self.save_checkpoint('best_model.pth', epoch, val['dice_wt'])
                    logger.info(f"  ★ 新最佳! 平均Dice={mean_dice:.4f}")
                else:
                    self.patience_counter += 1

                # 每10轮自动保存断点（用于恢复）
                if epoch % 10 == 0:
                    self.save_checkpoint('latest.pth', epoch, val['dice_wt'])
                    self._save_progress(epoch)

                if self.patience_counter >= patience:
                    logger.info(f"\n早停 — {patience}轮无提升 (Epoch {epoch})")
                    break

        except KeyboardInterrupt:
            logger.info(f"\n⚠ 用户中断 (Ctrl+C)，保存当前状态...")
            self.save_checkpoint('interrupted.pth', last_epoch, self.best_dice)
            self._save_progress(last_epoch)
            logger.info(f"恢复命令: --resume logs/.../latest.pth")
            raise

        except Exception as e:
            logger.error(f"\n✖ 训练异常中断: {e}")
            self.save_checkpoint('crashed.pth', last_epoch, self.best_dice)
            self._save_progress(last_epoch)
            logger.info(f"已保存崩溃前状态，恢复命令: --resume logs/.../crashed.pth")
            raise

        # 训练正常结束
        logger.info(f"\n{'='*60}")
        logger.info(f"训练完成! 最佳平均Dice={self.best_dice:.4f} (Epoch {self.best_epoch})")
        logger.info(f"{'='*60}")

        curve_path = TrainingVisualizer.plot_training_curves(self.history, str(self.log_dir))
        logger.info(f"训练曲线图已保存: {curve_path}")

        self._save_progress(last_epoch)

        return self.history

    def _save_progress(self, epoch: int):
        """保存训练进度（历史数据+当前状态）"""
        hist_path = self.log_dir / 'training_history.json'
        with open(hist_path, 'w') as f:
            json.dump({
                'last_epoch': epoch,
                'best_dice': self.best_dice,
                'best_epoch': self.best_epoch,
                'history': self.history,
            }, f, indent=2, default=str)

    @torch.no_grad()
    def _save_val_sample(self, val_loader: DataLoader, epoch: int):
        """遍历验证集，选当前模型效果最好的案例生成对比图"""
        self.model.eval()
        sample_dir = self.log_dir / 'comparison'
        sample_dir.mkdir(parents=True, exist_ok=True)

        best_mean_dice = -1.0
        best_data = None  # (img_4d, lbl_3d, pred_3d, dice_dict)

        for batch in val_loader:
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)

            outputs = self.model(images, None)
            preds = torch.argmax(outputs['seg_logits'], dim=1)

            for b in range(images.size(0)):
                img_4d = images[b].cpu().numpy()
                lbl_3d = labels[b].cpu().numpy()
                pred_3d = preds[b].cpu().numpy()

                # 计算各类别 Dice
                sample_dice = {}
                dice_list = []
                for c, cname in [(1, 'TC'), (2, 'ED'), (3, 'ET')]:
                    pc = (pred_3d == c).astype(np.float32)
                    gc = (lbl_3d == c).astype(np.float32)
                    inter = (pc * gc).sum()
                    union = pc.sum() + gc.sum()
                    d = float((2 * inter + 1e-5) / (union + 1e-5))
                    sample_dice[cname] = d
                    dice_list.append(d)

                # 全肿瘤 WT
                pw = (pred_3d > 0).astype(np.float32)
                gw = (lbl_3d > 0).astype(np.float32)
                inter = (pw * gw).sum()
                union = pw.sum() + gw.sum()
                d_wt = float((2 * inter + 1e-5) / (union + 1e-5))
                sample_dice['WT'] = d_wt

                mean_dice = np.mean([d_wt] + dice_list)
                if mean_dice > best_mean_dice:
                    best_mean_dice = mean_dice
                    sample_dice['epoch'] = epoch
                    best_data = (img_4d.copy(), lbl_3d.copy(), pred_3d.copy(), sample_dice)

        if best_data is None:
            logger.warning("  验证集为空，无法生成对比图")
            return

        img_4d, lbl_3d, pred_3d, best_dice = best_data
        logger.info(f"  最优案例 Mean Dice = {best_mean_dice:.4f} (WT={best_dice['WT']:.4f}, "
                    f"TC={best_dice['TC']:.4f}, ED={best_dice['ED']:.4f}, ET={best_dice['ET']:.4f})")

        # 用 FLAIR 做底图
        flair_idx = 3 if img_4d.shape[0] > 3 else 0
        mri_slice_3d = img_4d[flair_idx]

        path = str(sample_dir / f'comparison_epoch_{epoch:03d}.png')
        ValidationVisualizer.save_comparison(
            mri_slice_3d, lbl_3d, pred_3d, path,
            modality_name='FLAIR',
            dice_metrics=best_dice,
        )
        logger.info(f"  验证对比图已保存: comparison_epoch_{epoch:03d}.png (最优案例)")

    def save_checkpoint(self, filename: str, epoch: int, dice: float):
        path = self.log_dir / filename
        ckpt = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if hasattr(self, 'scheduler') else None,
            'best_dice': self.best_dice,
            'best_epoch': self.best_epoch,
            'patience_counter': self.patience_counter,
            'dice': dice,
            'history': self.history,
        }
        torch.save(ckpt, str(path))
        logger.info(f"  📦 已保存: {path.name}")


# ============================================================================
# 数据加载
# ============================================================================
def create_dataloaders_multi(data_dirs: List[str], config: Dict,
                             val_dir: str = None, seed: int = 42):
    preprocessor = MedicalImagePreprocessor(
        target_spacing=(1.0, 1.0, 1.0),
        target_size=model_config.image_size,
        normalize_mode="zscore",
        clip_intensity=(-3.0, 3.0),
    )

    # 训练集（多目录合并）
    datasets = []
    for d in data_dirs:
        if os.path.isdir(d):
            ds = BraTSDataset(data_dir=d, preprocessor=preprocessor, augment=True)
            datasets.append(ds)
            logger.info(f"  训练: {Path(d).name} ({len(ds)}例)")
        else:
            logger.warning(f"  ⚠ 路径不存在: {d}")

    train_full = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    logger.info(f"  训练集总计: {len(train_full)}例")

    # 验证集
    if val_dir and os.path.isdir(val_dir):
        val_dataset = BraTSDataset(data_dir=val_dir, preprocessor=preprocessor, augment=False)
        train_ds = train_full
        val_ds = val_dataset
        logger.info(f"  验证集(独立): {Path(val_dir).name} ({len(val_ds)}例)")
    else:
        total = len(train_full)
        val_size = max(int(total * 0.05), 5)
        train_ds, val_ds = random_split(
            train_full, [total - val_size, val_size],
            generator=torch.Generator().manual_seed(seed),
        )
        logger.info(f"  验证集(随机{val_size/total*100:.0f}%): {len(val_ds)}例")

    batch_size = config.get('batch_size', 1)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2,
        pin_memory=True, collate_fn=collate_brats_batch,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=2,
        pin_memory=True, collate_fn=collate_brats_batch,
    )
    return train_loader, val_loader


def collate_brats_batch(batch):
    """批次整理：中心裁剪到 (128,128,128)"""
    D, H, W = model_config.image_size
    processed_images, processed_labels = [], []

    for item in batch:
        img, lbl = item['image'], item['label']
        # 中心裁剪到目标尺寸
        d0 = max(0, (img.shape[1] - D) // 2)
        h0 = max(0, (img.shape[2] - H) // 2)
        w0 = max(0, (img.shape[3] - W) // 2)
        img_c = img[:, d0:d0 + D, h0:h0 + H, w0:w0 + W]
        lbl_c = lbl[d0:d0 + D, h0:h0 + H, w0:w0 + W]
        processed_images.append(torch.from_numpy(img_c).float())
        processed_labels.append(torch.from_numpy(lbl_c).long())

    return {
        'image': torch.stack(processed_images),
        'label': torch.stack(processed_labels),
        'clinical': None,
    }


# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        prog='HG-MFNet 训练',
        description='基于超图神经网络的多模态MRI脑肿瘤分割训练',
        epilog='作者: 梁昊 (2023413304)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ---- 数据参数 ----
    data = parser.add_argument_group('数据配置')
    data.add_argument('--data_dir', type=str, required=True,
                      help='主训练数据目录')
    data.add_argument('--extra_data', type=str, nargs='*', default=[],
                      help='额外训练数据目录（可多个，合并训练）')
    data.add_argument('--val_dir', type=str, default=None,
                      help='独立验证集目录（不指定则从训练集随机切5%%）')

    # ---- 训练参数 ----
    train_g = parser.add_argument_group('训练参数')
    train_g.add_argument('--epochs', type=int, default=100,
                         help='训练总轮数 (默认: 100)')
    train_g.add_argument('--batch_size', type=int, default=1,
                         help='批次大小 (默认: 1，适配8GB显存)')
    train_g.add_argument('--lr', type=float, default=2e-4,
                         help='初始学习率 (默认: 2e-4)')
    train_g.add_argument('--weight_decay', type=float, default=1e-5,
                         help='权重衰减 (默认: 1e-5)')
    train_g.add_argument('--patience', type=int, default=10,
                         help='早停耐心值 (默认: 10)')
    train_g.add_argument('--lr_patience', type=int, default=8,
                         help='学习率下降耐心值 (默认: 8)')
    train_g.add_argument('--warmup_epochs', type=int, default=5,
                         help='Warmup轮数 (默认: 5)')
    train_g.add_argument('--grad_clip', type=float, default=1.0,
                         help='梯度裁剪阈值 (默认: 1.0)')
    train_g.add_argument('--seed', type=int, default=42,
                         help='随机种子 (默认: 42)')

    # ---- 调度器 ----
    sched = parser.add_argument_group('学习率调度器')
    sched.add_argument('--lr_scheduler', type=str, default='plateau',
                       choices=['plateau', 'cosine'],
                       help='plateau=自适应降LR, cosine=余弦退火 (默认: plateau)')
    sched.add_argument('--cosine_T0', type=int, default=20,
                       help='Cosine退火周期 (默认: 20)')

    # ---- 模型 ----
    model_g = parser.add_argument_group('模型配置')
    model_g.add_argument('--pretrained', type=str,
                         default='checkpoints/hg_mfnet_pretrained.pth',
                         help='预训练权重路径')
    model_g.add_argument('--resume', type=str, default=None,
                         help='从检查点恢复训练')
    model_g.add_argument('--amp', action='store_true',
                         help='启用混合精度训练')
    model_g.add_argument('--log_dir', type=str, default=None,
                         help='日志输出目录 (默认自动生成时间戳目录，恢复训练时指定旧目录)')

    args = parser.parse_args()

    # 日志
    if args.log_dir:
        log_dir = args.log_dir
    else:
        log_dir = f'logs/train_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'{log_dir}/train.log', encoding='utf-8'),
        ],
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"设备: {device} | 显存: {mem:.1f}GB")
    else:
        logger.info(f"设备: {device}")

    # 设置种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # 数据
    data_dirs = [args.data_dir] + args.extra_data
    logger.info(f"数据目录: {data_dirs}")

    # 模型
    model = HGMFNet()
    if os.path.exists(args.pretrained):
        logger.info(f"加载预训练权重: {args.pretrained}")
        ckpt = torch.load(args.pretrained, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=False)
    else:
        logger.warning(f"预训练权重不存在: {args.pretrained}，将从头训练")

    # 数据加载
    train_loader, val_loader = create_dataloaders_multi(
        data_dirs, vars(args), val_dir=args.val_dir, seed=args.seed
    )

    # 训练配置
    config = {
        'epochs': args.epochs, 'batch_size': args.batch_size,
        'lr': args.lr, 'weight_decay': args.weight_decay,
        'patience': args.patience, 'lr_patience': args.lr_patience,
        'warmup_epochs': args.warmup_epochs, 'grad_clip': args.grad_clip,
        'use_amp': args.amp, 'lr_scheduler': args.lr_scheduler,
        'cosine_T0': args.cosine_T0,
        'log_dir': log_dir,
    }

    # 恢复训练
    start_epoch = 1
    resume_ckpt = None
    if args.resume:
        resume_path = args.resume
        if args.resume == 'latest':
            latest = Path(log_dir) / 'latest.pth'
            if latest.exists():
                resume_path = str(latest)
                logger.info(f"自动检测最新检查点: {resume_path}")
            else:
                logger.warning("未找到 latest.pth，从头训练")
                resume_path = None
        if resume_path:
            logger.info(f"恢复训练: {resume_path}")
            resume_ckpt = torch.load(resume_path, map_location='cpu', weights_only=False)
            model.load_state_dict(resume_ckpt['model_state_dict'])
            start_epoch = resume_ckpt.get('epoch', 0) + 1
            logger.info(f"从 Epoch {start_epoch} 继续")

    trainer = Trainer(model, config, device)

    # 恢复完整训练状态（optimizer、scheduler、best_dice、history等）
    if resume_ckpt is not None:
        trainer.optimizer.load_state_dict(resume_ckpt.get('optimizer_state_dict', {}))
        if resume_ckpt.get('scheduler_state_dict') and hasattr(trainer, 'scheduler'):
            try:
                trainer.scheduler.load_state_dict(resume_ckpt['scheduler_state_dict'])
            except Exception:
                pass  # scheduler 状态不兼容时忽略
        trainer.best_dice = resume_ckpt.get('best_dice', 0.0)
        trainer.best_epoch = resume_ckpt.get('best_epoch', 0)
        trainer.patience_counter = resume_ckpt.get('patience_counter', 0)
        if resume_ckpt.get('history'):
            trainer.history = resume_ckpt['history']
        resumed_lr = trainer.optimizer.param_groups[0]['lr']
        logger.info(f"已恢复训练状态: best_dice={trainer.best_dice:.4f}, "
                    f"patience_counter={trainer.patience_counter}, LR={resumed_lr:.1e}")
    trainer.train(train_loader, val_loader, start_epoch=start_epoch)


if __name__ == "__main__":
    main()
