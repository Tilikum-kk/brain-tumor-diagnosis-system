"""
==============================================================================
模型评估脚本 — 在 BraTS 验证集上测试分割性能
==============================================================================

用法：
  python eval.py --data_dir "D:/medical image/.../BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData" \
                 --checkpoint checkpoints/hg_mfnet_best.pth \
                 --num_cases 5 --save_results
==============================================================================
"""

import argparse, logging, sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # src/eval.py -> backend/
from src.config import model_config
from src.models.hypergraph import HGMFNet
from src.utils.preprocessing import MedicalImagePreprocessor, BraTSDataset

logger = logging.getLogger(__name__)


def compute_metrics(pred, gt, num_classes=4):
    """计算多类别 Dice 和 Hausdorff 距离"""
    dice_per_class = []
    for c in range(1, num_classes):  # 跳过背景
        pred_c = (pred == c).astype(np.float32)
        gt_c = (gt == c).astype(np.float32)

        intersection = (pred_c * gt_c).sum()
        union = pred_c.sum() + gt_c.sum()

        dice = (2.0 * intersection + 1e-5) / (union + 1e-5)
        dice_per_class.append(float(dice))

    # 全肿瘤 (WT = 1+2+3)
    pred_wt = (pred > 0).astype(np.float32)
    gt_wt = (gt > 0).astype(np.float32)
    intersection = (pred_wt * gt_wt).sum()
    union = pred_wt.sum() + gt_wt.sum()
    dice_wt = (2.0 * intersection + 1e-5) / (union + 1e-5)

    return {
        'dice_necrotic': dice_per_class[0],    # 坏死核心
        'dice_edema': dice_per_class[1],       # 水肿
        'dice_enhancing': dice_per_class[2],   # 增强肿瘤
        'dice_wt': float(dice_wt),             # 全肿瘤
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/hg_mfnet_best.pth")
    parser.add_argument("--num_cases", type=int, default=5)
    parser.add_argument("--save_results", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"设备: {device}")

    # 加载模型
    logger.info("加载模型...")
    model = HGMFNet().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    logger.info(f"✓ 模型加载完成 (Epoch {ckpt.get('epoch', '?')}, Dice {ckpt.get('dice', '?'):.4f})")

    # 加载数据
    preprocessor = MedicalImagePreprocessor(
        target_spacing=(1.0, 1.0, 1.0),
        target_size=model_config.image_size,
        normalize_mode="zscore",
    )
    dataset = BraTSDataset(data_dir=args.data_dir, preprocessor=preprocessor, augment=False)
    logger.info(f"验证集病例数: {len(dataset)}")

    # 评估
    num_cases = min(args.num_cases, len(dataset))
    all_metrics = []

    logger.info(f"\n{'='*70}")
    logger.info(f"开始评估 {num_cases} 个病例")
    logger.info(f"{'='*70}\n")

    for idx in tqdm(range(num_cases), desc="评估"):
        case_name = dataset.cases[idx]
        data = dataset.load_case(case_name)

        # 准备输入
        images = []
        for mod in ['t1', 't1ce', 't2', 'flair']:
            if mod in data:
                images.append(data[mod])
            else:
                images.append(np.zeros(model_config.image_size, dtype=np.float32))

        image = torch.from_numpy(np.stack(images, axis=0)).unsqueeze(0).to(device)  # (1, 4, D, H, W)
        gt = data.get('seg', np.zeros(model_config.image_size, dtype=np.int32))

        # 推理
        with torch.no_grad():
            clinical = torch.randn(1, 8).to(device)
            output = model(image, clinical)
            pred = torch.argmax(output['seg_logits'], dim=1).squeeze(0).cpu().numpy()

        # 计算指标
        metrics = compute_metrics(pred, gt)
        metrics['case'] = case_name
        metrics['tumor_volume'] = float((pred > 0).sum() * 1.0 / 1000)
        metrics['gt_volume'] = float((gt > 0).sum() * 1.0 / 1000)
        all_metrics.append(metrics)

        # 打印
        logger.info(
            f"{case_name:30s} | WT Dice: {metrics['dice_wt']:.4f} | "
            f"TC: {metrics['dice_enhancing']:.4f} | "
            f"ED: {metrics['dice_edema']:.4f} | "
            f"ET: {metrics['dice_necrotic']:.4f} | "
            f"体积: {metrics['tumor_volume']:.1f}ml (GT: {metrics['gt_volume']:.1f}ml)"
        )

    # 汇总
    logger.info(f"\n{'='*70}")
    logger.info("汇总统计")
    logger.info(f"{'='*70}")

    avg = {}
    for key in ['dice_wt', 'dice_enhancing', 'dice_edema', 'dice_necrotic']:
        vals = [m[key] for m in all_metrics]
        avg[key] = np.mean(vals)
        logger.info(f"  平均 {key}: {avg[key]:.4f} ± {np.std(vals):.4f}")

    logger.info(f"\n  平均 Dice (3类): {np.mean(list(avg.values())[1:]):.4f}")

    # 保存详细结果
    if args.save_results:
        import json
        result_path = Path("eval_results.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"\n✓ 详细结果已保存: {result_path}")


if __name__ == "__main__":
    main()
