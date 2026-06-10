"""
==============================================================================
脑肿瘤MRI智能辅助诊断系统 - 图像预处理管线
Brain Tumor MRI Intelligent Diagnosis System - Preprocessing Pipeline

功能描述：
    实现医学图像（NIfTI/DICOM格式）的标准化预处理流程：
        1. 图像加载与格式转换
        2. 重采样至统一分辨率（1mm³各向同性）
        3. 强度归一化（Z-score标准化）
        4. 数据增强（训练时）
        5. 脑部区域提取（颅骨剥离）
        6. Patch提取与拼接

遵循MONAI预处理最佳实践，确保与训练流程一致。

参考：
    [1] BraTS预处理流程
    [2] MONAI文档: https://docs.monai.io/
==============================================================================
"""

import os
import numpy as np
from typing import Tuple, Optional, Dict, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class MedicalImagePreprocessor:
    """
    医学图像预处理器

    封装完整的MRI图像预处理管线，支持NIfTI和DICOM格式。
    所有处理步骤均可在训练/推理模式下运行。

    处理流程：
        1. 加载NIfTI图像
        2. 重采样到目标分辨率
        3. Z-score强度归一化
        4. 裁剪/填充到统一尺寸
        5. （可选）数据增强
    """

    def __init__(self, target_spacing: Tuple[float, ...] = (1.0, 1.0, 1.0),
                 target_size: Tuple[int, ...] = (128, 128, 128),
                 clip_intensity: Tuple[float, float] = (-3.0, 3.0),
                 normalize_mode: str = "zscore"):
        """
        初始化预处理器

        Args:
            target_spacing: 目标体素间距 (mm)
            target_size: 目标图像尺寸
            clip_intensity: 强度裁剪范围
            normalize_mode: 归一化模式 ("zscore" | "minmax" | "none")
        """
        self.target_spacing = target_spacing
        self.target_size = target_size
        self.clip_intensity = clip_intensity
        self.normalize_mode = normalize_mode

    def load_nifti(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载NIfTI格式图像

        支持 .nii 和 .nii.gz 格式。

        Args:
            file_path: NIfTI文件路径

        Returns:
            image_data: 图像数据数组 (D, H, W)
            affine: 仿射变换矩阵 (4, 4)

        Raises:
            ImportError: nibabel未安装
            FileNotFoundError: 文件不存在
        """
        try:
            import nibabel as nib
        except ImportError:
            raise ImportError("请安装nibabel: pip install nibabel")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        img = nib.load(file_path)
        image_data = img.get_fdata().astype(np.float32)
        affine = img.affine

        logger.info(f"加载NIfTI: {file_path}, 形状: {image_data.shape}")
        return image_data, affine

    def save_nifti(self, data: np.ndarray, affine: np.ndarray,
                   file_path: str) -> None:
        """
        保存NIfTI格式图像

        Args:
            data: 图像数据
            affine: 仿射变换矩阵
            file_path: 输出路径
        """
        try:
            import nibabel as nib
        except ImportError:
            raise ImportError("请安装nibabel: pip install nibabel")

        img = nib.Nifti1Image(data.astype(np.float32), affine)
        nib.save(img, file_path)
        logger.info(f"保存NIfTI: {file_path}")

    def resample(self, image: np.ndarray, affine: np.ndarray,
                 original_spacing: Optional[Tuple[float, ...]] = None) -> np.ndarray:
        """
        重采样图像到目标分辨率 (128, 128, 128)

        直接使用 scipy zoom 缩放到固定尺寸，确保模型输入维度一致。

        Args:
            image: 输入图像 (D, H, W)，可以是任意尺寸
            affine: 仿射变换矩阵（未使用，保留接口兼容）
            original_spacing: 原始体素间距（未使用）

        Returns:
            重采样后的图像 (128, 128, 128)
        """
        target = np.array(self.target_size, dtype=np.float64)
        source = np.array(image.shape, dtype=np.float64)
        zoom_factors = target / source

        from scipy.ndimage import zoom
        resampled = zoom(image.astype(np.float64), zoom_factors, order=1)  # 线性插值
        return resampled.astype(np.float32)

    def _simple_resample(self, image: np.ndarray) -> np.ndarray:
        """简单的缩放重采样（备用方案）"""
        from scipy.ndimage import zoom
        zoom_factors = tuple(
            t / o for t, o in zip(self.target_size, image.shape)
        )
        return zoom(image, zoom_factors, order=1).astype(np.float32)

    def normalize_intensity(self, image: np.ndarray,
                            mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        强度归一化

        支持两种归一化模式：
            - zscore: Z-score标准化 (值 - 均值) / 标准差
            - minmax: 最小-最大归一化到[0, 1]

        Args:
            image: 输入图像
            mask: 前景掩码（仅对前景计算统计量）

        Returns:
            归一化后的图像
        """
        if self.normalize_mode == "zscore":
            return self._zscore_normalize(image, mask)
        elif self.normalize_mode == "minmax":
            return self._minmax_normalize(image, mask)
        else:
            return image

    def _zscore_normalize(self, image: np.ndarray,
                          mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Z-score标准化

        Args:
            image: 输入图像
            mask: 前景掩码

        Returns:
            标准化后的图像
        """
        if mask is not None and mask.sum() > 0:
            # 仅使用前景区域计算统计量
            foreground = image[mask > 0]
            mean = foreground.mean()
            std = foreground.std()
        else:
            mean = image.mean()
            std = image.std()

        std = max(std, 1e-8)  # 防止除零
        normalized = (image - mean) / std

        # 裁剪到指定范围
        normalized = np.clip(normalized,
                            self.clip_intensity[0],
                            self.clip_intensity[1])
        return normalized

    def _minmax_normalize(self, image: np.ndarray,
                          mask: Optional[np.ndarray] = None) -> np.ndarray:
        """最小-最大归一化"""
        if mask is not None and mask.sum() > 0:
            min_val = image[mask > 0].min()
            max_val = image[mask > 0].max()
        else:
            min_val = image.min()
            max_val = image.max()

        denom = max_val - min_val
        if denom == 0:
            return np.zeros_like(image)
        return (image - min_val) / denom

    def crop_or_pad(self, image: np.ndarray) -> np.ndarray:
        """
        裁剪或填充图像到目标尺寸

        Args:
            image: 输入图像 (D, H, W)

        Returns:
            处理后的图像 (target_D, target_H, target_W)
        """
        current_shape = np.array(image.shape)
        target_shape = np.array(self.target_size)

        # 对于每个维度
        result = np.zeros(self.target_size, dtype=image.dtype)

        # 计算有效范围
        min_idx = np.minimum(current_shape, target_shape)

        # 中心裁剪
        starts = ((current_shape - min_idx) // 2).astype(int)
        ends = (starts + min_idx).astype(int)

        target_starts = ((target_shape - min_idx) // 2).astype(int)
        target_ends = (target_starts + min_idx).astype(int)

        # 切片复制
        slices_src = tuple(slice(s, e) for s, e in zip(starts, ends))
        slices_dst = tuple(slice(s, e) for s, e in zip(target_starts, target_ends))

        result[slices_dst] = image[slices_src]

        return result

    def remove_non_brain(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        颅骨剥离（脑部区域提取）

        使用简单的阈值方法提取脑部区域。
        在实际部署中可替换为BET (Brain Extraction Tool) 或深度学习模型。

        Args:
            image: 输入MRI图像

        Returns:
            brain_image: 脑部区域图像
            brain_mask: 脑部掩码
        """
        # Otsu阈值分割
        from skimage.filters import threshold_otsu

        # 逐切片处理
        brain_mask = np.zeros_like(image, dtype=np.uint8)
        for z in range(image.shape[0]):
            slice_img = image[z]
            if slice_img.max() - slice_img.min() < 1e-6:
                continue
            try:
                thresh = threshold_otsu(slice_img[slice_img > slice_img.min()])
                brain_mask[z] = (slice_img > thresh).astype(np.uint8)
            except Exception:
                brain_mask[z] = (slice_img > 0).astype(np.uint8)

        # 形态学后处理
        from scipy.ndimage import binary_closing, binary_opening
        brain_mask = binary_opening(brain_mask, iterations=2)
        brain_mask = binary_closing(brain_mask, iterations=3)

        brain_image = image * brain_mask
        return brain_image.astype(np.float32), brain_mask.astype(np.uint8)

    def process(self, file_path: str, is_label: bool = False,
                return_mask: bool = False) -> np.ndarray:
        """
        完整预处理管线

        执行从文件加载到输出标准化图像的全部步骤。

        Args:
            file_path: 输入文件路径
            is_label: 是否为标注文件（标注文件不做归一化）
            return_mask: 是否返回脑部掩码

        Returns:
            预处理后的图像（和可选的掩码）
        """
        # 1. 加载图像
        image, affine = self.load_nifti(file_path)

        # 2. 重采样
        image = self.resample(image, affine)

        # 3. 强度归一化（仅图像，非标注）
        if not is_label:
            image = self.normalize_intensity(image)

        # 4. 裁剪/填充
        image = self.crop_or_pad(image)

        return image

    def process_multi_modal(self, file_paths: Dict[str, str],
                            clinical_data: Optional[np.ndarray] = None
                            ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        多模态MRI批量预处理

        处理T1, T1CE, T2, FLAIR四种模态，统一到相同空间。

        Args:
            file_paths: 模态映射 {"t1": path, "t1ce": path, ...}
            clinical_data: 临床数据（可选）

        Returns:
            multi_modal_image: (4, D, H, W) 多模态数据
            clinical_data: 临床数据
        """
        modality_order = ["t1", "t1ce", "t2", "flair"]
        images = []

        for modality in modality_order:
            if modality in file_paths and file_paths[modality]:
                img = self.process(file_paths[modality])
                images.append(img)
            else:
                # 缺失模态用零填充
                logger.warning(f"模态 {modality} 缺失，使用零填充")
                images.append(np.zeros(self.target_size, dtype=np.float32))

        multi_modal = np.stack(images, axis=0)  # (4, D, H, W)
        return multi_modal, clinical_data


# ============================================================================
# 数据增强 - Data Augmentation
# ============================================================================
class MRIDataAugmentation:
    """
    MRI数据增强器

    提供训练时使用的数据增强操作，增加模型泛化能力。

    增强策略：
        - 随机翻转（各轴，概率0.5）
        - 随机旋转（90°倍数）
        - 随机缩放（0.9-1.1）
        - 高斯噪声（σ=0.01）
        - 随机亮度/对比度调整
        - 弹性形变（概率0.2）
    """

    def __init__(self, prob_flip: float = 0.5, prob_rotate: float = 0.5,
                 prob_noise: float = 0.3, prob_elastic: float = 0.2):
        """
        初始化数据增强器

        Args:
            prob_flip: 翻转概率
            prob_rotate: 旋转概率
            prob_noise: 噪声概率
            prob_elastic: 弹性形变概率
        """
        self.prob_flip = prob_flip
        self.prob_rotate = prob_rotate
        self.prob_noise = prob_noise
        self.prob_elastic = prob_elastic

    def random_flip(self, image: np.ndarray, mask: Optional[np.ndarray] = None
                    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """随机翻转 — 自动适配 3D/4D 数组"""
        # image 是 4D (C, D, H, W)，mask 是 3D (D, H, W)
        # 沿空间维度翻转：对于 image 是 axes 1,2,3，对于 mask 是 axes 0,1,2
        img_axes = list(range(1, image.ndim))  # image: [1, 2, 3]  mask: [0, 1, 2]
        mask_axes = list(range(mask.ndim)) if mask is not None else []

        for i, axis in enumerate(img_axes):
            if np.random.random() < self.prob_flip:
                image = np.flip(image, axis=axis)
                if mask is not None:
                    mask = np.flip(mask, axis=mask_axes[i])
        return image, mask

    def random_rotate90(self, image: np.ndarray, mask: Optional[np.ndarray] = None
                        ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """随机90度旋转（H-W平面）— 自动适配 3D/4D 数组"""
        if np.random.random() < self.prob_rotate:
            k = np.random.randint(1, 4)
            # image 4D: axes (2, 3) = H, W; mask 3D: axes (1, 2) = H, W
            img_hw = (image.ndim - 2, image.ndim - 1)
            mask_hw = (mask.ndim - 2, mask.ndim - 1) if mask is not None else None
            image = np.rot90(image, k=k, axes=img_hw)
            if mask is not None:
                mask = np.rot90(mask, k=k, axes=mask_hw)
        return image, mask

    def random_noise(self, image: np.ndarray) -> np.ndarray:
        """添加高斯噪声"""
        if np.random.random() < self.prob_noise:
            noise = np.random.normal(0, 0.01, image.shape)
            image = image + noise
        return image

    def random_intensity_shift(self, image: np.ndarray) -> np.ndarray:
        """随机强度偏移"""
        if np.random.random() < 0.3:
            shift = np.random.uniform(-0.1, 0.1)
            scale = np.random.uniform(0.9, 1.1)
            image = image * scale + shift
        return image

    def augment(self, image: np.ndarray, mask: Optional[np.ndarray] = None
                ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        执行完整的数据增强管线

        Args:
            image: 输入图像 (C, D, H, W)
            mask: 标注掩码 (D, H, W)，可选

        Returns:
            增强后的图像和掩码
        """
        # 翻转
        image, mask = self.random_flip(image, mask)
        # 旋转
        image, mask = self.random_rotate90(image, mask)
        # 噪声
        image = self.random_noise(image)
        # 强度偏移
        image = self.random_intensity_shift(image)

        return image, mask


# ============================================================================
# BraTS数据集加载器 - BraTS Dataset Loader
# ============================================================================
class BraTSDataset:
    """
    BraTS数据集加载器

    处理BraTS (Brain Tumor Segmentation) 数据集的标准目录结构。
    BraTS包含四种MRI模态和专家标注的肿瘤分割。

    目录结构：
        BraTS20_Training_XXX/
            ├── BraTS20_Training_XXX_t1.nii.gz
            ├── BraTS20_Training_XXX_t1ce.nii.gz
            ├── BraTS20_Training_XXX_t2.nii.gz
            ├── BraTS20_Training_XXX_flair.nii.gz
            └── BraTS20_Training_XXX_seg.nii.gz

    标注说明：
        0: 背景
        1: 坏死核心 (NCR)
        2: 瘤周水肿 (ED)
        4: 增强肿瘤 (ET)
        → 重映射为: 0:背景, 1:坏死核心, 2:水肿, 3:增强肿瘤
    """

    def __init__(self, data_dir: str, preprocessor: MedicalImagePreprocessor,
                 augment: bool = False):
        """
        初始化BraTS数据集

        Args:
            data_dir: 数据集根目录
            preprocessor: 预处理器
            augment: 是否启用数据增强
        """
        self.data_dir = Path(data_dir)
        self.preprocessor = preprocessor
        self.augment = augment
        self.augmenter = MRIDataAugmentation() if augment else None

        # 扫描数据目录
        self.cases = self._scan_cases()

    def _scan_cases(self) -> List[str]:
        """扫描数据集目录，获取所有病例"""
        cases = []
        for item in sorted(self.data_dir.iterdir()):
            if item.is_dir() and "BraTS" in item.name:
                cases.append(item.name)
        logger.info(f"扫描到 {len(cases)} 个病例")
        return cases

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.cases)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """
        获取单个样本

        Args:
            idx: 样本索引

        Returns:
            字典包含 'image' (4, D, H, W) 和 'label' (D, H, W)
        """
        case_name = self.cases[idx]
        data = self.load_case(case_name)

        # 组装4模态图像
        modality_order = ['t1', 't1ce', 't2', 'flair']
        images = []
        for mod in modality_order:
            if mod in data:
                images.append(data[mod])
            else:
                # 缺失模态用零填充
                shape = self.preprocessor.target_size
                images.append(np.zeros(shape, dtype=np.float32))

        image = np.stack(images, axis=0)  # (4, D, H, W)

        # 标注
        label = data.get('seg', np.zeros(self.preprocessor.target_size, dtype=np.int32))

        # 数据增强
        if self.augmenter is not None:
            image, label = self.augmenter.augment(image, label)

        return {'image': image.astype(np.float32), 'label': label.astype(np.int64)}

    def _remap_labels(self, seg: np.ndarray) -> np.ndarray:
        """
        重映射BraTS分割标签

        BraTS原始: 0=背景, 1=坏死核心, 2=水肿, 4=增强肿瘤
        重映射为: 0=背景, 1=坏死核心, 2=水肿, 3=增强肿瘤
        """
        remapped = np.zeros_like(seg)
        remapped[seg == 1] = 1  # 坏死核心
        remapped[seg == 2] = 2  # 水肿
        remapped[seg == 4] = 3  # 增强肿瘤
        return remapped

    def load_case(self, case_name: str) -> Dict[str, np.ndarray]:
        """
        加载单个病例

        Args:
            case_name: 病例目录名

        Returns:
            字典包含四种模态图像和标注
        """
        case_dir = self.data_dir / case_name
        data = {}

        modality_map = {
            't1': '_t1.nii.gz',
            't1ce': '_t1ce.nii.gz',
            't2': '_t2.nii.gz',
            'flair': '_flair.nii.gz',
        }

        for mod, suffix in modality_map.items():
            # 同时匹配 .nii 和 .nii.gz
            matches = list(case_dir.glob(f"*{suffix}"))
            if not matches:
                # .nii.gz 没找到，尝试 .nii
                alt_suffix = suffix.replace('.gz', '') if suffix.endswith('.gz') else suffix + '.gz'
                matches = list(case_dir.glob(f"*{alt_suffix}"))
            if not matches:
                # 尝试不区分扩展名的模糊匹配
                matches = list(case_dir.glob(f"*{mod}*.nii*"))
                # 排除 seg 文件
                matches = [m for m in matches if 'seg' not in m.name.lower()]
            if matches:
                img = self.preprocessor.process(str(matches[0]))
                data[mod] = img

        # 加载标注（支持 .nii 和 .nii.gz）
        seg_files = list(case_dir.glob("*_seg.nii*"))
        if seg_files:
            seg = self.preprocessor.process(str(seg_files[0]), is_label=True)
            data['seg'] = self._remap_labels(seg.astype(np.int32))

        return data
