"""
==============================================================================
脑肿瘤MRI智能辅助诊断系统 - 配置文件
Brain Tumor MRI Intelligent Diagnosis System - Configuration

功能描述：
    定义系统全局配置参数，包括数据库连接、模型超参数、文件路径等。
    遵循面向对象设计原则，使用Pydantic进行配置验证。

参考标准：《智能系统应用开发（II）》课程设计要求
==============================================================================
"""

import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# ============================================================================
# 项目路径配置 - Project Path Configuration
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
UPLOAD_DIR = BASE_DIR / "uploads"
REPORT_DIR = BASE_DIR / "reports"
MODEL_DIR = BASE_DIR / "checkpoints"

# 确保目录存在
for dir_path in [DATASET_DIR, UPLOAD_DIR, REPORT_DIR, MODEL_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


# ============================================================================
# 数据库配置 - Database Configuration
# 默认使用SQLite，无需安装额外数据库即可运行
# 如需切换到PostgreSQL，设置环境变量 DB_TYPE=postgresql
# ============================================================================
@dataclass
class DatabaseConfig:
    """数据库连接配置（默认SQLite，可选PostgreSQL）"""
    db_type: str = os.getenv("DB_TYPE", "sqlite")
    # PostgreSQL配置
    host: str = os.getenv("DB_HOST", "localhost")
    port: int = int(os.getenv("DB_PORT", "5432"))
    username: str = os.getenv("DB_USERNAME", "postgres")
    password: str = os.getenv("DB_PASSWORD", "postgres")
    database: str = os.getenv("DB_NAME", "brain_tumor_diagnosis")
    # SQLite配置
    sqlite_path: str = os.getenv("DB_SQLITE_PATH", str(BASE_DIR / "brain_tumor.db"))

    @property
    def url(self) -> str:
        """构建数据库连接URL"""
        if self.db_type == "postgresql":
            return (
                f"postgresql://{self.username}:{self.password}"
                f"@{self.host}:{self.port}/{self.database}"
            )
        # 默认使用SQLite
        return f"sqlite:///{self.sqlite_path}"


# ============================================================================
# 模型配置 - Model Configuration
# ============================================================================
@dataclass
class ModelConfig:
    """超图神经网络模型超参数配置"""
    # 输入配置
    in_channels: int = 4          # MRI模态数：T1, T1CE, T2, FLAIR
    image_size: tuple = (96, 96, 96)  # 输入图像尺寸（8GB显存平衡速度与精度）
    clinical_dim: int = 8         # 临床特征维度

    # 编码器配置
    encoder_channels: list = field(default_factory=lambda: [32, 64, 128, 256])
    encoder_strides: list = field(default_factory=lambda: [2, 2, 2])

    # 超图配置
    hypergraph_hidden_dim: int = 256   # 超图隐藏层维度
    num_hypergraph_layers: int = 3     # 超图卷积层数
    num_hyperedges: int = 128          # 超边数量
    spatial_radius: float = 5.0        # 空间约束半径(mm)
    semantic_threshold: float = 0.85   # 语义相似性阈值

    # 注意力配置
    num_attention_heads: int = 8       # 多头注意力头数
    attention_dropout: float = 0.1

    # 分类器配置
    num_classes: int = 4               # 分割类别：背景、坏死核心、水肿、增强肿瘤
    who_grades: int = 3                # WHO分级：I, II, III, IV

    # 训练配置
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 2
    num_epochs: int = 200
    dice_loss_weight: float = 1.0
    cls_loss_weight: float = 0.5
    reg_loss_weight: float = 0.1


# ============================================================================
# API配置 - API Configuration
# ============================================================================
@dataclass
class APIConfig:
    """FastAPI服务配置"""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    secret_key: str = os.getenv("SECRET_KEY", "brain-tumor-diagnosis-secret-key-2024")
    access_token_expire_minutes: int = 60
    max_upload_size_mb: int = 500      # 最大上传文件大小


# ============================================================================
# MRI模态映射 - MRI Modality Mapping
# ============================================================================
MRI_MODALITY_NAMES = {
    "t1": "T1-weighted MRI",
    "t1ce": "T1 Contrast-Enhanced MRI",
    "t2": "T2-weighted MRI",
    "flair": "T2 FLAIR MRI",
}

# 肿瘤标签映射
TUMOR_LABELS = {
    0: "背景 (Background)",
    1: "坏死核心 (Necrotic Core)",
    2: "瘤周水肿 (Peritumoral Edema)",
    3: "增强肿瘤 (Enhancing Tumor)",
}

# WHO分级映射
WHO_GRADE_MAP = {
    0: "WHO Grade I (良性)",
    1: "WHO Grade II (低级别)",
    2: "WHO Grade III (间变性)",
    3: "WHO Grade IV (胶质母细胞瘤)",
}


# ============================================================================
# 全局配置实例 - Global Configuration Instances
# ============================================================================
db_config = DatabaseConfig()
model_config = ModelConfig()
api_config = APIConfig()
