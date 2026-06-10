"""
==============================================================================
脑肿瘤MRI智能辅助诊断系统 - 数据库模型定义
Brain Tumor MRI Intelligent Diagnosis System - Database Models

功能描述：
    定义系统中所有数据库表的ORM模型，包括用户、患者、检查记录和报告。
    使用SQLAlchemy ORM，遵循面向对象设计原则。

表结构：
    - User: 系统用户（医生/管理员）
    - Patient: 患者基本信息
    - Examination: 影像检查记录
    - Report: 诊断报告
    - ModelVersion: 模型版本管理

参考：《智能系统应用开发（II）》课程设计要求
==============================================================================
"""

import enum
from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, Text, DateTime,
    ForeignKey, Enum, JSON, Boolean
)
from sqlalchemy.orm import relationship
from .database import Base


# ============================================================================
# 枚举类型定义 - Enum Definitions
# ============================================================================
class UserRole(str, enum.Enum):
    """用户角色枚举"""
    ADMIN = "admin"         # 系统管理员
    DOCTOR = "doctor"       # 医生
    RESEARCHER = "researcher"  # 研究人员


class ExamStatus(str, enum.Enum):
    """检查状态枚举"""
    PENDING = "pending"         # 待处理
    PROCESSING = "processing"   # 处理中
    COMPLETED = "completed"     # 已完成
    FAILED = "failed"          # 处理失败


class Gender(str, enum.Enum):
    """性别枚举"""
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"


# ============================================================================
# 用户模型 - User Model
# ============================================================================
class User(Base):
    """
    系统用户表

    存储医生和管理员的账户信息，支持基于角色的访问控制。
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True, comment="用户ID")
    username = Column(String(50), unique=True, nullable=False, index=True, comment="用户名")
    email = Column(String(100), unique=True, nullable=False, comment="电子邮箱")
    hashed_password = Column(String(255), nullable=False, comment="加密密码")
    full_name = Column(String(100), nullable=False, comment="真实姓名")
    role = Column(Enum(UserRole), default=UserRole.DOCTOR, nullable=False, comment="用户角色")
    department = Column(String(100), comment="所属科室")
    hospital = Column(String(200), comment="所属医院")
    is_active = Column(Boolean, default=True, comment="是否激活")
    created_at = Column(DateTime, default=datetime.utcnow, comment="创建时间")
    last_login = Column(DateTime, comment="最后登录时间")

    # 关联关系：一个用户可以发起多次检查
    examinations = relationship("Examination", back_populates="user", lazy="dynamic")

    def __repr__(self) -> str:
        return f"<User(id={self.id}, username='{self.username}', role='{self.role}')>"


# ============================================================================
# 患者模型 - Patient Model
# ============================================================================
class Patient(Base):
    """
    患者信息表

    存储患者的基本信息和临床数据，包括年龄、性别、WHO分级等。
    这些临床数据将作为超图多模态融合的临床模态输入。
    """
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, autoincrement=True, comment="患者ID")
    patient_code = Column(String(50), unique=True, nullable=False, index=True, comment="患者编码")
    full_name = Column(String(100), comment="患者姓名（可脱敏）")
    age = Column(Integer, comment="年龄")
    gender = Column(Enum(Gender), comment="性别")
    who_grade = Column(Integer, comment="WHO肿瘤分级(0-3)")

    # 临床信息（作为超图临床模态节点特征）
    tumor_location = Column(String(200), comment="肿瘤位置（如：额叶、颞叶等）")
    medical_history = Column(Text, comment="病史摘要")
    symptoms = Column(Text, comment="临床症状")
    karnofsky_score = Column(Integer, comment="Karnofsky功能状态评分(0-100)")

    # 其他临床特征
    clinical_features = Column(JSON, comment="扩展临床特征(JSON格式)")

    created_at = Column(DateTime, default=datetime.utcnow, comment="创建时间")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, comment="更新时间")

    # 关联关系：一个患者可以有多次检查
    examinations = relationship("Examination", back_populates="patient", lazy="dynamic")

    def __repr__(self) -> str:
        return f"<Patient(id={self.id}, code='{self.patient_code}', age={self.age})>"


# ============================================================================
# 检查记录模型 - Examination Model
# ============================================================================
class Examination(Base):
    """
    影像检查记录表

    存储每次MRI检查的影像数据路径、处理状态和结果信息。
    支持四种MRI模态：T1, T1CE, T2, FLAIR。
    """
    __tablename__ = "examinations"

    id = Column(Integer, primary_key=True, autoincrement=True, comment="检查ID")
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False, comment="患者ID")
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, comment="操作医生ID")

    # 检查基本信息
    exam_date = Column(DateTime, default=datetime.utcnow, comment="检查日期")
    status = Column(Enum(ExamStatus), default=ExamStatus.PENDING, comment="处理状态")

    # MRI模态文件路径（NIfTI格式）
    t1_image_path = Column(String(500), comment="T1-weighted MRI文件路径")
    t1ce_image_path = Column(String(500), comment="T1CE MRI文件路径")
    t2_image_path = Column(String(500), comment="T2-weighted MRI文件路径")
    flair_image_path = Column(String(500), comment="FLAIR MRI文件路径")

    # 分割结果路径
    segmentation_mask_path = Column(String(500), comment="肿瘤分割掩码路径")
    segmentation_overlay_path = Column(String(500), comment="分割覆盖图路径")
    heatmap_path = Column(String(500), comment="超图注意力热力图路径")

    # 分析结果
    tumor_volume_ml = Column(Float, comment="肿瘤体积(ml)")
    edema_volume_ml = Column(Float, comment="水肿体积(ml)")
    enhancing_volume_ml = Column(Float, comment="增强区体积(ml)")
    dice_score = Column(Float, comment="分割Dice系数")
    malignant_probability = Column(Float, comment="恶性风险概率(0-1)")
    predicted_who_grade = Column(Integer, comment="预测WHO分级")

    # 超图分析结果
    hypergraph_attention_weights = Column(JSON, comment="超图注意力权重")
    modality_contributions = Column(JSON, comment="各模态贡献度")

    # 可视化和对比
    comparison_image_path = Column(String(500), comment="原图/预测对比图路径")

    # 元信息
    processing_time_seconds = Column(Float, comment="处理耗时(秒)")
    error_message = Column(Text, comment="错误信息")
    notes = Column(Text, comment="医生备注")

    created_at = Column(DateTime, default=datetime.utcnow, comment="创建时间")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, comment="更新时间")

    # 关联关系
    patient = relationship("Patient", back_populates="examinations")
    user = relationship("User", back_populates="examinations")
    report = relationship("Report", back_populates="examination", uselist=False, cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Examination(id={self.id}, status='{self.status}')>"


# ============================================================================
# 诊断报告模型 - Report Model
# ============================================================================
class Report(Base):
    """
    诊断报告表

    存储AI辅助诊断生成的完整报告，包括分割结果、分类预测、
    可视化和临床建议。
    """
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, autoincrement=True, comment="报告ID")
    examination_id = Column(Integer, ForeignKey("examinations.id"), unique=True, nullable=False, comment="检查ID")

    # 报告内容
    report_summary = Column(Text, comment="诊断摘要")
    segmentation_findings = Column(Text, comment="分割发现")
    classification_result = Column(Text, comment="分类结果")
    clinical_recommendations = Column(Text, comment="临床建议")

    # 量化指标
    tumor_dimensions = Column(JSON, comment="肿瘤三维尺寸")
    volumetric_analysis = Column(JSON, comment="体积分析数据")

    # 报告文件
    pdf_report_path = Column(String(500), comment="PDF报告文件路径")
    report_images = Column(JSON, comment="报告附图路径列表")

    # 审核信息
    is_verified = Column(Boolean, default=False, comment="是否经医生审核")
    verified_by = Column(Integer, ForeignKey("users.id"), comment="审核医生ID")
    verified_at = Column(DateTime, comment="审核时间")

    created_at = Column(DateTime, default=datetime.utcnow, comment="创建时间")

    # 关联关系
    examination = relationship("Examination", back_populates="report")

    def __repr__(self) -> str:
        return f"<Report(id={self.id}, exam_id={self.examination_id})>"


# ============================================================================
# 模型版本管理 - Model Version Model
# ============================================================================
class ModelVersion(Base):
    """
    AI模型版本管理表

    记录训练和部署的模型版本信息，便于模型迭代和回溯。
    """
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, autoincrement=True, comment="模型ID")
    model_name = Column(String(100), nullable=False, comment="模型名称")
    version = Column(String(50), nullable=False, comment="版本号")
    model_type = Column(String(50), comment="模型类型（如：HG-MFNet, 3D U-Net）")
    checkpoint_path = Column(String(500), comment="模型权重文件路径")
    metrics = Column(JSON, comment="模型评估指标")
    hyperparameters = Column(JSON, comment="超参数配置")
    is_deployed = Column(Boolean, default=False, comment="是否已部署")
    trained_at = Column(DateTime, comment="训练时间")
    created_at = Column(DateTime, default=datetime.utcnow, comment="创建时间")

    def __repr__(self) -> str:
        return f"<ModelVersion(model='{self.model_name}', version='{self.version}')>"
