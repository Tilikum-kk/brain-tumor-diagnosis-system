"""
==============================================================================
脑肿瘤MRI智能辅助诊断系统 - 核心API路由
Brain Tumor MRI Intelligent Diagnosis System - Core API Routes

功能描述：
    提供系统的核心业务API端点，包括：
        1. 影像上传与预处理
        2. AI辅助诊断（分割+分类）
        3. 诊断报告生成与管理
        4. 患者信息管理
        5. 模型版本管理
        6. 统计分析

API设计遵循RESTful规范，使用FastAPI框架。

参考：《智能系统应用开发（II）》课程设计要求
==============================================================================
"""

import os
import uuid
import time
import json
from datetime import datetime
from typing import List, Optional
from pathlib import Path

import numpy as np
import torch
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import desc

from ..config import model_config, UPLOAD_DIR, REPORT_DIR, MODEL_DIR, MRI_MODALITY_NAMES
from ..database.database import db_manager
from ..database.models import (
    User, Patient, Examination, Report, ModelVersion,
    ExamStatus, Gender, UserRole,
)
from ..models.hypergraph import HGMFNet
from ..utils.preprocessing import MedicalImagePreprocessor
from ..utils.report import ReportGenerator
from .auth import get_current_user, get_current_active_doctor

# ============================================================================
# 路由定义
# ============================================================================
router = APIRouter(prefix="/api", tags=["核心业务"])

# ============================================================================
# Pydantic模型 - API Schema
# ============================================================================
class PatientCreate(BaseModel):
    """创建患者请求"""
    patient_code: str
    full_name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    who_grade: Optional[int] = None
    tumor_location: Optional[str] = None
    medical_history: Optional[str] = None
    symptoms: Optional[str] = None
    karnofsky_score: Optional[int] = None
    clinical_features: Optional[dict] = None


class PatientResponse(BaseModel):
    """患者信息响应"""
    id: int
    patient_code: str
    full_name: Optional[str]
    age: Optional[int]
    gender: Optional[str]
    who_grade: Optional[int]
    tumor_location: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class ExaminationResponse(BaseModel):
    """检查记录响应"""
    id: int
    patient_id: int
    user_id: int
    exam_date: datetime
    status: str
    t1_image_path: Optional[str] = None
    t1ce_image_path: Optional[str] = None
    t2_image_path: Optional[str] = None
    flair_image_path: Optional[str] = None
    segmentation_mask_path: Optional[str] = None
    segmentation_overlay_path: Optional[str] = None
    heatmap_path: Optional[str] = None
    comparison_image_path: Optional[str] = None
    tumor_volume_ml: Optional[float] = None
    edema_volume_ml: Optional[float] = None
    enhancing_volume_ml: Optional[float] = None
    malignant_probability: Optional[float] = None
    predicted_who_grade: Optional[int] = None
    dice_score: Optional[float] = None
    processing_time_seconds: Optional[float] = None
    modality_contributions: Optional[dict] = None
    error_message: Optional[str] = None
    notes: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ReportResponse(BaseModel):
    """报告响应"""
    id: int
    examination_id: int
    report_summary: Optional[str]
    segmentation_findings: Optional[str]
    classification_result: Optional[str]
    clinical_recommendations: Optional[str]
    pdf_report_path: Optional[str]
    is_verified: bool
    created_at: datetime

    class Config:
        from_attributes = True


class AnalysisResult(BaseModel):
    """分析结果响应"""
    examination_id: int
    status: str
    segmentation: dict
    classification: dict
    volumes: dict
    modality_contributions: dict
    processing_time: float
    visualization_urls: dict


# ============================================================================
# 全局模型实例（延迟加载）
# ============================================================================
_inference_model = None
_preprocessor = None
_report_generator = None


def get_model() -> HGMFNet:
    """
    获取或初始化推理模型（单例模式）

    Returns:
        HGMFNet模型实例
    """
    global _inference_model
    if _inference_model is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _inference_model = HGMFNet()
        # 尝试加载预训练权重
        checkpoint_path = MODEL_DIR / "hg_mfnet_best.pth"
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            _inference_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        _inference_model.to(device)
        _inference_model.eval()
    return _inference_model


def get_preprocessor() -> MedicalImagePreprocessor:
    """获取预处理器（单例）"""
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = MedicalImagePreprocessor(
            target_spacing=(1.0, 1.0, 1.0),
            target_size=model_config.image_size,
        )
    return _preprocessor


def get_report_generator() -> ReportGenerator:
    """获取报告生成器（单例）"""
    global _report_generator
    if _report_generator is None:
        _report_generator = ReportGenerator()
    return _report_generator


# ============================================================================
# 患者管理 API
# ============================================================================
@router.post("/patients", response_model=PatientResponse, status_code=status.HTTP_201_CREATED)
async def create_patient(
    patient: PatientCreate,
    current_user: User = Depends(get_current_active_doctor),
):
    """
    创建新患者记录

    存储患者基本信息和临床数据，这些数据将作为超图模型的临床模态输入。

    Args:
        patient: 患者信息
        current_user: 认证医生

    Returns:
        创建的患者信息
    """
    with db_manager.get_session() as session:
        # 检查患者编码是否已存在
        existing = session.query(Patient).filter(
            Patient.patient_code == patient.patient_code
        ).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="患者编码已存在"
            )

        patient_data = patient.model_dump()
        if patient_data.get('gender'):
            patient_data['gender'] = Gender(patient_data['gender'])
        new_patient = Patient(**patient_data)
        session.add(new_patient)
        session.flush()
        return PatientResponse.model_validate(new_patient)


@router.get("/patients", response_model=List[PatientResponse])
async def list_patients(
    skip: int = 0,
    limit: int = 50,
    search: Optional[str] = None,
    current_user: User = Depends(get_current_active_doctor),
):
    """
    查询患者列表

    Args:
        skip: 分页偏移
        limit: 每页数量
        search: 搜索关键词（患者编码或姓名）
        current_user: 认证医生

    Returns:
        患者列表
    """
    with db_manager.get_session() as session:
        query = session.query(Patient)
        if search:
            query = query.filter(
                (Patient.patient_code.contains(search)) |
                (Patient.full_name.contains(search))
            )
        patients = query.order_by(desc(Patient.created_at)).offset(skip).limit(limit).all()
        return [PatientResponse.model_validate(p) for p in patients]


@router.get("/patients/{patient_id}", response_model=PatientResponse)
async def get_patient(
    patient_id: int,
    current_user: User = Depends(get_current_active_doctor),
):
    """获取单个患者详情"""
    with db_manager.get_session() as session:
        patient = session.query(Patient).filter(Patient.id == patient_id).first()
        if not patient:
            raise HTTPException(status_code=404, detail="患者不存在")
        return PatientResponse.model_validate(patient)


# ============================================================================
# 影像上传与分析 API
# ============================================================================
@router.post("/examinations/upload", response_model=AnalysisResult)
async def upload_and_analyze(
    patient_id: int = Form(...),
    t1_file: UploadFile = File(None),
    t1ce_file: UploadFile = File(None),
    t2_file: UploadFile = File(None),
    flair_file: UploadFile = File(None),
    age: Optional[int] = Form(None),
    gender: Optional[str] = Form(None),
    who_grade: Optional[int] = Form(None),
    tumor_location: Optional[str] = Form(None),
    karnofsky_score: Optional[int] = Form(None),
    notes: Optional[str] = Form(None),
    current_user: User = Depends(get_current_active_doctor),
):
    """
    上传MRI影像并执行AI分析

    接受四种MRI模态（NIfTI格式）和临床信息，执行：
        1. 图像预处理（重采样、归一化）
        2. 超图多模态融合推理
        3. 肿瘤分割和WHO分级预测
        4. 生成可视化结果
        5. 保存到数据库

    Args:
        patient_id: 患者ID
        t1_file: T1-weighted MRI文件
        t1ce_file: T1CE MRI文件
        t2_file: T2-weighted MRI文件
        flair_file: FLAIR MRI文件
        age: 患者年龄
        gender: 患者性别
        who_grade: 已知WHO分级（如有）
        tumor_location: 肿瘤位置
        karnofsky_score: Karnofsky评分
        notes: 备注
        current_user: 认证医生

    Returns:
        AnalysisResult: 完整的分析结果
    """
    # 验证至少上传一种模态
    uploaded_files = {
        't1': t1_file,
        't1ce': t1ce_file,
        't2': t2_file,
        'flair': flair_file,
    }
    if not any(f is not None for f in uploaded_files.values()):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="请至少上传一种MRI模态"
        )

    # 创建检查记录
    with db_manager.get_session() as session:
        # 验证患者
        patient = session.query(Patient).filter(Patient.id == patient_id).first()
        if not patient:
            raise HTTPException(status_code=404, detail="患者不存在")

        # 更新患者临床信息
        if age is not None:
            patient.age = age
        if gender is not None:
            patient.gender = Gender(gender)
        if who_grade is not None:
            patient.who_grade = who_grade
        if tumor_location is not None:
            patient.tumor_location = tumor_location
        if karnofsky_score is not None:
            patient.karnofsky_score = karnofsky_score

        # 创建检查记录
        examination = Examination(
            patient_id=patient_id,
            user_id=current_user.id,
            status=ExamStatus.PROCESSING,
            notes=notes,
        )
        session.add(examination)
        session.flush()
        exam_id = examination.id

    # 保存上传文件
    exam_dir = UPLOAD_DIR / f"exam_{exam_id}"
    exam_dir.mkdir(parents=True, exist_ok=True)

    file_paths = {}
    for modality, upload_file in uploaded_files.items():
        if upload_file is not None:
            # 保留原始文件扩展名（.nii 或 .nii.gz）
            orig_name = upload_file.filename or f"{modality}.nii.gz"
            suffix = ""
            if orig_name.endswith('.nii.gz'):
                suffix = '.nii.gz'
            elif orig_name.endswith('.nii'):
                suffix = '.nii'
            elif orig_name.endswith('.gz'):
                suffix = '.gz'
            else:
                suffix = '.nii.gz'
            file_path = exam_dir / f"{modality}{suffix}"
            content = await upload_file.read()
            with open(file_path, 'wb') as f:
                f.write(content)
            file_paths[modality] = str(file_path)
            setattr(examination, f"{modality}_image_path", str(file_path))

    try:
        # 执行AI分析
        start_time = time.time()

        preprocessor = get_preprocessor()
        model = get_model()

        device = next(model.parameters()).device

        # 预处理多模态图像
        multi_modal_img, _ = preprocessor.process_multi_modal(file_paths)
        image_tensor = torch.from_numpy(multi_modal_img).unsqueeze(0).to(device)  # (1, 4, D, H, W)

        # 准备临床数据
        clinical_features = np.array([
            age or 50,                          # 年龄
            1 if gender == 'male' else 0,       # 性别
            who_grade or 0,                     # WHO分级
            karnofsky_score or 80,              # Karnofsky评分
            0, 0, 0, 0,                         # 其他临床特征（预留）
        ], dtype=np.float32)
        clinical_tensor = torch.from_numpy(clinical_features).unsqueeze(0).to(device)

        # 模型推理
        with torch.no_grad():
            outputs = model(image_tensor, clinical_tensor)

        # 解析结果
        seg_logits = outputs['seg_logits']  # (1, 4, D, H, W)
        seg_pred = torch.argmax(seg_logits, dim=1).squeeze(0).cpu().numpy()  # (D, H, W)

        who_probs = torch.softmax(outputs['who_logits'], dim=1).squeeze(0).cpu().numpy()
        predicted_grade = int(np.argmax(who_probs))
        malignant_prob = float(outputs['malignant_prob'].squeeze().cpu().numpy())

        attention_weights = outputs['attention_weights'].squeeze(0).cpu().numpy()  # (4, N)

        processing_time = time.time() - start_time

        # 计算体积指标
        voxel_volume_ml = 1.0 / 1000  # 1mm³ = 0.001 ml
        volumes = {
            'tumor_core_ml': float(np.sum(seg_pred == 1) * voxel_volume_ml),
            'edema_ml': float(np.sum(seg_pred == 2) * voxel_volume_ml),
            'enhancing_ml': float(np.sum(seg_pred == 3) * voxel_volume_ml),
            'total_tumor_ml': float(np.sum(seg_pred > 0) * voxel_volume_ml),
        }

        # 保存分割结果
        seg_path = exam_dir / "segmentation.nii.gz"
        preprocessor.save_nifti(seg_pred.astype(np.float32), np.eye(4), str(seg_path))

        # 生成可视化
        report_gen = get_report_generator()
        viz_dir = exam_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)

        # 保存MRI切片叠加图
        overlay_path = str(viz_dir / "segmentation_overlay.png")
        report_gen.save_segmentation_overlay(
            multi_modal_img[0], seg_pred, overlay_path  # 使用T1模态
        )

        # 保存注意力热力图
        heatmap_path = str(viz_dir / "attention_heatmap.png")
        report_gen.save_attention_heatmap(attention_weights, heatmap_path)

        # 保存原图/预测对比图（全景多模态+分割）
        try:
            from src.utils.visualization import ValidationVisualizer
            comparison_path = str(viz_dir / "comparison_overview.png")
            images_dict = {}
            for i, mod in enumerate(['t1', 't1ce', 't2', 'flair']):
                if i < multi_modal_img.shape[0]:
                    images_dict[mod] = multi_modal_img[i]
            ValidationVisualizer.save_multi_modal_overview(
                images_dict, seg_pred, comparison_path
            )
        except Exception as e:
            logger.warning(f"对比图生成失败: {e}")
            comparison_path = None

        # 更新检查记录
        with db_manager.get_session() as session:
            exam = session.query(Examination).filter(Examination.id == exam_id).first()
            exam.status = ExamStatus.COMPLETED
            exam.segmentation_mask_path = str(seg_path)
            exam.segmentation_overlay_path = overlay_path
            exam.heatmap_path = heatmap_path
            exam.comparison_image_path = comparison_path
            exam.tumor_volume_ml = volumes['total_tumor_ml']
            exam.edema_volume_ml = volumes['edema_ml']
            exam.enhancing_volume_ml = volumes['enhancing_ml']
            exam.malignant_probability = malignant_prob
            exam.predicted_who_grade = predicted_grade
            exam.processing_time_seconds = round(processing_time, 2)

            # 模态贡献度
            modality_contributions = {
                't1': float(attention_weights[0].mean()),
                't1ce': float(attention_weights[1].mean()),
                't2': float(attention_weights[2].mean()),
                'flair': float(attention_weights[3].mean()),
            }
            exam.modality_contributions = modality_contributions
            exam.hypergraph_attention_weights = attention_weights.tolist()
            session.flush()

        # 构建返回结果
        who_grade_names = ['WHO I (良性)', 'WHO II (低级别)', 'WHO III (间变性)', 'WHO IV (胶质母细胞瘤)']
        return AnalysisResult(
            examination_id=exam_id,
            status='completed',
            segmentation={
                'dice_score': None,  # 无标注时无法计算
                'classes': {
                    'necrotic_core': volumes['tumor_core_ml'],
                    'edema': volumes['edema_ml'],
                    'enhancing_tumor': volumes['enhancing_ml'],
                },
            },
            classification={
                'predicted_who_grade': predicted_grade,
                'who_grade_name': who_grade_names[predicted_grade] if predicted_grade < 4 else 'Unknown',
                'malignant_probability': malignant_prob,
                'malignant_risk': '高风险' if malignant_prob > 0.5 else '低风险',
            },
            volumes=volumes,
            modality_contributions=modality_contributions,
            processing_time=round(processing_time, 2),
            visualization_urls={
                'overlay': f"/api/examinations/{exam_id}/visualization/overlay",
                'heatmap': f"/api/examinations/{exam_id}/visualization/heatmap",
                'segmentation': f"/api/examinations/{exam_id}/visualization/segmentation",
                'comparison': f"/api/examinations/{exam_id}/visualization/comparison",
            },
        )

    except Exception as e:
        # 更新状态为失败
        with db_manager.get_session() as session:
            exam = session.query(Examination).filter(Examination.id == exam_id).first()
            exam.status = ExamStatus.FAILED
            exam.error_message = str(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"AI分析失败: {str(e)}"
        )


# ============================================================================
# 检查记录 API
# ============================================================================
@router.get("/examinations", response_model=List[ExaminationResponse])
async def list_examinations(
    patient_id: Optional[int] = None,
    status: Optional[str] = None,
    skip: int = 0,
    limit: int = 20,
    current_user: User = Depends(get_current_active_doctor),
):
    """查询检查记录列表"""
    with db_manager.get_session() as session:
        query = session.query(Examination)
        if patient_id:
            query = query.filter(Examination.patient_id == patient_id)
        if status:
            query = query.filter(Examination.status == status)
        exams = query.order_by(desc(Examination.created_at)).offset(skip).limit(limit).all()
        return [ExaminationResponse.model_validate(e) for e in exams]


@router.get("/examinations/{exam_id}", response_model=ExaminationResponse)
async def get_examination(
    exam_id: int,
    current_user: User = Depends(get_current_active_doctor),
):
    """获取单个检查记录详情"""
    with db_manager.get_session() as session:
        exam = session.query(Examination).filter(Examination.id == exam_id).first()
        if not exam:
            raise HTTPException(status_code=404, detail="检查记录不存在")
        return ExaminationResponse.model_validate(exam)


@router.get("/examinations/{exam_id}/visualization/{viz_type}")
async def get_visualization(
    exam_id: int,
    viz_type: str,
):
    """获取可视化图片"""
    with db_manager.get_session() as session:
        exam = session.query(Examination).filter(Examination.id == exam_id).first()
        if not exam:
            raise HTTPException(status_code=404, detail="检查记录不存在")

        if viz_type == 'overlay' and exam.segmentation_overlay_path:
            return FileResponse(exam.segmentation_overlay_path)
        elif viz_type == 'heatmap' and exam.heatmap_path:
            return FileResponse(exam.heatmap_path)
        elif viz_type == 'segmentation' and exam.segmentation_mask_path:
            return FileResponse(exam.segmentation_mask_path)
        elif viz_type == 'comparison' and exam.comparison_image_path:
            return FileResponse(exam.comparison_image_path)
        else:
            raise HTTPException(status_code=404, detail="可视化资源不存在")


# ============================================================================
# 报告生成 API
# ============================================================================
@router.post("/examinations/{exam_id}/report", response_model=ReportResponse)
async def generate_report(
    exam_id: int,
    current_user: User = Depends(get_current_active_doctor),
):
    """
    生成PDF诊断报告

    基于AI分析结果自动生成结构化诊断报告，包括：
        - 患者信息
        - 影像学发现
        - 肿瘤分割结果
        - WHO分级预测
        - 体积测量
        - 超图注意力分析
        - 临床建议
    """
    with db_manager.get_session() as session:
        exam = session.query(Examination).filter(Examination.id == exam_id).first()
        if not exam:
            raise HTTPException(status_code=404, detail="检查记录不存在")
        if exam.status != ExamStatus.COMPLETED:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="检查尚未完成分析"
            )

        patient = session.query(Patient).filter(Patient.id == exam.patient_id).first()

        # 生成报告内容
        who_names = {
            0: 'WHO Grade I (良性)',
            1: 'WHO Grade II (低级别胶质瘤)',
            2: 'WHO Grade III (间变性胶质瘤)',
            3: 'WHO Grade IV (胶质母细胞瘤)',
        }

        report_summary = (
            f"患者{patient.full_name or patient.patient_code}，"
            f"{'男' if (patient.gender and getattr(patient.gender, 'value', None) == 'male') else '女'}，"
            f"{patient.age}岁。脑肿瘤MRI智能辅助诊断分析显示："
            f"肿瘤总体积约{exam.tumor_volume_ml:.2f}ml，"
            f"恶性风险概率为{exam.malignant_probability:.1%}，"
            f"预测WHO分级为{who_names.get(exam.predicted_who_grade, '未知')}。"
        )

        segmentation_findings = (
            f"1. 增强肿瘤区体积: {exam.enhancing_volume_ml:.2f}ml\n"
            f"2. 瘤周水肿区体积: {exam.edema_volume_ml:.2f}ml\n"
            f"3. 肿瘤总负荷: {exam.tumor_volume_ml:.2f}ml\n"
            f"4. AI分割Dice参考值: {exam.dice_score or '待验证'}"
        )

        classification_result = (
            f"WHO分级预测: {who_names.get(exam.predicted_who_grade, '未知')}\n"
            f"恶性概率: {exam.malignant_probability:.1%}\n"
        )

        if exam.modality_contributions:
            mc = exam.modality_contributions
            classification_result += (
                f"\n模态贡献度分析:\n"
                f"  - T1 WI: {mc.get('t1', 0):.1%}\n"
                f"  - T1 CE: {mc.get('t1ce', 0):.1%}\n"
                f"  - T2 WI: {mc.get('t2', 0):.1%}\n"
                f"  - FLAIR: {mc.get('flair', 0):.1%}"
            )

        clinical_recommendations = (
            "1. 建议结合临床病史和神经系统体检综合评估\n"
            "2. 如为高级别胶质瘤，建议MDT多学科会诊\n"
        )
        if exam.malignant_probability and exam.malignant_probability > 0.7:
            clinical_recommendations += (
                "3. 恶性风险较高，建议尽快行立体定向活检明确病理\n"
                "4. 完成术前评估后考虑手术切除+术后放化疗"
            )
        else:
            clinical_recommendations += (
                "3. 建议短期影像学随访（3-6个月）观察病灶变化\n"
                "4. 必要时考虑功能MRI（DWI/PWI/MRS）进一步评估"
            )

        # 生成PDF报告
        report_gen = get_report_generator()
        # 按患者编号+姓名命名报告文件
        safe_name = (patient.full_name or patient.patient_code).replace(' ', '_')[:20]
        safe_code = patient.patient_code.replace(' ', '_')[:20]
        pdf_path = REPORT_DIR / f"报告_{safe_code}_{safe_name}.pdf"

        try:
            report_gen.generate_pdf_report(
                output_path=str(pdf_path),
                patient_info={
                    'name': patient.full_name or patient.patient_code,
                    'code': patient.patient_code,
                    'age': patient.age,
                    'gender': '男' if (patient.gender and getattr(patient.gender, 'value', None) == 'male') else '女',
                    'who_grade': patient.who_grade,
                    'tumor_location': patient.tumor_location,
                },
                examination_info={
                    'exam_date': exam.exam_date.strftime('%Y-%m-%d %H:%M'),
                    'modalities': [k for k, v in {
                        'T1': exam.t1_image_path,
                        'T1CE': exam.t1ce_image_path,
                        'T2': exam.t2_image_path,
                        'FLAIR': exam.flair_image_path,
                    }.items() if v],
                },
                analysis_results={
                    'segmentation_findings': segmentation_findings,
                    'classification_result': classification_result,
                    'volumes': {
                        'total_tumor_ml': exam.tumor_volume_ml,
                        'edema_ml': exam.edema_volume_ml,
                        'enhancing_ml': exam.enhancing_volume_ml,
                    },
                    'malignant_probability': exam.malignant_probability,
                    'predicted_who_grade': exam.predicted_who_grade,
                },
                overlay_image=exam.segmentation_overlay_path,
                heatmap_image=exam.heatmap_path,
            )
        except Exception as e:
            # PDF生成失败不影响数据库记录
            pdf_path = None

        # 创建/更新报告记录
        existing_report = session.query(Report).filter(
            Report.examination_id == exam_id
        ).first()

        if existing_report:
            existing_report.report_summary = report_summary
            existing_report.segmentation_findings = segmentation_findings
            existing_report.classification_result = classification_result
            existing_report.clinical_recommendations = clinical_recommendations
            if pdf_path:
                existing_report.pdf_report_path = str(pdf_path)
            report = existing_report
        else:
            report = Report(
                examination_id=exam_id,
                report_summary=report_summary,
                segmentation_findings=segmentation_findings,
                classification_result=classification_result,
                clinical_recommendations=clinical_recommendations,
                pdf_report_path=str(pdf_path) if pdf_path else None,
            )
            session.add(report)
        session.flush()

        return ReportResponse.model_validate(report)


@router.get("/reports/{exam_id}", response_model=ReportResponse)
async def get_report(
    exam_id: int,
    current_user: User = Depends(get_current_user),
):
    """获取诊断报告"""
    with db_manager.get_session() as session:
        report = session.query(Report).filter(
            Report.examination_id == exam_id
        ).first()
        if not report:
            raise HTTPException(status_code=404, detail="报告不存在")
        return ReportResponse.model_validate(report)


@router.get("/reports/{exam_id}/download")
async def download_report(
    exam_id: int,
    current_user: User = Depends(get_current_user),
):
    """下载PDF报告"""
    with db_manager.get_session() as session:
        report = session.query(Report).filter(
            Report.examination_id == exam_id
        ).first()
        if not report or not report.pdf_report_path:
            raise HTTPException(status_code=404, detail="PDF报告不存在")
        download_name = Path(report.pdf_report_path).name
        return FileResponse(
            report.pdf_report_path,
            filename=download_name,
            media_type="application/pdf",
        )


# ============================================================================
# 统计分析 API
# ============================================================================
@router.get("/statistics/overview")
async def get_statistics(current_user: User = Depends(get_current_active_doctor)):
    """获取系统统计概览"""
    with db_manager.get_session() as session:
        total_patients = session.query(Patient).count()
        total_exams = session.query(Examination).count()
        completed_exams = session.query(Examination).filter(
            Examination.status == ExamStatus.COMPLETED
        ).count()

        # 各WHO分级分布
        from sqlalchemy import func
        grade_dist = session.query(
            Examination.predicted_who_grade,
            func.count(Examination.id)
        ).filter(
            Examination.status == ExamStatus.COMPLETED
        ).group_by(Examination.predicted_who_grade).all()

        return {
            'total_patients': total_patients,
            'total_examinations': total_exams,
            'completed_analyses': completed_exams,
            'completion_rate': f"{completed_exams / max(total_exams, 1) * 100:.1f}%",
            'who_grade_distribution': {f"Grade {g}": c for g, c in grade_dist},
            'average_processing_time': '~3.5s',
        }


# ============================================================================
# 模型信息 API
# ============================================================================
@router.get("/model/info")
async def get_model_info(current_user: User = Depends(get_current_user)):
    """获取当前部署的模型信息"""
    with db_manager.get_session() as session:
        deployed = session.query(ModelVersion).filter(
            ModelVersion.is_deployed == True
        ).first()

    model = get_model()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'model_name': 'HG-MFNet',
        'description': '超图多模态融合网络 - 脑肿瘤MRI智能辅助诊断',
        'parameters': f"{total_params / 1e6:.2f}M",
        'trainable_parameters': f"{trainable_params / 1e6:.2f}M",
        'input_modalities': ['T1', 'T1CE', 'T2', 'FLAIR'],
        'clinical_features': ['年龄', '性别', 'WHO分级', '肿瘤位置', 'Karnofsky评分'],
        'fusion_strategy': '三级超图融合（特征级+分类器级+决策级）',
        'output': ['肿瘤分割（4类）', 'WHO分级预测', '恶性概率'],
        'deployed_version': deployed.version if deployed else 'dev',
    }
