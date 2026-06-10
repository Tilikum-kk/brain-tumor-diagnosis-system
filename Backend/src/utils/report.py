"""
==============================================================================
脑肿瘤MRI智能辅助诊断系统 - 诊断报告生成器
Brain Tumor MRI Intelligent Diagnosis System - Report Generator

功能描述：
    自动生成结构化PDF诊断报告，包括：
        1. 报告封面与基本信息
        2. 影像学发现（肿瘤分割结果）
        3. WHO分级预测与恶性风险评估
        4. 体积测量数据
        5. 超图注意力分析（模态贡献度）
        6. 可视化图示（分割覆盖图、热力图）
        7. 临床建议与参考文献

使用ReportLab库生成PDF，Matplotlib生成嵌入图像。

参考：《学术论文编写规则》（GB/T 7713.2—2022）
==============================================================================
"""

import os
import io
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap


class ReportGenerator:
    """
    医学影像诊断报告生成器

    生成包含AI辅助诊断结果的完整PDF报告。
    支持中文文本渲染和医学图像嵌入。

    报告结构：
        1. 封面：标题、患者信息、检查日期
        2. 影像学发现：分割结果描述、体积测量
        3. AI分析结果：WHO分级、恶性概率、超图注意力
        4. 可视化图像：分割覆盖图、热力图
        5. 临床建议
    """

    def __init__(self):
        """初始化报告生成器"""
        self._init_plot_style()

    def _init_plot_style(self) -> None:
        """初始化matplotlib绘图样式"""
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['figure.dpi'] = 150
        plt.rcParams['savefig.dpi'] = 150
        plt.rcParams['savefig.bbox'] = 'tight'

    def save_segmentation_overlay(self, mri_slice: np.ndarray,
                                   seg_mask: np.ndarray,
                                   output_path: str,
                                   slice_idx: Optional[int] = None) -> str:
        """
        生成分割覆盖图

        将肿瘤分割结果以彩色轮廓叠加在MRI切片上。

        Args:
            mri_slice: MRI图像切片 (H, W) 或 (D, H, W)
            seg_mask: 分割掩码 (H, W) 或 (D, H, W)
            output_path: 输出文件路径
            slice_idx: 切片索引（如果输入为3D，默认选肿瘤面积最大的切片）

        Returns:
            输出文件路径
        """
        # 选择切片 — 优先选肿瘤面积最大的层
        if mri_slice.ndim == 3:
            if slice_idx is None and seg_mask.ndim == 3:
                tumor_per_slice = (seg_mask > 0).sum(axis=(1, 2))
                slice_idx = int(np.argmax(tumor_per_slice)) if tumor_per_slice.max() > 0 else mri_slice.shape[0] // 2
            elif slice_idx is None:
                slice_idx = mri_slice.shape[0] // 2
            mri_slice = mri_slice[slice_idx]
        if seg_mask.ndim == 3:
            if slice_idx is None:
                tumor_per_slice = (seg_mask > 0).sum(axis=(1, 2))
                slice_idx = int(np.argmax(tumor_per_slice)) if tumor_per_slice.max() > 0 else seg_mask.shape[0] // 2
            seg_mask = seg_mask[slice_idx]

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # 1. 原始MRI — 大图
        axes[0].imshow(mri_slice, cmap='gray')
        axes[0].set_title('Original MRI', fontsize=14, fontweight='bold')
        axes[0].axis('off')

        # 2. 分割覆盖图 — RGBA 固体叠加 + 图例
        axes[1].imshow(mri_slice, cmap='gray')
        # 使用与 ValidationVisualizer 相同的 RGBA 固体叠加方式
        rgba = np.zeros((*seg_mask.shape, 4))
        class_colors = {
            1: [1, 0, 0, 0.55],    # 红色 - 坏死核心
            2: [0, 0, 1, 0.55],    # 蓝色 - 水肿
            3: [1, 1, 0, 0.55],    # 黄色 - 增强肿瘤
        }
        for c, color in class_colors.items():
            rgba[seg_mask == c] = color
        axes[1].imshow(rgba)

        from matplotlib.patches import Patch
        legend_patches = [
            Patch(facecolor='#ff0000', alpha=0.55, label='Necrotic Core'),
            Patch(facecolor='#0000ff', alpha=0.55, label='Edema'),
            Patch(facecolor='#ffff00', alpha=0.55, label='Enhancing Tumor'),
        ]
        axes[1].legend(handles=legend_patches, loc='lower right', fontsize=11,
                      framealpha=0.9, edgecolor='gray')
        axes[1].set_title('Tumor Segmentation Overlay', fontsize=14, fontweight='bold')
        axes[1].axis('off')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return output_path

    def save_attention_heatmap(self, attention_weights: np.ndarray,
                                output_path: str) -> str:
        """
        生成超图注意力热力图

        显示各MRI模态对诊断的贡献度。

        Args:
            attention_weights: 注意力权重 (4,) 或 (4, N)
            output_path: 输出文件路径
        """
        modality_names = ['T1 WI', 'T1 CE', 'T2 WI', 'FLAIR']

        # 如果有多维数据，取平均值
        if attention_weights.ndim > 1:
            weights = attention_weights.mean(axis=1)
        else:
            weights = attention_weights

        # 归一化
        weights = weights / (weights.sum() + 1e-8)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # 柱状图
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        bars = ax1.bar(modality_names, weights, color=colors, edgecolor='white', linewidth=1.5)
        ax1.set_ylabel('注意力权重', fontsize=12)
        ax1.set_title('超图注意力 - 模态贡献度', fontsize=14, fontweight='bold')
        ax1.set_ylim(0, max(weights) * 1.3)

        # 添加数值标签
        for bar, weight in zip(bars, weights):
            ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                    f'{weight:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')

        # 饼图
        wedges, texts, autotexts = ax2.pie(
            weights, labels=modality_names, colors=colors,
            autopct='%1.1f%%', startangle=90,
            explode=(0.05, 0.05, 0.05, 0.05),
        )
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        ax2.set_title('模态贡献占比', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return output_path

    def save_3d_visualization(self, seg_volume: np.ndarray,
                               output_path: str) -> str:
        """
        生成3D肿瘤可视化

        使用多视角切片展示肿瘤的3D结构。

        Args:
            seg_volume: 3D分割掩码 (D, H, W)
            output_path: 输出文件路径
        """
        D, H, W = seg_volume.shape

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 轴状位（多个切片）
        slice_positions = [D // 4, D // 2, 3 * D // 4]
        for i, pos in enumerate(slice_positions):
            ax = axes[0, i]
            ax.imshow(np.rot90(seg_volume[pos]), cmap='jet', vmin=0, vmax=3)
            ax.set_title(f'轴状位 (Z={pos})', fontsize=10)
            ax.axis('off')

        # 矢状位
        s_positions = [H // 4, H // 2, 3 * H // 4]
        for i, pos in enumerate(s_positions):
            ax = axes[1, i]
            ax.imshow(np.rot90(seg_volume[:, pos, :]), cmap='jet', vmin=0, vmax=3)
            ax.set_title(f'矢状位 (Y={pos})', fontsize=10)
            ax.axis('off')

        plt.suptitle('肿瘤3D分割可视化（多视角）', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return output_path

    def save_training_curves(self, metrics: Dict[str, List[float]],
                              output_path: str) -> str:
        """
        保存训练曲线图

        用于TensorBoard替代的可视化输出。

        Args:
            metrics: 指标字典 {'loss': [...], 'dice': [...], 'auc': [...]}
            output_path: 输出文件路径
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Loss曲线
        if 'loss' in metrics:
            axes[0].plot(metrics['loss'], label='Training Loss', color='#3498db', linewidth=2)
            if 'val_loss' in metrics:
                axes[0].plot(metrics['val_loss'], label='Validation Loss', color='#e74c3c', linewidth=2)
            axes[0].set_xlabel('Epoch', fontsize=11)
            axes[0].set_ylabel('Loss', fontsize=11)
            axes[0].set_title('损失曲线', fontsize=13, fontweight='bold')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

        # Dice曲线
        if 'dice' in metrics:
            axes[1].plot(metrics['dice'], label='Training Dice', color='#2ecc71', linewidth=2)
            if 'val_dice' in metrics:
                axes[1].plot(metrics['val_dice'], label='Validation Dice', color='#f39c12', linewidth=2)
            axes[1].set_xlabel('Epoch', fontsize=11)
            axes[1].set_ylabel('Dice Score', fontsize=11)
            axes[1].set_title('Dice系数曲线', fontsize=13, fontweight='bold')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        # AUC曲线
        if 'auc' in metrics:
            axes[2].plot(metrics['auc'], label='AUC', color='#9b59b6', linewidth=2)
            axes[2].set_xlabel('Epoch', fontsize=11)
            axes[2].set_ylabel('AUC', fontsize=11)
            axes[2].set_title('AUC曲线', fontsize=13, fontweight='bold')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return output_path

    def generate_pdf_report(self, output_path: str,
                            patient_info: Dict,
                            examination_info: Dict,
                            analysis_results: Dict,
                            overlay_image: Optional[str] = None,
                            heatmap_image: Optional[str] = None) -> str:
        """
        生成完整的PDF诊断报告

        Args:
            output_path: PDF输出路径
            patient_info: 患者信息字典
            examination_info: 检查信息字典
            analysis_results: 分析结果字典
            overlay_image: 分割覆盖图路径
            heatmap_image: 注意力热力图路径

        Returns:
            PDF文件路径
        """
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.units import mm, cm
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.colors import HexColor, black, white, grey
            from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph, Spacer, Image, Table,
                TableStyle, PageBreak, KeepTogether,
            )
            from reportlab.platypus.flowables import HRFlowable
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont
        except ImportError:
            # 如果reportlab不可用，创建简单的文本报告
            return self._generate_simple_report(output_path, patient_info,
                                                 examination_info, analysis_results)

        # 注册中文字体
        try:
            # 尝试常见中文字体路径
            chinese_font_paths = [
                'C:/Windows/Fonts/simhei.ttf',
                'C:/Windows/Fonts/msyh.ttf',
                'C:/Windows/Fonts/simsun.ttf',
                '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
                '/System/Library/Fonts/Hiragino Sans GB.ttc',
            ]
            font_registered = False
            for font_path in chinese_font_paths:
                if os.path.exists(font_path):
                    pdfmetrics.registerFont(TTFont('ChineseFont', font_path))
                    font_registered = True
                    break

            if not font_registered:
                # 使用默认字体
                body_font = 'Helvetica'
                title_font = 'Helvetica-Bold'
            else:
                body_font = 'ChineseFont'
                title_font = 'ChineseFont'
        except Exception:
            body_font = 'Helvetica'
            title_font = 'Helvetica-Bold'

        # 创建文档
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=20*mm,
            leftMargin=20*mm,
            topMargin=15*mm,
            bottomMargin=15*mm,
            title=f"脑肿瘤MRI智能辅助诊断报告 - {patient_info.get('name', '')}",
            author="AI辅助诊断系统 (HG-MFNet)",
            subject="脑肿瘤MRI诊断报告",
        )

        # 样式
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'ChineseTitle', fontName=title_font, fontSize=18,
            leading=24, alignment=TA_CENTER, spaceAfter=10*mm,
            textColor=HexColor('#1a5276'),
        )
        heading1_style = ParagraphStyle(
            'ChineseHeading1', fontName=title_font, fontSize=14,
            leading=20, spaceBefore=8*mm, spaceAfter=4*mm,
            textColor=HexColor('#2c3e50'),
        )
        heading2_style = ParagraphStyle(
            'ChineseHeading2', fontName=title_font, fontSize=12,
            leading=18, spaceBefore=5*mm, spaceAfter=2*mm,
            textColor=HexColor('#34495e'),
        )
        body_style = ParagraphStyle(
            'ChineseBody', fontName=body_font, fontSize=10,
            leading=16, alignment=TA_JUSTIFY, spaceBefore=2*mm,
        )

        # 构建报告内容
        story = []

        # ========== 封面 ==========
        story.append(Spacer(1, 30*mm))
        story.append(Paragraph(
            "脑肿瘤MRI智能辅助诊断系统",
            title_style,
        ))
        story.append(Spacer(1, 5*mm))
        story.append(Paragraph(
            "Brain Tumor MRI Intelligent Diagnosis Report",
            ParagraphStyle('Subtitle', fontName='Helvetica', fontSize=12,
                          alignment=TA_CENTER, textColor=HexColor('#7f8c8d')),
        ))
        story.append(Spacer(1, 10*mm))
        story.append(HRFlowable(width="80%", thickness=1, color=HexColor('#3498db')))
        story.append(Spacer(1, 10*mm))

        # 报告信息表格（报告日期使用AI检测时间）
        ai_exam_date = examination_info.get('exam_date', datetime.now().strftime('%Y-%m-%d %H:%M'))
        info_data = [
            ['患者姓名', patient_info.get('name', 'N/A')],
            ['患者编码', patient_info.get('code', 'N/A')],
            ['年龄/性别', f"{patient_info.get('age', 'N/A')}岁 / {patient_info.get('gender', 'N/A')}"],
            ['AI检测日期', ai_exam_date],
            ['报告日期', ai_exam_date],
            ['检查模态', ', '.join(examination_info.get('modalities', ['N/A']))],
            ['分析模型', 'HG-MFNet (超图多模态融合网络)'],
        ]

        info_table = Table(info_data, colWidths=[40*mm, 100*mm])
        info_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), body_font),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('FONTNAME', (0, 0), (0, -1), title_font),
            ('TEXTCOLOR', (0, 0), (0, -1), HexColor('#2c3e50')),
            ('BACKGROUND', (0, 0), (0, -1), HexColor('#ebf5fb')),
            ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#bdc3c7')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(info_table)

        story.append(PageBreak())

        # ========== AI分析结果 ==========
        story.append(Paragraph("一、AI辅助分析结果", heading1_style))
        story.append(HRFlowable(width="100%", thickness=1, color=HexColor('#3498db')))

        # 分割发现
        story.append(Paragraph("1.1 肿瘤分割发现", heading2_style))
        story.append(Paragraph(
            analysis_results.get('segmentation_findings', '无数据').replace('\n', '<br/>'),
            body_style,
        ))

        # 分类结果
        story.append(Paragraph("1.2 WHO分级预测", heading2_style))
        cls_result = analysis_results.get('classification_result', '无数据')
        story.append(Paragraph(cls_result.replace('\n', '<br/>'), body_style))

        # 体积分析
        volumes = analysis_results.get('volumes', {})
        story.append(Paragraph("1.3 体积测量", heading2_style))
        vol_data = [
            ['测量项目', '体积 (ml)', '临床意义'],
            ['肿瘤总体积', f"{volumes.get('total_tumor_ml', 0):.2f}", '评估肿瘤负荷'],
            ['瘤周水肿', f"{volumes.get('edema_ml', 0):.2f}", '评估占位效应'],
            ['增强肿瘤区', f"{volumes.get('enhancing_ml', 0):.2f}", '活性肿瘤区域'],
        ]
        vol_table = Table(vol_data, colWidths=[60*mm, 40*mm, 60*mm])
        vol_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), body_font),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('FONTNAME', (0, 0), (-1, 0), title_font),
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#2c3e50')),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#bdc3c7')),
            ('ALIGN', (1, 1), (1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(vol_table)

        story.append(PageBreak())

        # ========== 可视化图像 ==========
        story.append(Paragraph("二、可视化结果", heading1_style))
        story.append(HRFlowable(width="100%", thickness=1, color=HexColor('#3498db')))

        if overlay_image and os.path.exists(overlay_image):
            story.append(Paragraph("2.1 肿瘤分割覆盖图", heading2_style))
            img = Image(overlay_image, width=160*mm, height=55*mm)
            story.append(img)
            story.append(Paragraph(
                "图1: 肿瘤分割结果。红色=坏死核心，蓝色=瘤周水肿，黄色=增强肿瘤区。",
                ParagraphStyle('Caption', fontName=body_font, fontSize=8,
                              textColor=HexColor('#7f8c8d'), alignment=TA_CENTER),
            ))

        if heatmap_image and os.path.exists(heatmap_image):
            story.append(Paragraph("2.2 超图注意力模态贡献分析", heading2_style))
            img = Image(heatmap_image, width=160*mm, height=60*mm)
            story.append(img)
            story.append(Paragraph(
                "图2: 超图注意力权重可视化 - 显示各MRI模态对肿瘤诊断的贡献度。",
                ParagraphStyle('Caption', fontName=body_font, fontSize=8,
                              textColor=HexColor('#7f8c8d'), alignment=TA_CENTER),
            ))

        # ========== 临床建议 ==========
        story.append(PageBreak())
        story.append(Paragraph("三、临床建议", heading1_style))
        story.append(HRFlowable(width="100%", thickness=1, color=HexColor('#3498db')))

        story.append(Paragraph(
            analysis_results.get('clinical_recommendations',
                                 '请结合临床病史综合评估。').replace('\n', '<br/>'),
            body_style,
        ))

        # 免责声明
        story.append(Spacer(1, 15*mm))
        story.append(HRFlowable(width="100%", thickness=0.5, color=HexColor('#bdc3c7')))
        story.append(Paragraph(
            "<i>声明：本报告由AI辅助诊断系统自动生成，仅供临床参考，不能替代专业医师的诊断意见。"
            "最终诊断结果需由具备资质的放射科医师确认。系统基于HG-MFNet超图多模态融合模型，"
            "在BraTS 2020数据集上训练验证。</i>",
            ParagraphStyle('Disclaimer', fontName=body_font, fontSize=7,
                          textColor=HexColor('#95a5a6'), alignment=TA_CENTER),
        ))

        # 生成PDF
        doc.build(story)
        return output_path

    def _generate_simple_report(self, output_path: str,
                                 patient_info: Dict,
                                 examination_info: Dict,
                                 analysis_results: Dict) -> str:
        """生成简单文本报告（reportlab不可用时的备用方案）"""
        report_text = f"""
============================================================
        脑肿瘤MRI智能辅助诊断系统 - 诊断报告
        Brain Tumor MRI Intelligent Diagnosis Report
============================================================

【基本信息】
  患者姓名: {patient_info.get('name', 'N/A')}
  患者编码: {patient_info.get('code', 'N/A')}
  年龄/性别: {patient_info.get('age', 'N/A')}岁 / {patient_info.get('gender', 'N/A')}
  肿瘤位置: {patient_info.get('tumor_location', 'N/A')}
  WHO分级: {patient_info.get('who_grade', 'N/A')}
  检查日期: {examination_info.get('exam_date', 'N/A')}
  报告日期: {datetime.now().strftime('%Y-%m-%d %H:%M')}
  检查模态: {', '.join(examination_info.get('modalities', ['N/A']))}

【AI分析结果】
  预测WHO分级: {analysis_results.get('predicted_who_grade', 'N/A')}
  恶性概率: {analysis_results.get('malignant_probability', 0):.1%}

【体积测量】
  肿瘤总体积: {analysis_results.get('volumes', {}).get('total_tumor_ml', 0):.2f} ml
  瘤周水肿: {analysis_results.get('volumes', {}).get('edema_ml', 0):.2f} ml
  增强肿瘤区: {analysis_results.get('volumes', {}).get('enhancing_ml', 0):.2f} ml

【分割发现】
{analysis_results.get('segmentation_findings', '无数据')}

【分类结果】
{analysis_results.get('classification_result', '无数据')}

============================================================
声明：本报告由AI辅助诊断系统自动生成，仅供临床参考。
============================================================
"""
        # 保存为txt（备用）
        txt_path = output_path.replace('.pdf', '.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(report_text)

        # 同时尝试生成PDF
        try:
            self._generate_minimal_pdf(output_path, report_text)
        except Exception:
            pass

        return output_path

    def _generate_minimal_pdf(self, output_path: str, text: str) -> None:
        """生成最小化PDF（无中文特殊处理）"""
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph
        from reportlab.lib.styles import ParagraphStyle

        doc = SimpleDocTemplate(output_path, pagesize=A4)
        style = ParagraphStyle('Normal', fontSize=10)
        story = [Paragraph(text.replace('\n', '<br/>'), style)]
        doc.build(story)
