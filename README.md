# 脑肿瘤MRI智能辅助诊断系统

## Brain Tumor MRI Intelligent Diagnosis System

基于**超图神经网络（HG-MFNet）**的多模态 MRI 脑肿瘤智能辅助诊断系统。支持 T1、T1CE、T2、FLAIR 四种 MRI 模态的自动肿瘤分割与 WHO 分级预测。

> **作者**: 梁昊 (2023413304)  
> **单位**: 重庆工商大学 人工智能学院  
> **课程**: 《智能系统应用开发（II）》课程设计

---

## 系统架构

```
┌──────────────────────────────────────────────────┐
│             前端 (Vue3 + ElementPlus)             │
│     HTTP/REST API                                 │
├──────────────────────────────────────────────────┤
│           后端 (FastAPI + Uvicorn)                │
│  ┌──────────┬────────────┬─────────────────┐    │
│  │ 认证模块  │  业务API    │  报告生成模块    │    │
│  ├──────────┴────────────┴─────────────────┤    │
│  │         超图融合推理引擎                   │    │
│  │     HG-MFNet (PyTorch)                  │    │
│  ├──────────────────────────────────────────┤    │
│  │      SQLite 数据库 (SQLAlchemy)          │    │
│  └──────────────────────────────────────────┘    │
└──────────────────────────────────────────────────┘
```

---

## 项目结构

```
brain-tumor-diagnosis-system/
├── backend/                        # 后端服务
│   ├── main.py                     # FastAPI 入口（含完整 argparse CLI）
│   ├── requires.txt                # Python 依赖
│   ├── plot_charts.py              # 训练曲线 + 模型对比图生成脚本
│   ├── .env.example                # 环境变量示例
│   ├── src/
│   │   ├── config.py               # 全局配置（数据库/模型/API）
│   │   ├── train.py                # 训练脚本 v3（Warmup/Plateau/容错）
│   │   ├── eval.py                 # 独立评估脚本
│   │   ├── api/
│   │   │   ├── auth.py             # JWT 认证
│   │   │   └── routes.py           # 核心业务 API（上传/分析/报告）
│   │   ├── database/
│   │   │   ├── database.py         # 数据库连接管理
│   │   │   └── models.py           # ORM 模型（User/Patient/Examination/Report）
│   │   ├── models/
│   │   │   ├── hypergraph.py       # HG-MFNet 超图多模态融合网络（~930行）
│   │   │   ├── unet3d.py           # 3D U-Net / Attention U-Net 基线
│   │   │   └── fusion.py           # 多模态融合策略工厂（6种策略）
│   │   └── utils/
│   │       ├── preprocessing.py    # MRI 图像预处理管线（NIfTI/重采样/增强）
│   │       ├── pretrained.py       # MONAI 预训练权重下载与迁移
│   │       ├── visualization.py    # 训练曲线 + 验证对比图生成
│   │       └── report.py           # PDF 诊断报告生成（ReportLab）
│   ├── dataset/                    # [不上传] 数据集目录
│   ├── uploads/                    # [不上传] 用户上传文件
│   ├── reports/                    # [不上传] 生成的 PDF 报告
│   ├── checkpoints/                # [不上传] 模型权重
│   └── logs/                       # [不上传] 训练日志与 checkpoint
│
├── frontend/                       # 前端应用
│   ├── package.json                # npm 依赖
│   ├── vite.config.js              # Vite 构建配置
│   ├── index.html                  # HTML 入口
│   └── src/
│       ├── main.js                 # Vue 入口
│       ├── App.vue                 # 根组件（布局/主题/导航）
│       ├── router/index.js         # 路由配置
│       ├── api/index.js            # Axios API 层（JWT 拦截器）
│       └── views/
│           ├── Workspace.vue       # 核心工作台（上传→分析→结果）
│           ├── Result.vue          # 分析结果详情
│           ├── Login.vue           # 登录页
│           ├── Patients.vue        # 患者管理
│           └── History.vue         # 检查记录
│
├── .gitignore                      # Git 忽略规则
└── README.md
```

---

## 核心功能

| 功能 | 描述 |
|------|------|
| 🧠 多模态 MRI 分析 | 支持 T1、T1CE、T2、FLAIR 四种 MRI 模态 |
| 🔬 超图多模态融合 | 三级融合策略（特征级 → 分类器级 → 决策级） |
| 📊 肿瘤自动分割 | 识别坏死核心(NCR)、瘤周水肿(ED)、增强肿瘤(ET) |
| 🏥 WHO 分级预测 | 自动预测脑肿瘤 WHO 分级（I-IV 级） |
| 📋 智能报告 | 自动生成结构化 PDF 诊断报告 |
| 📈 可视化分析 | 验证对比图、训练曲线、模态贡献度分析 |

---

## HG-MFNet 模型

### 架构概览

```
T1 ──→ Encoder ──┐
T1CE → Encoder ──┤
T2 ──→ Encoder ──┼──→ HypergraphBuilder ──→ HypergraphConv(×3) ──→ Attention ──┬──→ SegDecoder → 肿瘤掩码
FLAIR → Encoder ──┘                                                            └──→ ClassHead  → WHO分级
```

### 三级融合策略

| 层级 | 方法 | 作用 |
|------|------|------|
| **特征级** | 超图卷积（128 超边，3 层） | 底层特征深度聚合 |
| **分类器级** | 8 头交叉模态注意力 | 高层语义自适应加权 |
| **决策级** | 超图正则化损失 | 约束各模态预测一致性 |

### 关键参数

| 参数 | 值 |
|------|-----|
| 输入尺寸 | 96×96×96 |
| 编码器通道 | [32, 64, 128, 256] |
| 超边数量 | 128 |
| 注意力头数 | 8 |
| 输出类别 | 4（背景/坏死核心/水肿/增强肿瘤） |

### 训练结果

| 指标 | 最佳值 | 对应 Epoch |
|------|:------:|:----------:|
| WT (全肿瘤) | 0.8785 | 80 |
| TC (坏死核心) | 0.6332 | 后续 |
| ED (水肿) | 0.8070 | 85 |
| ET (增强肿瘤) | 0.7592 | 80 |
| **Mean Dice** | **0.7682** | 85 |

> 训练集: BraTS 2020(494例) + BraTS 2021(1189例) = 1683 例  
> 验证集: BraTS 独立验证集 (62 例)  
> 训练耗时: ~34 小时 (NVIDIA RTX 4060 8GB, 100 epochs)

---

## 快速开始

### 环境要求

- Python 3.9+
- Node.js 18+
- NVIDIA GPU (8GB+ VRAM) 或 CPU（仅推理）
- Windows 10+ / Linux

### 1. 克隆项目

```bash
git clone https://github.com/your-username/brain-tumor-diagnosis-system.git
cd brain-tumor-diagnosis-system
```

### 2. 下载数据集

本系统使用 **BraTS (Brain Tumor Segmentation)** 公开数据集进行训练：

| 数据集 | 用途 | 下载链接 |
|--------|------|----------|
| BraTS 2020 | 训练 (494例) | [Synapse.org](https://www.synapse.org/Synapse:syn25829067) |
| BraTS 2021 | 训练 (1189例) | [Synapse.org](https://www.synapse.org/Synapse:syn27046444) |
| BraTS 验证集 | 验证 (62例) | 从 BraTS 2021 中筛选或下载官方验证集 |

下载后将数据集按以下结构放置：

```
backend/
└── dataset/
    ├── BraTS2020/
    │   ├── BraTS20_Training_001/
    │   │   ├── BraTS20_Training_001_flair.nii
    │   │   ├── BraTS20_Training_001_t1.nii
    │   │   ├── BraTS20_Training_001_t1ce.nii
    │   │   ├── BraTS20_Training_001_t2.nii
    │   │   └── BraTS20_Training_001_seg.nii
    │   └── ...
    ├── BraTS2021/
    │   └── ...
    └── BraTS_validationData/
        └── ...
```

> ⚠️ BraTS 数据集需在 Synapse.org 注册账号后下载，约 50GB+。本项目不包含数据集文件。

### 3. 配置环境

```bash
cd backend

# 复制环境变量文件并修改
cp .env.example .env
# 编辑 .env，修改 SECRET_KEY 等配置

# 安装 Python 依赖
pip install -r requires.txt
```

### 4. 下载预训练权重（可选）

本项目支持从 MONAI 下载预训练权重进行迁移学习：

```bash
python -m src.utils.pretrained --model brats_mri_segmentation --output checkpoints/
```

### 5. 训练模型

```bash
# 从预训练权重开始训练（推荐）
python -m src.train \
    --data_dir "dataset/BraTS2020" \
    --extra_data "dataset/BraTS2021" \
    --val_dir "dataset/BraTS_validationData" \
    --pretrained checkpoints/hg_mfnet_pretrained.pth \
    --epochs 100 \
    --patience 10 \
    --batch_size 1
```

#### 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_dir` | (必填) | 主训练数据目录 |
| `--extra_data` | [] | 额外训练数据目录（可多个） |
| `--val_dir` | None | 独立验证集，不指定则随机切分 5% |
| `--epochs` | 100 | 最大训练轮数 |
| `--lr` | 2e-4 | 初始学习率 |
| `--patience` | 10 | 早停耐心值 |
| `--batch_size` | 1 | 批次大小（8GB 显存推荐 1） |
| `--resume` | None | 断点恢复（`latest` 或指定 .pth 路径） |
| `--log_dir` | 自动生成 | 训练日志输出目录 |

#### 断点恢复

```bash
# Ctrl+C 中断 → 自动保存 interrupted.pth
# 崩溃 → 自动保存 crashed.pth
# 恢复训练
python -m src.train ... --resume logs/train_xxx/crashed.pth --log_dir logs/train_xxx
```

### 6. 评估模型

```bash
python -m src.eval \
    --data_dir "dataset/BraTS_validationData" \
    --checkpoint checkpoints/hg_mfnet_best.pth \
    --num_cases 62 \
    --save_results
```

### 7. 启动后端服务

```bash
python main.py --host 0.0.0.0 --port 8000
# API 文档: http://localhost:8000/docs
# 默认管理员: admin / admin123
```

### 8. 启动前端

```bash
cd frontend
npm install
npm run dev
# 访问 http://localhost:5173
```

---

## 模型权重

训练后的模型权重文件（.pth）约 320MB，超过 Git 限制，通过以下方式获取：

- **百度网盘**: [点击下载](https://pan.baidu.com/s/1-8LzEUyIt4zexgo_kYzoTg?pwd=1025)  提取码: `1025`
- **自行训练**: 按照上方训练步骤在 BraTS 数据集上训练（预估 ~34 小时，NVIDIA RTX 4060 8GB）

网盘内容说明：

| 文件 | 用途 |
|------|------|
| `hg_mfnet_best.pth` | 最佳模型权重（Mean Dice 0.7682），直接用于推理部署 |
| `hg_mfnet_pretrained.pth` | MONAI 预训练权重，用于从头/继续训练 |

最终部署使用的权重文件应放置于：

```
backend/checkpoints/hg_mfnet_best.pth
```

---

## 绘制训练曲线

```bash
cd backend
python plot_charts.py --log_dir logs/train_20260608_160648
```

生成两张图表：
- `training_curves_full.png` — Loss / WT Dice / 各类别 Dice / LR 四面板曲线
- `model_comparison.png` — HG-MFNet vs U-Net 3D vs V-Net 性能对比柱状图

---

## 技术栈

| 层级 | 技术 |
|------|------|
| 前端 | Vue 3 + Element Plus + ECharts + Axios |
| 后端 | Python + FastAPI + Uvicorn + ReportLab |
| 深度学习 | PyTorch 2.x + MONAI + NumPy + SciPy |
| 数据库 | SQLite (默认) / PostgreSQL + SQLAlchemy |
| 图像处理 | Nibabel + SimpleITK + Matplotlib |
| 认证 | JWT (HS256) + Bcrypt |

---

## 开发备忘

- **Windows 用户**: DataLoader `num_workers` 建议设为 0（避免 worker 崩溃）
- **显存不足**: 可降低 `image_size` 至 (64, 64, 64)（config.py → ModelConfig）
- **CPU 推理**: 训练脚本自动检测 GPU，推理可设置 `device='cpu'`
- **中文显示**: 绘图使用 SimHei 黑体，Linux 系统需手动安装中文字体

---

## 许可证

本项目仅用于学术研究与课程学习目的。BraTS 数据集版权归原作者及机构所有。

---

## 参考文献

1. Isensee, F., et al. "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation." *Nature Methods*, 2021.
2. Milletari, F., et al. "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation." *3DV*, 2016.
3. Menze, B.H., et al. "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)." *IEEE TMI*, 2015.
4. Bakas, S., et al. "Identifying the Best Machine Learning Algorithms for Brain Tumor Segmentation." *JACR*, 2018.
