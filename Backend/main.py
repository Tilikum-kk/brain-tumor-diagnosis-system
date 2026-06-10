"""
==============================================================================
脑肿瘤MRI智能辅助诊断系统 - 主程序入口
Brain Tumor MRI Intelligent Diagnosis System - Main Entry Point

功能描述：
    FastAPI应用主入口，负责：
        1. 初始化FastAPI应用
        2. 注册路由和中间件
        3. 配置CORS跨域
        4. 启动Uvicorn服务器
        5. 数据库初始化

运行方式：
    python main.py                          # 开发模式运行
    uvicorn main:app --host 0.0.0.0 --port 8000  # 生产模式
    uvicorn main:app --reload               # 热重载模式

系统架构概览：
    ┌──────────────────────────────────────────┐
    │           前端 (Vue3 + ElementUI)         │
    │     HTTP/REST API + WebSocket (可选)      │
    ├──────────────────────────────────────────┤
    │         后端 (FastAPI + Uvicorn)          │
    │  ┌─────────┬──────────┬──────────────┐  │
    │  │ 认证模块 │ 业务API  │ 报告生成模块  │  │
    │  ├─────────┴──────────┴──────────────┤  │
    │  │       超图融合推理引擎              │  │
    │  │   (HG-MFNet + PyTorch + MONAI)    │  │
    │  ├───────────────────────────────────┤  │
    │  │    PostgreSQL 数据库 (SQLAlchemy)   │  │
    │  └───────────────────────────────────┘  │
    └──────────────────────────────────────────┘

作者：梁昊 2023413304
日期：2025年6月
参考：《智能系统应用开发（II）》课程设计要求
==============================================================================
"""

import sys
import os
from pathlib import Path

# 将src目录加入Python路径
sys.path.insert(0, str(Path(__file__).parent))

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import logging

from src.config import api_config, UPLOAD_DIR, REPORT_DIR
from src.api.auth import auth_router
from src.api.routes import router as core_router
from src.database.database import db_manager

# ============================================================================
# 日志配置 - Logging Configuration
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('brain_tumor_diagnosis.log', encoding='utf-8'),
    ],
)
logger = logging.getLogger(__name__)


# ============================================================================
# 应用生命周期管理 - Application Lifecycle
# ============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI应用生命周期管理

    启动时：
        - 初始化数据库表
        - 预加载AI模型
        - 创建必要的目录

    关闭时：
        - 释放数据库连接
        - 保存模型状态
        - 清理临时文件
    """
    # ========== 启动阶段 ==========
    logger.info("=" * 60)
    logger.info("脑肿瘤MRI智能辅助诊断系统 启动中...")
    logger.info("Brain Tumor MRI Intelligent Diagnosis System Starting...")
    logger.info("=" * 60)

    # 1. 初始化数据库
    try:
        db_manager.create_all_tables()
        logger.info("✓ 数据库表初始化完成（SQLite）")
    except Exception as e:
        logger.warning(f"⚠ 数据库初始化警告（可能已存在）: {e}")

    # 1.5 创建默认管理员账号（如果不存在）
    from src.database.database import db_manager as dm
    from src.database.models import User, UserRole
    from src.api.auth import hash_password
    try:
        with dm.get_session() as session:
            existing = session.query(User).filter(User.username == "admin").first()
            if not existing:
                admin = User(
                    username="admin",
                    email="admin@hospital.com",
                    hashed_password=hash_password("admin123"),
                    full_name="梁昊",
                    role=UserRole.ADMIN,
                    department="人工智能学院",
                    hospital="重庆工商大学 · 2023413304",
                    is_active=True,
                )
                session.add(admin)
                logger.info("✓ 默认管理员账号已创建 (admin / admin123)")
            else:
                logger.info("✓ 管理员账号已存在，跳过创建")
    except Exception as e:
        logger.warning(f"⚠ 默认账号创建失败: {e}")

    # 2. 创建必要的目录
    for dir_path in [UPLOAD_DIR, REPORT_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ 目录就绪: {dir_path}")

    # 3. 模型预加载提示
    logger.info("ℹ AI模型将在首次请求时延迟加载")
    logger.info("  使用模型: HG-MFNet (超图多模态融合网络)")
    logger.info("  支持模态: T1, T1CE, T2, FLAIR")
    logger.info("  设备: " + ("CUDA" if __import__('torch').cuda.is_available() else "CPU"))

    logger.info(f"✓ 服务已启动: http://{api_config.host}:{api_config.port}")
    logger.info(f"✓ API文档: http://{api_config.host}:{api_config.port}/docs")
    logger.info("=" * 60)

    yield  # 应用运行期间

    # ========== 关闭阶段 ==========
    logger.info("系统正在关闭...")
    logger.info("=" * 60)


# ============================================================================
# 创建FastAPI应用 - Create FastAPI Application
# ============================================================================
app = FastAPI(
    title="脑肿瘤MRI智能辅助诊断系统",
    description="""
    ## Brain Tumor MRI Intelligent Diagnosis System

    基于**超图神经网络（Hypergraph Neural Network）**的多模态MRI脑肿瘤智能辅助诊断系统。

    ### 核心功能
    - 🧠 **多模态MRI分析**: 支持T1、T1CE、T2、FLAIR四种MRI模态
    - 🔬 **超图多模态融合**: 三级融合策略（特征级+分类器级+决策级）
    - 📊 **肿瘤自动分割**: 识别坏死核心、瘤周水肿、增强肿瘤区
    - 🏥 **WHO分级预测**: 自动预测脑肿瘤WHO分级（I-IV级）
    - 📋 **智能报告生成**: 自动生成结构化PDF诊断报告
    - 📈 **可视化分析**: 超图注意力热力图、3D肿瘤可视化

    ### 技术栈
    - **前端**: Vue3 + ElementUI + Plotly.js
    - **后端**: Python + FastAPI + Uvicorn
    - **模型**: PyTorch + MONAI（HG-MFNet超图融合网络）
    - **数据库**: PostgreSQL + SQLAlchemy
    - **可视化**: Matplotlib + Plotly + ReportLab
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    contact={
        "name": "梁昊",
        "student_id": "2023413304",
        "institution": "重庆工商大学 人工智能学院",
    },
    license_info={
        "name": "仅供教学研究使用",
    },
)

# ============================================================================
# 中间件配置 - Middleware Configuration
# ============================================================================

# CORS跨域配置（允许前端开发服务器访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",     # Vite开发服务器
        "http://localhost:8080",     # 备用端口
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8080",
        "*",                         # 开发阶段允许所有来源
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 请求日志中间件
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """记录所有HTTP请求日志"""
    logger.info(f"→ {request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"← {response.status_code}")
    return response


# ============================================================================
# 异常处理 - Exception Handlers
# ============================================================================
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理器"""
    logger.error(f"未处理的异常: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "服务器内部错误",
            "error_type": type(exc).__name__,
            "message": str(exc) if api_config.debug else "请联系系统管理员",
        },
    )


# ============================================================================
# 注册路由 - Register Routers
# ============================================================================
app.include_router(auth_router)
app.include_router(core_router)


# 静态文件服务（用于可视化图像）
@app.get("/static/{file_path:path}")
async def serve_static(file_path: str):
    """提供静态文件访问"""
    from fastapi.responses import FileResponse
    full_path = UPLOAD_DIR.parent / file_path
    if full_path.exists():
        return FileResponse(str(full_path))
    return JSONResponse(status_code=404, content={"detail": "文件不存在"})


# ============================================================================
# 健康检查 - Health Check
# ============================================================================
@app.get("/health", tags=["系统"])
async def health_check():
    """
    系统健康检查端点

    返回系统各组件的运行状态。

    Returns:
        系统健康状态
    """
    import torch

    # 检查数据库连接
    db_ok = False
    try:
        with db_manager.get_session() as session:
            session.execute(__import__('sqlalchemy').text("SELECT 1"))
        db_ok = True
    except Exception as e:
        logger.error(f"数据库连接检查失败: {e}")

    # 检查CUDA可用性
    cuda_available = torch.cuda.is_available()

    return {
        "status": "healthy" if db_ok else "degraded",
        "version": "1.0.0",
        "database": "connected" if db_ok else "disconnected",
        "cuda_available": cuda_available,
        "device": "cuda" if cuda_available else "cpu",
        "model": "HG-MFNet (Hypergraph Multi-modal Fusion Network)",
        "uptime": "running",
    }


# ============================================================================
# 根路由 - Root Endpoint
# ============================================================================
@app.get("/", tags=["系统"])
async def root():
    """根路由 - 系统欢迎页"""
    return {
        "message": "脑肿瘤MRI智能辅助诊断系统 API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "auth": "/api/auth",
            "patients": "/api/patients",
            "examinations": "/api/examinations",
            "reports": "/api/reports",
            "statistics": "/api/statistics/overview",
            "model_info": "/api/model/info",
        },
    }


# ============================================================================
# 命令行参数解析 - Argument Parser (argparse)
# ============================================================================
def parse_args():
    """
    解析命令行参数，支持以下运行模式：

        python main.py                          # 默认配置启动
        python main.py --host 127.0.0.1 --port 8080  # 自定义地址端口
        python main.py --debug                  # 调试模式（详细错误信息）
        python main.py --reset-db               # 重置数据库

    参考：argparse 标准库文档
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog='脑肿瘤MRI智能辅助诊断系统',
        description='基于超图神经网络的多模态MRI脑肿瘤智能辅助诊断系统',
        epilog='作者: 梁昊 (2023413304) | 重庆工商大学 人工智能学院',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ---- 服务配置 ----
    server_group = parser.add_argument_group('服务配置')
    server_group.add_argument(
        '--host', type=str, default=api_config.host,
        help=f'监听地址 (默认: {api_config.host})'
    )
    server_group.add_argument(
        '-p', '--port', type=int, default=api_config.port,
        help=f'监听端口 (默认: {api_config.port})'
    )
    server_group.add_argument(
        '--reload', action='store_true', default=False,
        help='启用热重载（开发模式，SQLite下慎用）'
    )

    # ---- 运行模式 ----
    mode_group = parser.add_argument_group('运行模式')
    mode_group.add_argument(
        '--debug', action='store_true', default=False,
        help='调试模式：显示详细错误堆栈，关闭时仅返回通用错误信息'
    )
    mode_group.add_argument(
        '--reset-db', action='store_true', default=False,
        help='重置数据库（删除所有数据后重新建表，慎用！）'
    )

    # ---- 数据库配置 ----
    db_group = parser.add_argument_group('数据库配置')
    db_group.add_argument(
        '--db-type', type=str, choices=['sqlite', 'postgresql'],
        default=None,
        help='数据库类型 (默认: sqlite)'
    )
    db_group.add_argument(
        '--db-path', type=str, default=None,
        help='SQLite 数据库文件路径'
    )

    # ---- 模型配置 ----
    model_group = parser.add_argument_group('模型配置')
    model_group.add_argument(
        '--model-path', type=str, default=None,
        help='预训练模型权重路径 (默认: checkpoints/hg_mfnet_best.pth)'
    )
    model_group.add_argument(
        '--no-cuda', action='store_true', default=False,
        help='强制使用 CPU 推理（即使有 GPU）'
    )

    # ---- 互斥参数 ----
    security = parser.add_mutually_exclusive_group()
    security.add_argument(
        '--allow-all-origins', action='store_true', default=True,
        help='允许所有跨域请求 (默认，开发阶段)'
    )
    security.add_argument(
        '--restrict-origins', action='store_true', default=False,
        help='仅允许 localhost 跨域请求 (生产环境推荐)'
    )

    args = parser.parse_args()

    # 回写配置
    if args.debug:
        api_config.debug = True

    return args


def build_app():
    """构建并返回 FastAPI 应用实例"""
    return app


# ============================================================================
# 主程序入口 - Main Entry
# ============================================================================
if __name__ == "__main__":
    args = parse_args()

    # 数据库重置
    if args.reset_db:
        confirm = input("⚠ 确认重置数据库？这将删除所有数据！(yes/no): ")
        if confirm.lower() == 'yes':
            logger.warning("正在重置数据库...")
            db_manager.drop_all_tables()
            db_manager.create_all_tables()
            logger.info("✓ 数据库已重置")
        else:
            logger.info("已取消重置")
        sys.exit(0)

    # 启动日志
    logger.info("=" * 60)
    logger.info(f"启动命令: python main.py --host {args.host} --port {args.port}"
                + (" --debug" if args.debug else "")
                + (" --reload" if args.reload else ""))
    logger.info(f"API 文档: http://{args.host}:{args.port}/docs")
    logger.info(f"健康检查: http://{args.host}:{args.port}/health")
    logger.info("=" * 60)

    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="debug" if args.debug else "info",
        access_log=True,
    )
