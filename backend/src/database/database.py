"""
==============================================================================
脑肿瘤MRI智能辅助诊断系统 - 数据库连接管理
Brain Tumor MRI Intelligent Diagnosis System - Database Connection

功能描述：
    管理PostgreSQL数据库连接，提供会话管理和基础操作。
    使用SQLAlchemy ORM框架，支持连接池和事务管理。

类说明：
    DatabaseManager: 数据库管理器，负责引擎创建和会话管理
==============================================================================
"""

from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from sqlalchemy.pool import QueuePool
from typing import Generator
from contextlib import contextmanager
import logging

from ..config import db_config

# 配置日志
logger = logging.getLogger(__name__)

# SQLAlchemy声明式基类
Base = declarative_base()


class DatabaseManager:
    """
    数据库管理器类

    职责：
        1. 创建和管理数据库引擎
        2. 提供会话工厂
        3. 管理数据库表的创建和删除
        4. 提供上下文管理器用于安全的事务处理

    使用示例：
        db = DatabaseManager()
        with db.get_session() as session:
            user = session.query(User).filter_by(username="admin").first()
    """

    def __init__(self):
        """初始化数据库管理器，创建引擎和会话工厂"""
        self._engine: Engine = None
        self._session_factory: sessionmaker = None
        self._initialize_engine()

    def _initialize_engine(self) -> None:
        """
        初始化数据库引擎

        配置连接池参数：
            - pool_size: 连接池大小（默认20）
            - max_overflow: 最大溢出连接数（默认10）
            - pool_pre_ping: 连接前检测可用性
        """
        try:
            self._engine = create_engine(
                db_config.url,
                poolclass=QueuePool,
                pool_size=20,
                max_overflow=10,
                pool_pre_ping=True,          # 连接前检测有效性
                pool_recycle=3600,           # 1小时后回收连接
                echo=False,                  # 不打印SQL日志
            )
            self._session_factory = sessionmaker(
                bind=self._engine,
                autocommit=False,
                autoflush=False,
            )
            logger.info(f"数据库引擎初始化成功: {db_config.host}:{db_config.port}")
        except Exception as e:
            logger.error(f"数据库引擎初始化失败: {e}")
            raise

    @property
    def engine(self) -> Engine:
        """获取数据库引擎"""
        return self._engine

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        获取数据库会话的上下文管理器

        自动处理提交和回滚，确保资源正确释放。

        Yields:
            Session: SQLAlchemy数据库会话
        """
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"数据库操作异常，已回滚: {e}")
            raise
        finally:
            session.close()

    def create_all_tables(self) -> None:
        """创建所有数据库表"""
        Base.metadata.create_all(bind=self._engine)
        logger.info("所有数据库表创建完成")

    def drop_all_tables(self) -> None:
        """删除所有数据库表（谨慎使用）"""
        Base.metadata.drop_all(bind=self._engine)
        logger.warning("所有数据库表已删除")


# 全局数据库管理器实例
db_manager = DatabaseManager()
