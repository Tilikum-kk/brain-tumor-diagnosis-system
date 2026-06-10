"""脑肿瘤MRI智能辅助诊断系统 - API包"""

from .routes import router
from .auth import auth_router

__all__ = ['router', 'auth_router']
