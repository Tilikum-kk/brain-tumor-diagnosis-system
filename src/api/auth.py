"""
==============================================================================
脑肿瘤MRI智能辅助诊断系统 - 认证与授权
Brain Tumor MRI Intelligent Diagnosis System - Authentication & Authorization

功能描述：
    实现基于JWT的用户认证和角色权限控制。
    支持：
        - 用户注册和登录
        - JWT Token生成与验证
        - 基于角色的访问控制（RBAC）
        - Token刷新机制

安全措施：
    - 密码使用bcrypt哈希
    - JWT使用HS256签名
    - Token过期自动刷新
==============================================================================
"""

from datetime import datetime, timedelta
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, validator
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from ..config import api_config
from ..database.database import db_manager
from ..database.models import User, UserRole

# ============================================================================
# 路由定义
# ============================================================================
auth_router = APIRouter(prefix="/api/auth", tags=["认证"])

# ============================================================================
# 密码加密和Token配置
# ============================================================================
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

SECRET_KEY = api_config.secret_key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = api_config.access_token_expire_minutes


# ============================================================================
# Pydantic模型 - 请求/响应 Schema
# ============================================================================
class UserRegister(BaseModel):
    """用户注册请求"""
    username: str
    email: EmailStr
    password: str
    full_name: str
    role: UserRole = UserRole.DOCTOR
    department: Optional[str] = None
    hospital: Optional[str] = None

    @validator('password')
    def password_strength(cls, v: str) -> str:
        """验证密码强度"""
        if len(v) < 6:
            raise ValueError('密码长度不能少于6位')
        return v


class UserLogin(BaseModel):
    """用户登录请求"""
    username: str
    password: str


class UserResponse(BaseModel):
    """用户信息响应"""
    id: int
    username: str
    email: str
    full_name: str
    role: UserRole
    department: Optional[str]
    hospital: Optional[str]

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    """Token响应"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse


class TokenData(BaseModel):
    """Token载荷数据"""
    user_id: Optional[int] = None
    username: Optional[str] = None
    role: Optional[str] = None


# ============================================================================
# 辅助函数
# ============================================================================
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证密码"""
    return pwd_context.verify(plain_password, hashed_password)


def hash_password(password: str) -> str:
    """哈希密码"""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    创建JWT访问令牌

    Args:
        data: 令牌载荷数据
        expires_delta: 过期时间增量

    Returns:
        JWT令牌字符串
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(
    token: str = Depends(oauth2_scheme)
) -> User:
    """
    从JWT令牌获取当前用户

    用作FastAPI依赖注入。

    Args:
        token: JWT令牌

    Returns:
        当前用户对象

    Raises:
        HTTPException: 认证失败时抛出401
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="无效的认证凭证",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: int = payload.get("user_id")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    with db_manager.get_session() as session:
        user = session.query(User).filter(User.id == user_id).first()
        if user is None:
            raise credentials_exception
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="用户账户已被禁用"
            )
        # 从Session中分离，避免后续访问属性时报DetachedInstanceError
        session.expunge(user)
        return user


async def get_current_active_doctor(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    验证当前用户为医生角色

    Args:
        current_user: 当前用户

    Returns:
        当前医生用户
    """
    if current_user.role not in [UserRole.DOCTOR, UserRole.ADMIN]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="只有医生和管理员可以执行此操作"
        )
    return current_user


async def get_current_admin(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    验证当前用户为管理员角色

    Args:
        current_user: 当前用户

    Returns:
        当前管理员用户
    """
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="只有管理员可以执行此操作"
        )
    return current_user


# ============================================================================
# API端点
# ============================================================================
@auth_router.post("/register", response_model=TokenResponse)
async def register(user_data: UserRegister):
    """
    用户注册接口

    创建新用户账户并返回JWT令牌。

    Args:
        user_data: 注册信息

    Returns:
        TokenResponse: 包含JWT令牌和用户信息
    """
    with db_manager.get_session() as session:
        # 检查用户名是否已存在
        existing_user = session.query(User).filter(
            User.username == user_data.username
        ).first()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="用户名已被注册"
            )

        # 检查邮箱是否已存在
        existing_email = session.query(User).filter(
            User.email == user_data.email
        ).first()
        if existing_email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="邮箱已被注册"
            )

        # 创建新用户
        new_user = User(
            username=user_data.username,
            email=user_data.email,
            hashed_password=hash_password(user_data.password),
            full_name=user_data.full_name,
            role=user_data.role,
            department=user_data.department,
            hospital=user_data.hospital,
        )
        session.add(new_user)
        session.flush()

        # 生成令牌
        access_token = create_access_token(
            data={"user_id": new_user.id, "username": new_user.username,
                  "role": new_user.role.value}
        )

        return TokenResponse(
            access_token=access_token,
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user=UserResponse.model_validate(new_user),
        )


@auth_router.post("/login", response_model=TokenResponse)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    用户登录接口

    验证用户名密码，返回JWT令牌。

    Args:
        form_data: OAuth2登录表单

    Returns:
        TokenResponse: 包含JWT令牌和用户信息
    """
    with db_manager.get_session() as session:
        # 查找用户
        user = session.query(User).filter(
            User.username == form_data.username
        ).first()

        if not user or not verify_password(form_data.password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="用户名或密码错误",
                headers={"WWW-Authenticate": "Bearer"},
            )

        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="用户账户已被禁用"
            )

        # 更新最后登录时间
        user.last_login = datetime.utcnow()
        session.flush()

        # 生成令牌
        access_token = create_access_token(
            data={
                "user_id": user.id,
                "username": user.username,
                "role": user.role.value,
            }
        )

        return TokenResponse(
            access_token=access_token,
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user=UserResponse.model_validate(user),
        )


@auth_router.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    """
    获取当前用户信息

    需要有效的JWT令牌。

    Args:
        current_user: 当前认证用户

    Returns:
        UserResponse: 用户信息
    """
    return UserResponse.model_validate(current_user)


@auth_router.post("/refresh")
async def refresh_token(current_user: User = Depends(get_current_user)):
    """
    刷新JWT令牌

    Args:
        current_user: 当前认证用户

    Returns:
        新的令牌信息
    """
    access_token = create_access_token(
        data={
            "user_id": current_user.id,
            "username": current_user.username,
            "role": current_user.role.value,
        }
    )
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    }
