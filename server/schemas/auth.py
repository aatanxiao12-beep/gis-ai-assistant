"""
认证相关 Pydantic 模型
"""

from pydantic import BaseModel, Field


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=2, max_length=50, description="用户名")
    password: str = Field(..., min_length=6, max_length=100, description="密码")
    captcha_id: str = Field(..., min_length=1, description="验证码 ID")
    captcha_text: str = Field(..., min_length=4, max_length=6, description="验证码答案")


class LoginRequest(BaseModel):
    username: str = Field(..., description="用户名")
    password: str = Field(..., description="密码")


class CaptchaResponse(BaseModel):
    captcha_id: str
    captcha_image: str  # base64 PNG


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: int
    username: str


class UserResponse(BaseModel):
    id: int
    username: str
    email: str | None = None
    is_active: bool
    created_at: str
