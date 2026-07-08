"""
SQLAlchemy 2.0 ORM 模型
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, Index, Integer, String, Text, ForeignKey, DateTime
from sqlalchemy.orm import Mapped, mapped_column, relationship

from server.db.conf import Base


class User(Base):
    """用户表"""
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True, comment="用户ID"
    )
    username: Mapped[str] = mapped_column(
        String(50), unique=True, nullable=False, comment="用户名"
    )
    hashed_password: Mapped[str] = mapped_column(
        String(200), nullable=False, comment="bcrypt 哈希密码"
    )
    email: Mapped[Optional[str]] = mapped_column(
        String(100), nullable=True, comment="邮箱"
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean, default=True, comment="是否激活"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, comment="注册时间"
    )

    conversations: Mapped[list["Conversation"]] = relationship(
        "Conversation", back_populates="user"
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "is_active": self.is_active,
            "created_at": _fmt(self.created_at),
        }

    def __repr__(self) -> str:
        return f"<User(id={self.id}, username={self.username!r})>"


class Conversation(Base):
    """对话表"""
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, comment="对话ID"
    )
    title: Mapped[str] = mapped_column(
        String(200), default="", comment="对话标题（取第一个用户问题）"
    )
    model: Mapped[str] = mapped_column(
        String(50), default="gis-assistant", comment="模型名"
    )
    message_count: Mapped[int] = mapped_column(
        Integer, default=0, comment="消息计数"
    )
    user_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=True, comment="所属用户ID（空为匿名）"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, comment="创建时间"
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, onupdate=datetime.now, comment="更新时间"
    )

    messages: Mapped[list["Message"]] = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    user: Mapped[Optional["User"]] = relationship(
        "User", back_populates="conversations"
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "model": self.model,
            "message_count": self.message_count,
            "user_id": self.user_id,
            "created_at": _fmt(self.created_at),
            "updated_at": _fmt(self.updated_at),
        }

    def __repr__(self) -> str:
        return f"<Conversation(id={self.id!r}, title={self.title!r})>"


class Message(Base):
    """消息表"""
    __tablename__ = "messages"
    __table_args__ = (
        Index("idx_messages_conv", "conversation_id", "id"),
    )

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True, comment="消息ID"
    )
    conversation_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        comment="所属对话ID",
    )
    role: Mapped[str] = mapped_column(
        String(20), nullable=False, comment="角色: user/assistant/system/tool"
    )
    content: Mapped[str] = mapped_column(
        Text, default="", comment="消息内容"
    )
    tool_calls: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True, comment="工具调用 JSON"
    )
    tool_name: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True, comment="工具名称（仅 tool 消息）"
    )
    token_count: Mapped[int] = mapped_column(
        Integer, default=0, comment="token 计数"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, comment="创建时间"
    )

    conversation: Mapped["Conversation"] = relationship(
        "Conversation", back_populates="messages"
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "role": self.role,
            "content": self.content,
            "tool_calls": self.tool_calls,
            "tool_name": self.tool_name,
            "token_count": self.token_count,
            "created_at": _fmt(self.created_at),
        }

    def __repr__(self) -> str:
        return f"<Message(id={self.id}, role={self.role!r})>"


def _fmt(dt: datetime | None) -> str:
    """将 datetime 转为与之前一致的字符串格式"""
    return dt.strftime("%Y-%m-%d %H:%M:%S") if dt else ""
