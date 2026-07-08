"""
异步 CRUD 操作

每个函数第一个参数为 db: AsyncSession
"""

import json
import uuid

from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from server.db.models import Conversation, Message, User


# ============================================================
# User
# ============================================================

async def create_user(
    db: AsyncSession,
    username: str,
    hashed_password: str,
    email: str | None = None,
) -> dict:
    """创建用户，返回用户字典"""
    user = User(
        username=username,
        hashed_password=hashed_password,
        email=email,
    )
    db.add(user)
    await db.flush()
    return user.to_dict()


async def get_user_by_username(
    db: AsyncSession, username: str
) -> dict | None:
    """按用户名查找用户（不含密码哈希，用于 API 返回）"""
    result = await db.execute(
        select(User).filter(User.username == username)
    )
    user = result.scalar_one_or_none()
    return user.to_dict() if user else None


async def get_user_by_username_with_password(
    db: AsyncSession, username: str
) -> User | None:
    """按用户名查找用户（含密码哈希，仅用于登录验证）"""
    result = await db.execute(
        select(User).filter(User.username == username)
    )
    return result.scalar_one_or_none()


async def get_user_by_id(
    db: AsyncSession, user_id: int
) -> dict | None:
    """按 ID 查找用户"""
    user = await db.get(User, user_id)
    return user.to_dict() if user else None


# ============================================================
# Conversation
# ============================================================

async def create_conversation(
    db: AsyncSession,
    conversation_id: str = "",
    model: str = "gis-assistant",
    title: str = "",
    user_id: int | None = None,
) -> str:
    """创建对话，返回 conversation_id"""
    cid = conversation_id or uuid.uuid4().hex[:16]
    conv = Conversation(id=cid, title=title, model=model, user_id=user_id)
    db.add(conv)
    await db.flush()
    return cid


async def get_conversation(
    db: AsyncSession, conversation_id: str
) -> dict | None:
    """获取单个对话"""
    conv = await db.get(Conversation, conversation_id)
    return conv.to_dict() if conv else None


async def list_conversations(
    db: AsyncSession,
    limit: int = 20,
    offset: int = 0,
    user_id: int | None = None,
) -> list[dict]:
    """分页获取对话列表（按更新时间倒序），可按用户过滤"""
    stmt = select(Conversation).order_by(desc(Conversation.updated_at))
    if user_id is not None:
        stmt = stmt.filter(Conversation.user_id == user_id)
    stmt = stmt.offset(offset).limit(limit)
    result = await db.execute(stmt)
    return [r.to_dict() for r in result.scalars().all()]


async def list_anonymous_conversations(
    db: AsyncSession,
    limit: int = 20,
    offset: int = 0,
) -> list[dict]:
    """分页获取匿名对话（user_id IS NULL），按更新时间倒序"""
    stmt = (
        select(Conversation)
        .filter(Conversation.user_id.is_(None))
        .order_by(desc(Conversation.updated_at))
        .offset(offset)
        .limit(limit)
    )
    result = await db.execute(stmt)
    return [r.to_dict() for r in result.scalars().all()]


async def update_conversation_title(
    db: AsyncSession, conversation_id: str, title: str
):
    """更新对话标题"""
    conv = await db.get(Conversation, conversation_id)
    if conv:
        conv.title = title


async def bump_message_count(
    db: AsyncSession, conversation_id: str, delta: int = 1
):
    """增加消息计数"""
    conv = await db.get(Conversation, conversation_id)
    if conv:
        conv.message_count += delta


async def delete_conversation(db: AsyncSession, conversation_id: str):
    """删除对话及其所有消息（CASCADE）"""
    conv = await db.get(Conversation, conversation_id)
    if conv:
        await db.delete(conv)
        await db.flush()


# ============================================================
# Message
# ============================================================

async def save_message(
    db: AsyncSession,
    conversation_id: str,
    role: str,
    content: str = "",
    tool_calls: list[dict] | None = None,
    tool_name: str | None = None,
    token_count: int = 0,
):
    """保存一条消息"""
    tc_json = json.dumps(tool_calls, ensure_ascii=False) if tool_calls else None
    msg = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        tool_calls=tc_json,
        tool_name=tool_name,
        token_count=token_count,
    )
    db.add(msg)


async def get_messages(
    db: AsyncSession, conversation_id: str
) -> list[dict]:
    """获取某个对话的全部消息（按 id 正序）"""
    result = await db.execute(
        select(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.id.asc())
    )
    return [r.to_dict() for r in result.scalars().all()]
