"""
异步 CRUD 操作

每个函数第一个参数为 db: AsyncSession
"""

import json
import uuid

from sqlalchemy import select, desc, delete
from sqlalchemy.ext.asyncio import AsyncSession

from server.db.models import Conversation, Message


# ============================================================
# Conversation
# ============================================================

async def create_conversation(
    db: AsyncSession,
    conversation_id: str = "",
    model: str = "gis-assistant",
    title: str = "",
) -> str:
    """创建对话，返回 conversation_id"""
    cid = conversation_id or uuid.uuid4().hex[:16]
    conv = Conversation(id=cid, title=title, model=model)
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
    db: AsyncSession, limit: int = 20, offset: int = 0
) -> list[dict]:
    """分页获取对话列表（按更新时间倒序）"""
    result = await db.execute(
        select(Conversation)
        .order_by(desc(Conversation.updated_at))
        .offset(offset)
        .limit(limit)
    )
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
