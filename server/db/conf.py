"""
数据库引擎 + 会话工厂 + FastAPI 依赖项

用法:
    from server.db.conf import get_database, async_engine, AsyncSessionLocal, Base
"""

import os

from sqlalchemy.ext.asyncio import (
    create_async_engine,
    async_sessionmaker,
    AsyncSession,
)
from sqlalchemy.orm import DeclarativeBase

# ── 数据库文件路径：server/data/chat.db ──────────────────────

DB_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DB_PATH = os.path.join(DB_DIR, "chat.db")
DATABASE_URL = f"sqlite+aiosqlite:///{DB_PATH}"

os.makedirs(DB_DIR, exist_ok=True)

# ── 异步引擎 ─────────────────────────────────────────────────

async_engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False},
)

# ── 会话工厂 ─────────────────────────────────────────────────

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# ── 基类 ────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ── 依赖项 ──────────────────────────────────────────────────

async def get_database():
    """FastAPI 依赖：为每个请求创建异步数据库会话"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ── 建表 ────────────────────────────────────────────────────

async def init_db():
    """应用启动时调用，幂等"""
    import server.db.models  # noqa: F401 — 注册 ORM 模型
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
