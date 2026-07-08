"""
FastAPI 应用入口

用法:
    uvicorn server.main:app --reload
    python run_server.py
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.config import CORS_ORIGINS
from server.routes import health, chat


# ── 生命周期 ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时初始化数据库 + 创建 agent，关闭时释放资源"""
    from server.db.conf import init_db
    await init_db()
    from agent import create_agent
    app.state.agent = create_agent()
    yield
    from server.db.conf import async_engine
    await async_engine.dispose()


# ── 应用 ──────────────────────────────────────────────────

app = FastAPI(
    title="GIS AI Standard Assistant",
    description="基于 RAG 的 GIS 标准智能问答助手",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS.split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(chat.router)


# ── 启动入口 ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    from server.config import HOST, PORT
    uvicorn.run("server.main:app", host=HOST, port=PORT, reload=True)
