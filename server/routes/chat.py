"""
OpenAI 兼容路由

- POST /v1/chat/completions  聊天接口（支持 stream）
- GET  /v1/models             模型列表
- GET  /v1/conversations      对话列表
- GET  /v1/conversations/{id} 对话详情
- DELETE /v1/conversations/{id} 删除对话
"""

from fastapi import APIRouter, Request, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from server.schemas.openai import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    DeltaMessage,
    ModelList,
    make_response,
    make_chunk,
    make_chat_id,
)
from server.db.conf import get_database
from server.db import crud

router = APIRouter()


# ============================================================
# 持久化辅助函数
# ============================================================

async def _resolve_conversation(
    db: AsyncSession, conversation_id: str | None, model: str
) -> tuple[str, int]:
    """
    解析会话：有 conversation_id 且存在则复用，否则新建。
    返回 (conv_id, 已存在的消息数)。

    注意：无 conversation_id 时不创建对话（如 NextChat 标题生成等
    非对话请求不应产生游离记录）。
    """
    if not conversation_id:
        return "", 0

    conv = await crud.get_conversation(db, conversation_id)
    if conv:
        return conv["id"], conv["message_count"]

    # 新建对话，使用客户端提供的 ID
    cid = await crud.create_conversation(db, conversation_id=conversation_id, model=model)
    return cid, 0


async def _save_new_messages(
    db: AsyncSession, conv_id: str, messages
) -> int:
    """
    保存请求中尚未持久化的 user 消息。

    策略：从消息列表尾部向前查找最后一个 user 消息（即触发本次
    请求的新消息），若与 DB 末尾不同则保存。不依赖 skip_count，
    避免因前后端消息列表错位导致消息遗漏或重复。
    system 角色消息不存入 DB。
    """
    # 从尾部找到最后一个 user 消息
    new_content = None
    for m in reversed(messages):
        role = m.role if hasattr(m, "role") else m.get("role", "user")
        if role == "system":
            continue
        content = m.content if hasattr(m, "content") else m.get("content", "")
        if role == "user" and content:
            new_content = content
            break

    if new_content is None:
        return 0

    # 重试保护：如果 DB 最后一条就是相同的 user 消息，跳过
    existing = await crud.get_messages(db, conv_id)
    if existing:
        last = existing[-1]
        if last["role"] == "user" and last["content"] == new_content:
            return 0

    await crud.save_message(db, conv_id, "user", new_content)
    return 1


async def _set_title(db: AsyncSession, conv_id: str, messages) -> str:
    """取第一个 user 消息的前50字符作为对话标题（仅新建会话时调用）"""
    for m in messages:
        role = m.role if hasattr(m, "role") else m.get("role", "user")
        if role == "user":
            content = m.content if hasattr(m, "content") else m.get("content", "")
            title = content[:50].replace("\n", " ") if content else ""
            if title:
                await crud.update_conversation_title(db, conv_id, title)
            return title
    return ""


# ============================================================
# 模型列表
# ============================================================

@router.get("/v1/models")
async def list_models():
    return ModelList()


# ============================================================
# 聊天接口
# ============================================================

@router.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    req: Request,
    db: AsyncSession = Depends(get_database),
):
    """OpenAI 兼容聊天接口，自动持久化到 SQLite。

    请求体可携带 conversation_id 复用已有会话，避免重复存储历史消息。
    """
    agent = req.app.state.agent
    model = request.model or "gis-assistant"

    # 解析会话：有 conversation_id 则复用，否则新建
    conv_id, existing_count = await _resolve_conversation(
        db, request.conversation_id, model
    )

    # 无 conversation_id 的请求不持久化（如 NextChat 标题生成等）
    if not conv_id:
        if request.stream:
            return await _stream_response(agent, request.messages, model, None, "", 0)
        else:
            return await _non_stream_response(agent, request.messages, model, None, "", 0)

    is_new = existing_count == 0

    # 仅保存新增的消息（基于 DB 去重，不依赖 skip_count）
    msg_count = await _save_new_messages(db, conv_id, request.messages)

    # 新建会话时设置标题
    if is_new:
        await _set_title(db, conv_id, request.messages)

    if request.stream:
        return await _stream_response(agent, request.messages, model, db,
                                      conv_id, msg_count)
    else:
        return await _non_stream_response(agent, request.messages, model, db,
                                          conv_id, msg_count)


# ============================================================
# 非流式响应
# ============================================================

async def _non_stream_response(
    agent, messages, model: str, db: AsyncSession,
    conv_id: str, msg_count: int,
) -> ChatCompletionResponse:
    """运行 agent，收集完整回答，持久化到 DB"""
    from agent.graph import run_agent_stream

    parts: list[str] = []
    persist = bool(conv_id)

    # 运行 agent，收集增量内容和工具调用
    for event in run_agent_stream(agent, messages):
        if event["msg_type"] == "ai":
            if event.get("tool_calls"):
                if persist:
                    await crud.save_message(
                        db, conv_id, "assistant", "",
                        tool_calls=event["tool_calls"],
                    )
                    msg_count += 1
            elif event.get("delta"):
                parts.append(event["content"])

        elif event["msg_type"] == "tool":
            if persist:
                await crud.save_message(
                    db, conv_id, "tool", event["content"],
                    tool_name=event.get("tool_name"),
                )
                msg_count += 1

    # 保存最终回答
    answer = "".join(parts) or "未能生成回答。"
    if persist:
        await crud.save_message(db, conv_id, "assistant", answer)
        msg_count += 1
        await crud.bump_message_count(db, conv_id, msg_count)
    return make_response(model, answer, conversation_id=conv_id or None)


# ============================================================
# 流式响应
# ============================================================

async def _stream_response(
    agent, messages, model: str, db: AsyncSession,
    conv_id: str, msg_count: int,
):
    """SSE 流式响应：逐 token 推送 OpenAI chunk，同时写 DB"""
    from agent.graph import run_agent_stream

    chunk_id = make_chat_id()
    role_sent = False
    answer_parts: list[str] = []   # 收集完整文本，流结束后落库
    persist = bool(conv_id)

    async def generate():
        nonlocal role_sent, msg_count

        for event in run_agent_stream(agent, messages):
            if event["msg_type"] == "ai":
                is_delta = event.get("delta", False)
                has_tools = bool(event.get("tool_calls"))

                # 首个 chunk 必须带 role
                if not role_sent:
                    yield (
                        f"data: {make_chunk(model, DeltaMessage(role='assistant'), chunk_id=chunk_id).model_dump_json()}\n\n"
                    )
                    role_sent = True

                if has_tools:
                    if persist:
                        await crud.save_message(
                            db, conv_id, "assistant", "",
                            tool_calls=event["tool_calls"],
                        )
                        msg_count += 1

                elif is_delta:
                    # 逐字增量 → 推 SSE + 累积
                    content = event["content"] or ""
                    if content:
                        answer_parts.append(content)
                        yield (
                            f"data: {make_chunk(model, DeltaMessage(content=content), chunk_id=chunk_id).model_dump_json()}\n\n"
                        )

            elif event["msg_type"] == "tool":
                if persist:
                    await crud.save_message(
                        db, conv_id, "tool", event["content"],
                        tool_name=event.get("tool_name"),
                    )
                    msg_count += 1

        # 流结束 → 落库最终回答 + 更新计数 + 发送 finish_reason
        answer = "".join(answer_parts)
        if persist and answer:
            await crud.save_message(db, conv_id, "assistant", answer)
            msg_count += 1

        if persist:
            await crud.bump_message_count(db, conv_id, msg_count)

        final = make_chunk(model, DeltaMessage(), finish_reason="stop", chunk_id=chunk_id)
        yield f"data: {final.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================
# 对话历史 API
# ============================================================

@router.get("/v1/conversations")
async def list_conversations(
    limit: int = 20,
    offset: int = 0,
    db: AsyncSession = Depends(get_database),
):
    """对话列表，按更新时间倒序，支持分页"""
    conversations = await crud.list_conversations(db, limit, offset)
    return {"object": "list", "data": conversations}


@router.get("/v1/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    db: AsyncSession = Depends(get_database),
):
    """获取单个对话详情，包含全部消息"""
    conv = await crud.get_conversation(db, conversation_id)
    if not conv:
        return JSONResponse({"error": "conversation not found"}, status_code=404)
    messages = await crud.get_messages(db, conversation_id)
    conv["messages"] = messages
    return conv


@router.delete("/v1/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    db: AsyncSession = Depends(get_database),
):
    """删除对话及其所有消息"""
    await crud.delete_conversation(db, conversation_id)
    return {"status": "ok"}
