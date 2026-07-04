"""
OpenAI 兼容路由

- POST /v1/chat/completions  聊天接口（支持 stream）
- GET  /v1/models             模型列表
"""

import json

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from server.schemas.openai import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChunk,
    DeltaMessage,
    ModelList,
    make_response,
    make_chunk,
    make_chat_id,
)

router = APIRouter()


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
async def chat_completions(request: ChatCompletionRequest, req: Request):
    """OpenAI 兼容的聊天接口，支持多轮对话"""
    agent = req.app.state.agent
    model = request.model or "gis-assistant"

    if request.stream:
        return _stream_response(agent, request.messages, model)
    else:
        return _non_stream_response(agent, request.messages, model)


# ============================================================
# 非流式响应
# ============================================================

def _non_stream_response(agent, messages, model: str) -> ChatCompletionResponse:
    """运行 agent，逐字收集内容，返回 OpenAI ChatCompletion 格式"""
    from agent.graph import run_agent_stream

    parts: list[str] = []

    for event in run_agent_stream(agent, messages):
        if event["msg_type"] == "ai" and event.get("delta"):
            parts.append(event["content"])

    answer = "".join(parts)
    if not answer:
        answer = "未能生成回答。"

    return make_response(model, answer)


# ============================================================
# 流式响应
# ============================================================

def _stream_response(agent, messages, model: str):
    """SSE 逐字流式生成器，每个 token 作为 OpenAI chunk 发送"""
    from agent.graph import run_agent_stream

    chunk_id = make_chat_id()
    role_sent = False

    async def generate():
        nonlocal role_sent

        for event in run_agent_stream(agent, messages):
            if event["msg_type"] == "ai":
                is_delta = event.get("delta", False)
                has_tools = bool(event.get("tool_calls"))

                if not role_sent:
                    # 首个 chunk: 发送 role
                    yield f"data: {make_chunk(model, DeltaMessage(role='assistant'), chunk_id=chunk_id).model_dump_json()}\n\n"
                    role_sent = True

                if is_delta and not has_tools:
                    # 逐字增量
                    content = event["content"] or ""
                    if content:
                        yield f"data: {make_chunk(model, DeltaMessage(content=content), chunk_id=chunk_id).model_dump_json()}\n\n"

        # finish_reason
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
