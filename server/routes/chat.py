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
# 消息提取
# ============================================================

def _get_attr(m, key: str, default=""):
    """兼容 Pydantic 模型和 dict 的字段读取"""
    if hasattr(m, key):
        return getattr(m, key)
    if isinstance(m, dict):
        return m.get(key, default)
    return default


def _extract_question(messages: list) -> str:
    """从 OpenAI messages 中提取最后一个 user 消息内容"""
    user_msgs = [m for m in messages if _get_attr(m, "role") == "user"]
    if not user_msgs:
        user_msgs = list(messages)
    if user_msgs:
        return _get_attr(user_msgs[-1], "content")
    return ""


def _extract_system_prompt(messages: list) -> str | None:
    """提取 system / developer 消息内容（用于覆盖默认提示词）"""
    for m in messages:
        if _get_attr(m, "role") in ("system", "developer"):
            content = _get_attr(m, "content")
            if content:
                return content
    return None


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
    """OpenAI 兼容的聊天接口"""
    agent = req.app.state.agent
    from agent.graph import run_agent_stream

    # 将 OpenAI messages 转成 question string
    question = _extract_question(request.messages)
    model = request.model or "gis-assistant"

    if request.stream:
        return _stream_response(agent, question, model)
    else:
        return _non_stream_response(agent, question, model)


# ============================================================
# 非流式响应
# ============================================================

def _non_stream_response(agent, question: str, model: str) -> ChatCompletionResponse:
    """运行 agent，收集最终回答，返回 OpenAI ChatCompletion 格式"""
    from agent.graph import run_agent_stream

    answer = ""

    for event in run_agent_stream(agent, question):
        # 取最后一个 ai 消息（不含 tool_calls）作为最终回答
        if event["msg_type"] == "ai" and not event.get("tool_calls"):
            answer = event["content"]

    if not answer:
        answer = "未能生成回答。"

    return make_response(model, answer)


# ============================================================
# 流式响应
# ============================================================

def _stream_response(agent, question: str, model: str):
    """SSE 流式生成器，按 agent 步骤粒度输出 OpenAI 格式 chunks"""
    from agent.graph import run_agent_stream

    chunk_id = make_chat_id()
    role_sent = False

    async def generate():
        nonlocal role_sent

        for event in run_agent_stream(agent, question):
            if event["msg_type"] == "ai":
                content = event["content"] or ""
                has_tools = bool(event.get("tool_calls"))

                if not role_sent:
                    # 首个 chunk: 仅发送 role（不含 content）
                    yield f"data: {make_chunk(model, DeltaMessage(role='assistant'), chunk_id=chunk_id).model_dump_json()}\n\n"
                    role_sent = True

                # 有内容的 ai 消息：发送 content
                if content and not has_tools:
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
