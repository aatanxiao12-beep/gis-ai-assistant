"""
聊天路由 —— /chat 和 /chat/stream
"""

import json
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from server.schemas.chat import ChatRequest, ChatResponse, ToolCallInfo, StepInfo

router = APIRouter()


def _build_tool_call_info(tool_calls: list) -> list[ToolCallInfo]:
    result = []
    for tc in tool_calls:
        result.append(ToolCallInfo(
            name=tc.get("name", "?"),
            args=tc.get("args", {}),
        ))
    return result


def _build_step_info(event: dict) -> StepInfo:
    tc_info = _build_tool_call_info(event.get("tool_calls", []))
    return StepInfo(
        step=event["step"],
        msg_type=event["msg_type"],
        content=event.get("content", ""),
        tool_calls=tc_info,
        tool_name=event.get("tool_name", ""),
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, req: Request):
    """普通问答：收集全部事件，返回完整 JSON 响应"""
    agent = req.app.state.agent
    from agent.graph import run_agent_stream

    events: list[StepInfo] = []
    steps: list[str] = []
    answer = ""
    all_tool_calls: list[ToolCallInfo] = []

    for event in run_agent_stream(agent, request.question):
        si = _build_step_info(event)
        events.append(si)
        steps.append(event["step"])

        if event["msg_type"] == "ai" and not event.get("tool_calls"):
            answer = event["content"]
        elif event.get("tool_calls"):
            all_tool_calls.extend(_build_tool_call_info(event["tool_calls"]))

    return ChatResponse(
        answer=answer,
        steps=steps,
        tool_calls=all_tool_calls,
        events=events,
    )


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest, req: Request):
    """SSE 流式问答：每个事件实时推送"""
    agent = req.app.state.agent
    from agent.graph import run_agent_stream

    async def generate():
        for event in run_agent_stream(agent, request.question):
            si = _build_step_info(event)
            yield f"data: {si.model_dump_json()}\n\n"
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
