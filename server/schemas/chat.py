"""
Pydantic 请求 / 响应模型
"""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户问题")


class ToolCallInfo(BaseModel):
    name: str
    args: dict = {}


class StepInfo(BaseModel):
    step: str          # "agent" | "tools"
    msg_type: str      # "ai" | "tool"
    content: str = ""
    tool_calls: list[ToolCallInfo] = []
    tool_name: str = ""


class ChatResponse(BaseModel):
    answer: str
    steps: list[str] = []
    tool_calls: list[ToolCallInfo] = []
    events: list[StepInfo] = []
