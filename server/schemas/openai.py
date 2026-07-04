"""
OpenAI API 兼容的 Pydantic 请求/响应模型

参考: https://platform.openai.com/docs/api-reference/chat/create
"""

import time
import uuid
from pydantic import BaseModel, Field


# ============================================================
# 请求
# ============================================================

class Message(BaseModel):
    role: str = "user"              # system / user / assistant / developer
    content: str = Field(..., min_length=1)


class ChatCompletionRequest(BaseModel):
    model: str = "gis-assistant"
    messages: list[Message] = Field(..., min_length=1)
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    stop: list[str] | None = None


# ============================================================
# 响应 (非流式)
# ============================================================

class DeltaMessage(BaseModel):
    role: str | None = None
    content: str | None = None


class Choice(BaseModel):
    index: int = 0
    message: DeltaMessage = Field(default_factory=lambda: DeltaMessage(role="assistant", content=""))
    finish_reason: str | None = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str = ""
    object: str = "chat.completion"
    created: int = 0
    model: str = "gis-assistant"
    choices: list[Choice] = []
    usage: Usage = Field(default_factory=Usage)


# ============================================================
# 流式 chunk
# ============================================================

class StreamChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage = Field(default_factory=lambda: DeltaMessage())
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    id: str = ""
    object: str = "chat.completion.chunk"
    created: int = 0
    model: str = "gis-assistant"
    choices: list[StreamChoice] = []


# ============================================================
# 模型列表
# ============================================================

class ModelInfo(BaseModel):
    id: str = "gis-assistant"
    object: str = "model"
    created: int = 1700000000
    owned_by: str = "gis-ai-assistant"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelInfo] = Field(default_factory=lambda: [ModelInfo()])


# ============================================================
# 帮助函数
# ============================================================

def make_chat_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def _now() -> int:
    return int(time.time())


def make_response(model: str, content: str) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id=make_chat_id(),
        created=_now(),
        model=model,
        choices=[Choice(
            index=0,
            message=DeltaMessage(role="assistant", content=content),
            finish_reason="stop",
        )],
    )


def make_chunk(model: str, delta: DeltaMessage, finish_reason: str | None = None,
               chunk_id: str = "") -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id=chunk_id or make_chat_id(),
        created=_now(),
        model=model,
        choices=[StreamChoice(
            index=0,
            delta=delta,
            finish_reason=finish_reason,
        )],
    )
