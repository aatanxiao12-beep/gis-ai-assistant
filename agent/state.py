"""
Agent 状态定义

用法:
    from agent.state import AgentState
"""

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """
    Agent 全局状态。

    messages 使用 operator.add 作为 reducer，
    新消息追加到历史列表末尾而非覆盖。
    """
    messages: Annotated[list[BaseMessage], operator.add]
