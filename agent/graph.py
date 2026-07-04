"""
GIS AI Standard Assistant — LangGraph 工作流

编排流程:
    START → agent(node) → tools? → tools(node) → agent → ... → END

用法:
    from agent.graph import create_agent, run_agent, run_agent_stream
"""

import os

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from agent.state import AgentState
from agent.tools import TOOLS
from utils.logger_handle import logger

# ============================================================
# 系统提示词
# ============================================================

SYSTEM_PROMPT = """\
你是一个 **GIS AI 标准助手** (GIS AI Standard Assistant)。

## 职责
回答 OGC 标准、GML 规范、GIS 数据模型等地理信息领域专业问题。

## 知识范围
- OGC GML 3.2.1 / 3.3.1 标准文档（本地知识库）
- XSD Schema (feature, geometry, topology, temporal, coverage, CRS, measures...)
- ISO 19136 地理信息标准
- GML 应用模式、几何原语、拓扑关系、坐标参照系、时间模型、覆盖数据

## 可用工具

| 工具 | 用途 | 调用限制 |
|------|------|---------|
| `search_gis_standards` | 搜索本地知识库（标准规范、XSD、技术细节） | **整个对话只能调用 1 次** |
| `web_search` | 互联网搜索（最新动态、开源工具、补充资料） | 按需调用，每次用好结果 |
| `get_current_time` | 获取当前系统时间 | 需要时间上下文时调用 |

## 工具使用规则

1. **search_gis_standards 只能调用一次**：一次调用返回 8 个最相关的文档片段。拿到结果后立即基于结果回答，禁止再次调用。如果结果不完美，用已有的最相关信息尽力回答即可。

2. **涉及时间/版本/最新动态的问题**：
   - 先调用 `get_current_time` 知道当前日期
   - 再调用 `web_search` 搜索网络最新信息
   - 不要仅凭知识库的旧文档判断"最新"

3. **标准规范的技术问题**：直接调用 `search_gis_standards`，结合 web_search 补充最新动态。

4. **拿到工具结果后尽快回答**：工具返回结果后，整理信息输出最终回答。不要因为"觉得信息不够"而反复检索。每多一轮工具调用都会让上下文膨胀。

## 回答要求
- 涉及时间的问题，注明"截至 YYYY年MM月"
- 关键事实标注来源（本地知识库用 `[1]`，网络来源附 URL）
- 有代码片段/XSD 时优先展示
- 中文回答，专业术语保留英文原名
"""

# ============================================================
# 模型配置（解耦合）
# ============================================================

configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens", "api_key"),
)

DEFAULT_MODEL_CONFIG = {
    "model": "deepseek:deepseek-chat",
    "max_tokens": 4096,
    "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
    "base_url": "https://api.deepseek.com/v1",
    "temperature": 0.3,
    "tags": ["gis-assistant"],
}


def _get_llm():
    """构建带工具绑定的 LLM"""
    model = configurable_model.with_config(DEFAULT_MODEL_CONFIG)
    return model.bind_tools(TOOLS)


# ============================================================
# Agent 构建
# ============================================================

def create_agent():
    """
    构建 GIS AI Standard Assistant agent。

    使用 init_chat_model(configurable_fields=...) 解耦模型参数，
    通过 with_config 在调用时注入配置。
    """
    # 主线程预热：向量库 + BM25 索引（强制构建），避免工具线程初始化竞态
    from rag.vectordb import get_vector_store
    get_vector_store()
    from rag.hybrid_retriever import get_hybrid_retriever
    hr = get_hybrid_retriever()
    _ = hr.bm25  # 触发 BM25 索引构建

    llm_with_tools = _get_llm()

    # ── 节点 ──

    def agent_node(state: AgentState) -> dict:
        messages = state["messages"]
        if not any(getattr(m, "type", "") == "system" for m in messages):
            from langchain_core.messages import SystemMessage
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    tools_node = ToolNode(TOOLS)

    # ── 路由 ──

    def should_continue(state: AgentState) -> str:
        last_msg = state["messages"][-1]
        if not (hasattr(last_msg, "tool_calls") and last_msg.tool_calls):
            return "end"

        # ── search_gis_standards 限制为 1 次 ──
        gis_calls = sum(
            1 for m in state["messages"]
            if getattr(m, "type", "") == "tool"
            and getattr(m, "name", "") == "search_gis_standards"
        )
        wants_gis = any(
            tc.get("name") == "search_gis_standards"
            for tc in last_msg.tool_calls
        )
        if wants_gis and gis_calls >= 1:
            logger.warning("search_gis_standards 已调用一次，拒绝重复调用，强制结束")
            return "end"

        # ── 全局保护：总工具执行轮数不超过 8（防止上下文膨胀） ──
        total_rounds = sum(
            1 for m in state["messages"]
            if getattr(m, "type", "") == "tool"
        )
        if total_rounds >= 8:
            logger.warning("总工具调用轮数已达 %d，强制终止", total_rounds)
            return "end"

        return "tools"

    # ── 构图 ──

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tools_node)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue, {
        "tools": "tools",
        "end": END,
    })
    builder.add_edge("tools", "agent")

    agent = builder.compile()
    agent = agent.with_config({"recursion_limit": 10})
    logger.info("GIS AI Standard Assistant 工作流已构建")
    return agent


# ============================================================
# 消息规范化
# ============================================================

def _normalize_messages(messages) -> list[dict]:
    """将 Pydantic Message 对象或 dict 统一转为 dict 列表，供 LangGraph 使用"""
    result = []
    for m in messages:
        if hasattr(m, "model_dump"):
            result.append(m.model_dump())
        elif isinstance(m, dict):
            result.append(m)
        else:
            result.append({"role": "user", "content": str(m)})
    return result


# ============================================================
# 运行接口
# ============================================================

def run_agent(agent, messages) -> str:
    """向 agent 发送消息列表，支持多轮对话，返回最终回答文本"""
    result = agent.invoke({
        "messages": _normalize_messages(messages),
    })

    for msg in reversed(result["messages"]):
        if getattr(msg, "type", "") == "ai" and msg.content:
            return msg.content

    return "未能生成回答。"


def run_agent_stream(agent, messages):
    """
    逐字流式运行 agent，LLM 每生成一个 token 就 yield。

    参数:
        messages: OpenAI 格式的消息列表 [{"role": "user", "content": "..."}, ...]
                  或 Pydantic Message 对象列表

    每条事件包含:
        - step: 节点名称 (agent / tools)
        - msg_type: 消息类型 (ai / tool)
        - delta: True 表示增量内容，False 表示完整事件
        - content: 增量文本（delta=True）或空（delta=False 时在 tool_calls 中）
        - tool_calls: 工具调用信息（仅当 LLM 决定调用工具时一次性给出）
        - tool_name: 工具名称（仅 tools 步骤）
    """
    tool_call_acc: dict[int, dict] = {}       # index → {name, args}
    current_tool_index: int | None = None
    normalized = _normalize_messages(messages)

    for msg, metadata in agent.stream(
        {"messages": normalized},
        stream_mode="messages",
    ):
        node_name = metadata.get("langgraph_node", "")

        if node_name == "agent":
            content_delta = getattr(msg, "content", "") or ""
            tc = getattr(msg, "tool_calls", None) or []
            tc_chunks = getattr(msg, "tool_call_chunks", None) or []

            # ── 处理 tool_call_chunks（增量构建工具调用）──
            for chunk in tc_chunks:
                idx = chunk.get("index", 0)
                if idx not in tool_call_acc:
                    tool_call_acc[idx] = {"name": "", "args": {}}
                if chunk.get("name"):
                    tool_call_acc[idx]["name"] = chunk["name"]
                if chunk.get("args"):
                    tool_call_acc[idx]["args"] = chunk["args"]
                current_tool_index = idx

            # ── 有完整的 tool_calls → 一次性 yield ──
            if tc:
                # 过滤掉 name 为空的中间态 tool_call
                valid = [t for t in tc if t.get("name")]
                if valid:
                    yield {
                        "step": "agent",
                        "msg_type": "ai",
                        "delta": False,
                        "content": "",
                        "tool_calls": valid,
                    }
                tool_call_acc.clear()
                continue

            # ── 有 tool_call_chunks 但无内容 → 跳过（中间态，等完整 tc）──
            if tc_chunks:
                continue

            # ── 有增量内容 → 逐字 yield ──
            if content_delta:
                yield {
                    "step": "agent",
                    "msg_type": "ai",
                    "delta": True,
                    "content": content_delta,
                    "tool_calls": [],
                }

        elif node_name == "tools":
            yield {
                "step": "tools",
                "msg_type": "tool",
                "delta": False,
                "tool_name": getattr(msg, "name", "?"),
                "content": str(msg.content),
            }
