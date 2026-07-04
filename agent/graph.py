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
- OGC GML 3.2.1 / 3.3.1 标准文档
- XSD Schema (feature, geometry, topology, temporal, coverage, CRS, measures...)
- ISO 19136 地理信息标准
- GML 应用模式、几何原语、拓扑关系、坐标参照系、时间模型、覆盖数据

## 工具使用原则（必须严格遵守）

1. **首次检索即终止**：调用一次 search_gis_standards 后，无论返回什么结果，都必须基于现有结果直接给出回答。**绝对禁止**对同一问题或相似关键词重复调用 search_gis_standards。

2. **结果不完美也要回答**：检索结果可能不100%覆盖问题，这是正常的。请基于最相关的片段尽力回答，并注明信息来源于哪些文档。不要因为"觉得不够"而再次检索。

3. **web_search 仅用于补充**：只有当问题涉及最新动态、开源工具（GDAL/QGIS/GeoServer）、网络资源等知识库不覆盖的内容时，才调用 web_search。对于标准规范类问题，search_gis_standards 一次即可。

4. **极其重要：每次看到工具返回结果后，你的下一个动作必须是输出最终回答（不带 tool_calls），而不是继续调用工具。**

## 回答要求
- 基于检索结果回答，关键事实标注来源编号 `[1]`、`[2]`
- 如有 XML/XSD 代码片段，优先展示
- 来源标注格式: `[1] 07-036r1.pdf — 10.4.2 CurvePropertyType`
- 使用中文回答，专业术语保留英文原名
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
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            # 统计已执行完成的工具调用轮数（ToolMessage 数量即已执行的轮数）
            completed_rounds = sum(
                1 for m in state["messages"]
                if getattr(m, "type", "") == "tool"
            )
            if completed_rounds >= 2:
                # 已执行 2 轮工具，强制结束，避免死循环
                logger.warning("已达到最大工具调用轮数 (2)，强制终止")
                return "end"
            return "tools"
        return "end"

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
    agent = agent.with_config({"recursion_limit": 6})
    logger.info("GIS AI Standard Assistant 工作流已构建")
    return agent


# ============================================================
# 运行接口
# ============================================================

def run_agent(agent, question: str) -> str:
    """向 agent 发送问题，返回最终回答文本"""
    result = agent.invoke({
        "messages": [{"role": "user", "content": question}],
    })

    for msg in reversed(result["messages"]):
        if getattr(msg, "type", "") == "ai" and msg.content:
            return msg.content

    return "未能生成回答。"


def run_agent_stream(agent, question: str):
    """
    流式运行 agent，yield 每个 graph step 的事件。

    每条事件包含:
        - step: 节点名称 (agent / tools)
        - msg_type: 消息类型 (ai / tool / human)
        - content: 消息内容
        - tool_calls: 工具调用信息（如有）
    """
    for event in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="updates",
    ):
        for node_name, node_output in event.items():
            if node_name == "agent":
                msg = node_output["messages"][-1]
                tc = getattr(msg, "tool_calls", None)
                yield {
                    "step": "agent",
                    "msg_type": "ai",
                    "content": msg.content or "",
                    "tool_calls": tc or [],
                }
            elif node_name == "tools":
                for msg in node_output["messages"]:
                    yield {
                        "step": "tools",
                        "msg_type": "tool",
                        "tool_name": getattr(msg, "name", "?"),
                        "content": str(msg.content),
                    }
