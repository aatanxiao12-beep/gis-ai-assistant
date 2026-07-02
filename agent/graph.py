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

## 工具使用原则
1. GIS 标准/规范问题 → 调用 **search_gis_standards**，一次调用返回 5 个已排序的最佳结果
2. 最新动态、工具库、网络资源 → 调用 **web_search**
3. **严禁重复调用同一工具**：search_gis_standards 返回的结果已是最优，直接从中提取信息回答即可
4. 只有当首次结果标题完全不相关时，换一个关键词重试，最多 1 次

## 检索结果使用
检索结果通常包含多个来源，请充分利用：
1. 引用最相关的 **2-3 个**不同来源，交叉验证信息完整性
2. 不同来源有补充信息时，整合呈现，不要只取第一条
3. 每个关键事实后面标注来源编号 `[1]`、`[2]` 等
4. 如果某个来源包含代码块或 XML Schema，优先展示

## 回答结构
请按以下层次组织回答，确保结构完整：

1. **核心定义** — 引用规范原文，给出准确的技术定义
2. **关键属性** — 列出重要的属性、约束和类型
3. **继承与关系** — 说明与其他类型/元素的继承、引用、组合关系
4. **XML/XSD 示例** — 如有代码片段，用代码块展示
5. **引用来源** — 末尾列出所有引用的来源文件

## 格式要求
- 使用中文回答，专业术语保留英文原名
- 每个章节标注编号（一、二、三…），结构必须覆盖完整，不可只写部分
- 来源标注格式: `[1] 07-036r1.pdf — 10.4.2 CurvePropertyType`
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
    agent = agent.with_config({"recursion_limit": 10})
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
