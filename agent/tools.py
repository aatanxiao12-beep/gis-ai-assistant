"""
Agent 工具

用法:
    from agent.tools import search_gis_standards, web_search
"""

import os

from langchain.tools import tool

from rag.rag_service import hybrid


@tool
def search_gis_standards(query: str) -> str:
    """
    搜索本地 GIS 标准知识库（OGC GML / ISO 19136 / XSD Schema）。

    一次调用返回 top-8 最相关文档片段，已按混合检索（语义+关键词）排序。
    结果包含完整的技术定义、代码示例和来源标注。

    **严格限制：整个对话中只能调用本工具一次。收到结果后必须直接回答，禁止再次调用。**
    """
    docs = hybrid(query, top_k=10)
    if not docs:
        return "未在知识库中找到相关内容。建议尝试其他关键词或调用 web_search。"

    docs = docs[:8]

    summary = f"共检索到 {len(docs)} 个相关文档片段（已按相关度排序）:\n"
    parts: list[str] = [summary]

    for i, doc in enumerate(docs):
        m = doc.metadata
        src = m.get("source", "?")
        clause = m.get("clause", "")
        comp_name = m.get("component_name", "")
        comp_type = m.get("component_type", "")
        category = m.get("category", "")

        if category == "xsd_schema" and comp_name:
            label = f"{comp_name} ({comp_type})"
        elif clause:
            label = clause
        else:
            label = src

        header = f"[{i + 1}] {label}  — {src}"
        content = doc.page_content[:1200]
        if len(doc.page_content) > 1200:
            content += "..."
        parts.append(f"{header}\n{content}")

    return "\n\n---\n\n".join(parts)


@tool
def web_search(query: str) -> str:
    """
    使用 Tavily 搜索引擎查询网络上的 GIS 相关信息。

    适用: 最新 GIS 标准动态、开源工具 (GDAL/QGIS/GeoServer)、
    技术博客、社区问答、知识库未覆盖的补充资料。
    """
    from tavily import TavilyClient

    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY", ""))
    try:
        result = client.search(
            query,
            search_depth="advanced",
            max_results=5,
            include_answer=True,
        )
    except Exception as e:
        return f"Tavily 搜索失败: {e}"

    parts: list[str] = []

    answer = result.get("answer")
    if answer:
        parts.append(f"[综合]\n{answer}")

    for i, r in enumerate(result.get("results", [])):
        parts.append(
            f"[{i + 1}] {r.get('title', '')}\n"
            f"    URL: {r.get('url', '')}\n"
            f"    {r.get('content', '')}"
        )

    return "\n\n".join(parts) if parts else "未找到相关结果。"


TOOLS = [search_gis_standards, web_search]
