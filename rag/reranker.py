"""
重排序模块

当前仅做截断（RERANK_METHOD="none"），后续可按需扩展。
"""

from langchain_core.documents import Document

from rag import config
from utils.logger_handle import logger


def rerank(query: str, documents: list[Document], top_n: int | None = None) -> list[Document]:
    """对候选文档截断，返回 top_n 个"""
    if not documents:
        return []

    n = top_n or config.RERANK_TOP_N
    logger.debug("重排序: %d → %d", len(documents), min(n, len(documents)))
    return documents[:n]
