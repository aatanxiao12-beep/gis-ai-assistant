"""
检索模块

用法:
    from rag.retriever import retrieve
    docs = retrieve(store, "三维空间拓扑关系有哪些？")
"""

from langchain_chroma import Chroma
from langchain_core.documents import Document

from rag import config
from utils.logger_handle import logger


def retrieve(
    store: Chroma | None,
    query: str,
    top_k: int | None = None,
    score_threshold: float | None = None,
) -> list[Document]:
    """
    从向量库中检索与 query 最相关的文档。

    Args:
        store: Chroma 向量库实例
        query: 查询文本
        top_k: 返回文档数，默认 config.RETRIEVAL_TOP_K
        score_threshold: 相似度阈值，低于此分数的文档被过滤

    Returns:
        相关文档列表（按相似度降序）
    """
    if store is None:
        logger.warning("store 为空，返回空结果")
        return []

    if not query or not query.strip():
        logger.warning("query 为空，返回空结果")
        return []

    k = top_k or config.RETRIEVAL_TOP_K
    stype = config.RETRIEVAL_SEARCH_TYPE
    threshold = score_threshold if score_threshold is not None else config.RETRIEVAL_SCORE_THRESHOLD

    logger.debug("检索: type=%s top_k=%d threshold=%.2f query=%s...", stype, k, threshold, query[:80])

    if stype == "similarity":
        docs = store.similarity_search(query, k=k)

    elif stype == "mmr":
        docs = store.max_marginal_relevance_search(
            query, k=k,
            fetch_k=config.RETRIEVAL_FETCH_K,
            lambda_mult=config.RETRIEVAL_LAMBDA_MULT,
        )

    elif stype == "similarity_score_threshold":
        # relevance_score 已归一化到 [0,1]，越高越相关
        docs_with_scores = store.similarity_search_with_relevance_scores(query, k=k)
        docs = [doc for doc, score in docs_with_scores if score >= threshold]
        logger.debug("阈值过滤: %d → %d (threshold=%.2f)", len(docs_with_scores), len(docs), threshold)

    else:
        raise ValueError(f"不支持的检索类型: {stype}")

    logger.info("检索完成: 返回 %d 个文档", len(docs))
    return docs
