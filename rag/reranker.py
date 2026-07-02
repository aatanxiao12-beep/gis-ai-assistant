"""
重排序模块

支持两种模式:
    cross_encoder — 使用 sentence-transformers CrossEncoder 模型精排
    none           — 简单截断，不做重排

用法:
    from rag.reranker import rerank
    docs = rerank("GML Curve 的定义", documents)
"""

from langchain_core.documents import Document

from rag import config
from utils.logger_handle import logger

# 模型单例
_cross_encoder = None


def _get_cross_encoder():
    """懒加载 CrossEncoder 模型（首次调用时从 HuggingFace / 本地缓存加载）"""
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder

        model_name = config.RERANK_MODEL_NAME
        logger.info("加载 CrossEncoder: %s", model_name)
        _cross_encoder = CrossEncoder(
            model_name,
            trust_remote_code=True,
        )
    return _cross_encoder


def rerank(
    query: str,
    documents: list[Document],
    top_n: int | None = None,
) -> list[Document]:
    """
    对候选文档重排序，返回 top_n 个最相关文档。

    Args:
        query: 用户查询
        documents: 候选文档列表
        top_n: 返回文档数

    Returns:
        重排序后的文档列表（相关度降序）
    """
    if not documents:
        return []

    method = config.RERANK_METHOD
    n = top_n or config.RERANK_TOP_N

    # ── 不重排，直接截断 ──
    if method == "none":
        logger.debug("重排序(skip): %d → %d", len(documents), min(n, len(documents)))
        return documents[:n]

    # ── Cross-encoder 精排 ──
    if method == "cross_encoder":
        model = _get_cross_encoder()

        # 构造 (query, doc) 对
        pairs = [(query, doc.page_content) for doc in documents]
        scores = model.predict(pairs, show_progress_bar=False)

        # scores 可能是 numpy array 或 list，统一处理
        scored = list(zip(scores, documents))
        scored.sort(key=lambda x: x[0], reverse=True)

        logger.debug("CrossEncoder 重排序: %d → %d", len(documents), min(n, len(scored)))
        return [doc for _, doc in scored[:n]]

    logger.warning("未知重排序方法 '%s'，回退到截断模式", method)
    return documents[:n]
