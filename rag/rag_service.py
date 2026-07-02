"""
RAG 服务层 —— 对外暴露四个检索接口

    稠密检索    —  Chroma 向量语义相似度
    BM25 稀疏   —  内存 BM25 关键词倒排索引
    RRF 融合    —  稠密 + 稀疏 RRF 融合
    Cross-Encoder 精排 — 对候选文档用交叉编码器重排序

用法:
    from rag.rag_service import dense, bm25, hybrid, rerank

    docs = dense("GML Curve 的定义", top_k=10)
    docs = bm25("拓扑关系")
    docs = hybrid("空间数据模型", top_k=5)
    docs = rerank("坐标系", docs_candidates, top_n=3)
"""

from langchain_core.documents import Document

from rag import config
from rag.retriever import retrieve as _dense_retrieve
from rag.hybrid_retriever import get_hybrid_retriever
from rag.reranker import rerank as _cross_encoder_rerank


# ── 单例 ──

_hr = None

def _get_hr():
    global _hr
    if _hr is None:
        _hr = get_hybrid_retriever()
    return _hr


# ============================================================
# 稠密检索（语义向量）
# ============================================================

def dense(
    query: str,
    top_k: int | None = None,
    score_threshold: float | None = None,
) -> list[Document]:
    """
    Chroma 向量语义检索。

    Args:
        query: 查询文本
        top_k: 返回数量（默认 config.RETRIEVAL_TOP_K）
        score_threshold: 相似度阈值
    """
    hr = _get_hr()
    k = top_k or config.RETRIEVAL_TOP_K
    threshold = score_threshold if score_threshold is not None else config.RETRIEVAL_SCORE_THRESHOLD
    return _dense_retrieve(hr.store, query, top_k=k, score_threshold=threshold)


# ============================================================
# BM25 稀疏检索（关键词）
# ============================================================

def bm25(
    query: str,
    top_k: int | None = None,
) -> list[Document]:
    """
    内存 BM25 关键词检索（中文 bigram + 英文词）。

    Args:
        query: 查询文本
        top_k: 返回数量
    """
    hr = _get_hr()
    k = top_k or config.RETRIEVAL_TOP_K
    return hr.sparse(query, top_k=k)


# ============================================================
# RRF 混合检索（稠密 + 稀疏）
# ============================================================

def hybrid(
    query: str,
    top_k: int | None = None,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
    rrf_k: int = 60,
) -> list[Document]:
    """
    RRF 混合检索：融合稠密语义 + BM25 关键词。

    Args:
        query: 查询文本
        top_k: 返回数量
        dense_weight: 稠密检索权重
        sparse_weight: 稀疏检索权重
        rrf_k: RRF 融合参数 k
    """
    hr = _get_hr()
    k = top_k or config.RETRIEVAL_TOP_K
    return hr.hybrid(
        query, top_k=k,
        dense_weight=dense_weight, sparse_weight=sparse_weight,
        rrf_k=rrf_k,
    )


# ============================================================
# Cross-Encoder 精排
# ============================================================

def rerank(
    query: str,
    documents: list[Document],
    top_n: int | None = None,
) -> list[Document]:
    """
    对候选文档用 cross-encoder 精排。

    Args:
        query: 查询文本
        documents: 候选文档列表
        top_n: 精排后保留数
    """
    return _cross_encoder_rerank(query, documents, top_n=top_n)
