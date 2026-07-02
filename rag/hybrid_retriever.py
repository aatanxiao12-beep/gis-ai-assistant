"""
混合检索模块

- 稠密检索: Chroma 向量相似度
- 稀疏检索: 内存 BM25 倒排索引（从 Chroma 加载文档构建，不写入 metadata）
- 混合检索: RRF 融合

用法:
    from rag.hybrid_retriever import get_hybrid_retriever
    hr = get_hybrid_retriever()
    docs = hr.hybrid("GML Curve 的定义")
"""

import math
from collections import defaultdict

from langchain_chroma import Chroma
from langchain_core.documents import Document

from rag import config
from utils.logger_handle import logger


# ============================================================
# 分词 — 中文 bigram + 英文按词
# ============================================================

def _tokenize(text: str) -> list[str]:
    """混合分词: 中文 bigram + 英文小写词"""
    import re
    tokens = []
    segments = re.findall(r'[\u4e00-\u9fa5]+|[a-zA-Z0-9]+', text.lower())
    for seg in segments:
        if re.match(r'[\u4e00-\u9fa5]', seg):
            if len(seg) == 1:
                tokens.append(seg)
            else:
                tokens.extend(seg[i:i+2] for i in range(len(seg) - 1))
        else:
            if len(seg) > 1:
                tokens.append(seg)
    return tokens


# ============================================================
# 内存 BM25 倒排索引
# ============================================================

class BM25Index:
    """内存 BM25 索引，从文档列表构建，不持久化"""

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._docs: list[Document] = []
        self._doc_freq: dict[str, int] = {}
        self._term_freqs: list[dict[str, int]] = []
        self._doc_lengths: list[int] = []
        self._avg_dl: float = 0.0

    def build(self, documents: list[Document]) -> None:
        self._docs = documents
        n = len(documents)
        if n == 0:
            return

        self._doc_freq = defaultdict(int)
        self._term_freqs = []
        self._doc_lengths = []

        for doc in documents:
            tokens = _tokenize(doc.page_content)
            tf = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            self._term_freqs.append(tf)
            self._doc_lengths.append(len(tokens))
            for t in set(tokens):
                self._doc_freq[t] += 1

        self._avg_dl = sum(self._doc_lengths) / n

    def search(self, query: str, top_k: int = 10) -> list[tuple[float, Document]]:
        if not self._docs or self._avg_dl == 0.0:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        n = len(self._docs)
        scores = []
        for i in range(n):
            tf = self._term_freqs[i]
            dl = self._doc_lengths[i]
            if dl == 0:
                continue
            score = 0.0
            for token in set(query_tokens):
                df = self._doc_freq.get(token, 0)
                if df == 0:
                    continue
                idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
                f = tf.get(token, 0)
                numerator = f * (self.k1 + 1)
                denominator = f + self.k1 * (1 - self.b + self.b * dl / self._avg_dl)
                score += idf * numerator / denominator
            if score > 0:
                scores.append((score, self._docs[i]))

        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[:top_k]


# ============================================================
# RRF 融合
# ============================================================

def reciprocal_rank_fusion(
    result_sets: list[list[Document]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[Document]:
    """RRF 融合多个已排序的检索结果"""
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    if weights is None:
        weights = [1.0] * len(result_sets)

    for w, results in zip(weights, result_sets):
        for rank, doc in enumerate(results):
            key = doc.metadata.get("source", "") + "::" + doc.page_content[:200]
            scores[key] = scores.get(key, 0.0) + w / (k + rank + 1)
            doc_map[key] = doc

    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[key] for key in sorted_keys]


# ============================================================
# 混合检索器
# ============================================================

class HybridRetriever:
    """
    混合检索器。

    - 稠密: Chroma 语义向量
    - 稀疏: 内存 BM25（从 Chroma 加载文档构建，不写 metadata）
    - 混合: RRF 融合
    """

    def __init__(self, store: Chroma | None = None):
        self._store = store
        self._bm25: BM25Index | None = None

    @property
    def store(self) -> Chroma:
        if self._store is None:
            from rag.vectordb import get_vector_store
            self._store = get_vector_store()
        return self._store

    @property
    def bm25(self) -> BM25Index:
        if self._bm25 is None:
            logger.info("构建内存 BM25 索引...")
            docs = self._load_all_documents()
            self._bm25 = BM25Index()
            self._bm25.build(docs)
            logger.info("BM25 索引完成 | %d 篇文档", len(docs))
        return self._bm25

    def _load_all_documents(self) -> list[Document]:
        try:
            results = self.store.get(include=["metadatas", "documents"])
            if not results or not results["ids"]:
                return []
            docs = []
            for text, meta in zip(results["documents"], results["metadatas"] or [{}] * len(results["ids"])):
                docs.append(Document(page_content=text, metadata=meta or {}))
            return docs
        except Exception:
            return []

    def rebuild_index(self) -> None:
        """向量库更新后强制重建 BM25 索引"""
        self._bm25 = None

    # ── 检索 ──

    def dense(self, query: str, top_k: int | None = None,
              score_threshold: float | None = None) -> list[Document]:
        """稠密向量检索"""
        from rag.retriever import retrieve
        k = top_k or config.RETRIEVAL_TOP_K
        threshold = score_threshold if score_threshold is not None else config.RETRIEVAL_SCORE_THRESHOLD
        return retrieve(self.store, query, top_k=k, score_threshold=threshold)

    def sparse(self, query: str, top_k: int | None = None) -> list[Document]:
        """BM25 稀疏检索（内存倒排索引）"""
        k = top_k or config.RETRIEVAL_TOP_K
        results = self.bm25.search(query, top_k=k)
        return [doc for _, doc in results]

    def hybrid(self, query: str, top_k: int | None = None,
               rrf_k: int = 60, dense_weight: float = 1.0,
               sparse_weight: float = 1.0) -> list[Document]:
        """RRF 混合检索"""
        k = top_k or config.RETRIEVAL_TOP_K

        dense_results = self.dense(query, top_k=k * 3)
        sparse_results = self.sparse(query, top_k=k * 3)

        if not dense_results and not sparse_results:
            return []
        if not dense_results:
            return sparse_results[:k]
        if not sparse_results:
            return dense_results[:k]

        fused = reciprocal_rank_fusion(
            [dense_results, sparse_results],
            k=rrf_k, weights=[dense_weight, sparse_weight],
        )
        logger.debug("混合检索: dense=%d sparse=%d → fused=%d",
                      len(dense_results), len(sparse_results), len(fused))
        return fused[:k]


# ============================================================
# 单例
# ============================================================

_retriever: HybridRetriever | None = None


def get_hybrid_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever
