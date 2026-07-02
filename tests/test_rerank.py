"""
重排序对比测试

展示初筛结果（向量相似度）与 cross-encoder 精排结果的对比，
直观看到重排序前后文档排位的变化。

用法:
    python tests/test_rerank.py "GML Curve 的定义"
    python tests/test_rerank.py "拓扑关系" --candidates 15 --top_n 5
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.vectordb import get_vector_store
from rag.reranker import _get_cross_encoder
from rag import config


def compare(query: str, candidates: int = 12, top_n: int = 5):
    store = get_vector_store()
    total = store._collection.count()
    print(f"查询: {query}")
    print(f"向量库文档数: {total} | 初筛候选: {candidates} | 最终输出: {top_n}")
    print()

    # ── 阶段1: 向量初筛（保留分数） ──
    raw = store.similarity_search_with_relevance_scores(query, k=candidates)
    # 放宽阈值，保证候选足够
    initial = [(doc, score) for doc, score in raw if score >= 0.2]

    if not initial:
        print("无结果（阈值过高或向量库为空）")
        return

    print(f"初筛阶段: 向量相似度检索 → 召回 {len(initial)} 篇候选")
    print()

    # ── 阶段2: Cross-encoder 精排 ──
    docs_only = [doc for doc, _ in initial]
    model = _get_cross_encoder()
    pairs = [(query, doc.page_content) for doc in docs_only]
    ce_scores = model.predict(pairs, show_progress_bar=False)
    # 归一化到 [0,1] 便于对比
    ce_min, ce_max = min(ce_scores), max(ce_scores)
    if ce_max - ce_min > 0:
        ce_norm = [(s - ce_min) / (ce_max - ce_min) for s in ce_scores]
    else:
        ce_norm = [0.5] * len(ce_scores)

    reranked = list(zip(ce_norm, ce_scores, docs_only))
    reranked.sort(key=lambda x: x[0], reverse=True)

    # ── 排名变化追踪 ──
    doc_to_initial_rank = {}
    for rank, (doc, _) in enumerate(initial, 1):
        key = doc.metadata.get("source", "") + "::" + doc.page_content[:120]
        doc_to_initial_rank[key] = rank

    # ── 输出 ──

    # 初筛排名
    print("=" * 72)
    print("  初筛排名（向量相似度）")
    print("=" * 72)
    for rank, (doc, score) in enumerate(initial, 1):
        _print_doc(rank, doc, score)
        if rank >= candidates:
            break
    print()

    # 精排排名
    print("=" * 72)
    print("  精排排名（Cross-Encoder 重排序）")
    print("=" * 72)
    for new_rank, (norm, raw_ce, doc) in enumerate(reranked[:top_n], 1):
        key = doc.metadata.get("source", "") + "::" + doc.page_content[:120]
        old_rank = doc_to_initial_rank.get(key, 0)
        if old_rank == 0:
            change = "  NEW"
        elif new_rank < old_rank:
            change = f"  ↑{old_rank - new_rank} (原第{old_rank}位)"
        elif new_rank > old_rank:
            change = f"  ↓{new_rank - old_rank} (原第{old_rank}位)"
        else:
            change = "  ─ 位次不变"
        _print_doc(new_rank, doc, raw_ce, tag=change)

    print()
    print(f"最终输出 top-{top_n}，已按 cross-encoder 分数降序排列")


def _print_doc(rank: int, doc, score: float, tag: str = ""):
    meta = doc.metadata
    source = meta.get("source", "?")
    clause = meta.get("clause", "")
    comp = meta.get("component_name", "")
    label = clause or comp or source

    content = doc.page_content.replace("\n", " ")[:120]
    print(f"  [{rank:2d}] {label}")
    print(f"      分数: {score:.4f}{tag}")
    print(f"      内容: {content}...")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python tests/test_rerank.py <查询> [--candidates N] [--top_n N]")
        print('示例: python tests/test_rerank.py "GML Curve 的定义" --candidates 15')
        sys.exit(1)

    query = sys.argv[1]
    candidates = 12
    top_n = 5

    for i, arg in enumerate(sys.argv):
        if arg == "--candidates" and i + 1 < len(sys.argv):
            candidates = int(sys.argv[i + 1])
        elif arg == "--top_n" and i + 1 < len(sys.argv):
            top_n = int(sys.argv[i + 1])

    compare(query, candidates=candidates, top_n=top_n)
