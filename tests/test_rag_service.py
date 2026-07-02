"""
测试 rag_service 四个接口，对比同一查询的结果

用法:
    python tests/test_rag_service.py "GML Curve 的定义"
    python tests/test_rag_service.py "拓扑关系" --top_k 5
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.rag_service import dense, bm25, hybrid, rerank


def test(query: str, top_k: int = 5):
    print(f"查询: {query}  |  top_k={top_k}")
    print()

    # 三个初筛
    dense_docs = dense(query, top_k=top_k * 2)
    bm25_docs = bm25(query, top_k=top_k * 2)
    hybrid_docs = hybrid(query, top_k=top_k * 2)

    # 精排：以 hybrid 初筛结果作为候选
    rerank_docs = rerank(query, hybrid_docs, top_n=top_k)

    methods = [
        ("稠密检索 (语义向量)", dense_docs[:top_k]),
        ("BM25 稀疏 (关键词)", bm25_docs[:top_k]),
        ("RRF 混合", hybrid_docs[:top_k]),
        ("Hybrid + Cross-Encoder 精排", rerank_docs),
    ]

    for name, docs in methods:
        print(f"{'─' * 60}")
        print(f"  {name}  ({len(docs)} 条)")
        print(f"{'─' * 60}")
        if not docs:
            print("  (无结果)\n")
            continue
        for i, doc in enumerate(docs):
            src = doc.metadata.get("source", "?")
            clause = doc.metadata.get("clause", "") or doc.metadata.get("component_name", "") or src
            text = doc.page_content.replace("\n", " ")[:150]
            print(f"  [{i+1}] {clause}")
            print(f"      {text}...")
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('用法: python tests/test_rag_service.py "查询内容" [--top_k N]')
        sys.exit(1)

    query = sys.argv[1]
    top_k = 5
    for i, arg in enumerate(sys.argv):
        if arg == "--top_k" and i + 1 < len(sys.argv):
            top_k = int(sys.argv[i + 1])

    test(query, top_k=top_k)
