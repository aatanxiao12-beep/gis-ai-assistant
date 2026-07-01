"""
对比三种检索方法对同一查询的结果

运行前确保已构建向量库: python -m rag.pipeline build

用法:
    python tests/compare_retrieval.py "GML Curve 的定义"
    python tests/compare_retrieval.py "拓扑关系" --top_k 5
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.hybrid_retriever import get_hybrid_retriever
from rag.vectordb import get_vector_store


def compare(query: str, top_k: int = 3):
    hr = get_hybrid_retriever()
    store = get_vector_store()
    total = store._collection.count()

    print(f"查询: {query}")
    print(f"向量库文档数: {total} | Top-K: {top_k}")
    print()

    # 三种检索
    dense_results = hr.dense(query, top_k=top_k)
    sparse_results = hr.sparse(query, top_k=top_k)
    hybrid_results = hr.hybrid(query, top_k=top_k)

    methods = [
        ("稠密检索 (语义向量)", dense_results),
        ("稀疏检索 (BM25 关键词)", sparse_results),
        ("混合检索 (RRF 融合)", hybrid_results),
    ]

    for name, results in methods:
        print("=" * 65)
        print(f"  {name}  →  {len(results)} 个结果")
        print("=" * 65)
        if not results:
            print("  (无结果)\n")
            continue
        for i, doc in enumerate(results):
            meta = doc.metadata
            source = meta.get("source", "?")
            clause = meta.get("clause", "")
            comp = meta.get("component_name", "")
            label = clause or comp or source
            content = doc.page_content
            if len(content) > 300:
                content = content[:300] + "..."
            print(f"\n── {i+1}. {label}")
            print(f"   来源: {source}")
            print(f"   长度: {len(doc.page_content)} 字符")
            print(f"   ┌ {'─'*58}")
            for line in content.split("\n")[:8]:
                print(f"   │ {line}")
            print(f"   └ {'─'*58}")
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python tests/compare_retrieval.py <查询> [--top_k N]")
        print("示例: python tests/compare_retrieval.py \"GML Curve 的定义\"")
        sys.exit(1)

    query = sys.argv[1]
    top_k = 3
    for i, arg in enumerate(sys.argv):
        if arg == "--top_k" and i + 1 < len(sys.argv):
            top_k = int(sys.argv[i + 1])

    compare(query, top_k=top_k)
