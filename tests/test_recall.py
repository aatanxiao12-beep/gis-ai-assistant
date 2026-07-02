"""
检索召回率测试

在已构建的向量库上运行预定义查询，检查检索结果是否命中预期关键词。

运行前确保已完成 build:
    python -m rag.pipeline build

运行:
    python tests/test_recall.py              # 全部测试
    python tests/test_recall.py --detail     # 逐条展示命中/遗漏
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.pipeline import search
from rag.vectordb import get_vector_store
from rag import config


# ============================================================
# 测试用例: (查询, [预期关键词列表])
# 关键词是检索返回的文档中应该包含的术语或短语
# ============================================================

TEST_CASES = [
    # ── GML 基础概念 ──
    (
        "What is the Geography Markup Language (GML)?",
        ["Geography Markup Language", "GML"],
    ),
    (
        "What are the core geometry types in GML?",
        ["Point", "Curve", "Surface", "geometry"],
    ),
    (
        "What is a gml:AbstractFeatureType?",
        ["AbstractFeatureType", "abstract", "feature"],
    ),

    # ── 几何原语 ──
    (
        "How is a gml:Point defined in GML schema?",
        ["Point", "pos", "directPosition"],
    ),
    (
        "What is gml:Curve and how is it represented?",
        ["Curve", "segments", "controlPoint"],
    ),
    (
        "How does GML define gml:Surface?",
        ["Surface", "patches", "polygon"],
    ),
    (
        "What is gml:Solid in 3D geometry?",
        ["Solid", "shell", "exterior"],
    ),

    # ── 拓扑 ──
    (
        "How does GML handle topology?",
        ["TopoPoint", "TopoCurve", "TopoSurface", "topology"],
    ),
    (
        "What is a TopoComplex in GML?",
        ["TopoComplex", "topoPrimitive"],
    ),

    # ── 坐标参照系 ──
    (
        "What coordinate reference systems does GML support?",
        ["CRS", "coordinate", "datum", "referenceSystem"],
    ),
    (
        "What is a gml:CoordinateSystem?",
        ["CoordinateSystem", "axis", "coordinate"],
    ),

    # ── 时间 ──
    (
        "How does GML model temporal information?",
        ["TimeInstant", "TimePeriod", "temporal"],
    ),

    # ── 覆盖 ──
    (
        "What coverage types are defined in GML?",
        ["Coverage", "domainSet", "rangeSet"],
    ),

    # ── OGC 标准条款 ──
    (
        "What is the scope of the OGC GML standard?",
        ["Scope", "Geography Markup Language"],
    ),
    (
        "What are the conformance requirements for GML?",
        ["conformance", "requirement", "Conformance"],
    ),

    # ── 字典与定义 ──
    (
        "How does GML define dictionaries and definitions?",
        ["Dictionary", "Definition", "dictionary"],
    ),

    # ── 动态要素 ──
    (
        "What is a dynamic feature in GML?",
        ["Dynamic", "dynamic", "Feature"],
    ),

    # ── 值对象 ──
    (
        "What value objects are defined in GML?",
        ["value", "Quantity", "Category"],
    ),
]


def run_recall_tests(show_detail: bool = False) -> dict:
    """返回 {query: {hit: int, miss: int, total: int}}"""
    store = get_vector_store()
    count = store._collection.count()
    if count == 0:
        print("向量库为空，请先运行: python -m rag.pipeline build\n")
        return {}

    print(f"向量库文档数: {count}")
    print(f"检索类型: {config.RETRIEVAL_SEARCH_TYPE}")
    print(f"Top-K: {config.RETRIEVAL_TOP_K}")
    print(f"阈值: {config.RETRIEVAL_SCORE_THRESHOLD}\n")
    print("=" * 70)

    results = {}
    total_hits = 0
    total_keywords = 0
    total_queries_with_results = 0

    for query, keywords in TEST_CASES:
        docs = search(query, top_k=config.RETRIEVAL_TOP_K)
        combined = "\n".join(d.page_content for d in docs)

        hits = [kw for kw in keywords if kw.lower() in combined.lower()]
        misses = [kw for kw in keywords if kw.lower() not in combined.lower()]

        if docs:
            total_queries_with_results += 1
        total_hits += len(hits)
        total_keywords += len(keywords)
        results[query] = {"hits": hits, "misses": misses, "doc_count": len(docs)}

        if show_detail:
            status = "✓" if not misses else f"✗ 缺 {len(misses)}/{len(keywords)}"
            print(f"\n【{status}】{query}")
            print(f"  返回 {len(docs)} 个文档")
            if hits:
                print(f"  命中: {', '.join(hits)}")
            if misses:
                print(f"  遗漏: {', '.join(misses)}")
            # 展示检索到的来源
            sources = {d.metadata.get('source', '?') for d in docs}
            print(f"  来源: {', '.join(sorted(sources))}")

    # 汇总
    recall = total_hits / total_keywords * 100 if total_keywords else 0
    q_with_results = total_queries_with_results / len(TEST_CASES) * 100
    print(f"\n{'='*70}")
    print(f"汇总: {len(TEST_CASES)} 条查询")
    print(f"至少返回 1 个结果的查询: {total_queries_with_results}/{len(TEST_CASES)} ({q_with_results:.0f}%)")
    print(f"关键词召回: {total_hits}/{total_keywords} ({recall:.1f}%)")

    return results


if __name__ == "__main__":
    show_detail = "--detail" in sys.argv or "-d" in sys.argv
    run_recall_tests(show_detail=show_detail)
