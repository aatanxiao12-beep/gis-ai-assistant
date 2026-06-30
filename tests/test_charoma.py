# tests/test_retrieval_with_scores.py
"""
测试向量检索功能，显示相似度分数
"""
from rag.vectordb import get_vector_store
from rag.config import RETRIEVAL_TOP_K, RETRIEVAL_SEARCH_TYPE, RETRIEVAL_FETCH_K, RETRIEVAL_LAMBDA_MULT
import logging

# 设置日志级别，减少干扰
logging.getLogger("agent").setLevel(logging.WARNING)


def search_with_scores(query: str, k: int = None):
    """
    执行检索并返回带分数的结果

    Args:
        query: 查询文本
        k: 返回结果数量，默认使用配置值

    Returns:
        list: (Document, score) 元组列表
    """
    store = get_vector_store()
    k = k or RETRIEVAL_TOP_K

    # 根据配置选择检索方式
    if RETRIEVAL_SEARCH_TYPE == "similarity_score_threshold":
        results = store.similarity_search_with_relevance_scores(query, k=k)
    else:
        # similarity 或 mmr 使用标准相似度搜索
        results = store.similarity_search_with_score(query, k=k)

    return results


def print_results(query: str, results: list, show_full: bool = False):
    """
    打印检索结果

    Args:
        query: 查询文本
        results: (Document, score) 列表
        show_full: 是否显示完整内容
    """
    print(f"\n{'=' * 70}")
    print(f"❓ 查询: {query}")
    print(f"📊 找到 {len(results)} 个结果")
    print('=' * 70)

    if not results:
        print("⚠️  未找到相关文档")
        return

    for i, (doc, score) in enumerate(results, 1):
        print(f"\n📄 结果 {i} (相似度: {score:.4f})")
        print(f"  来源: {doc.metadata.get('source', '未知')}")

        # 显示章节信息（如果有）
        section = doc.metadata.get('section', doc.metadata.get('header_1', ''))
        if section:
            print(f"  章节: {section}")

        # 显示内容
        content = doc.page_content
        if show_full:
            print(f"\n  完整内容:\n{content}")
        else:
            preview = content[:300] + "..." if len(content) > 300 else content
            print(f"\n  内容预览:\n{preview}")

        print('-' * 70)


def test_retrieval():
    """主测试函数"""

    # 测试查询列表
    test_queries = [
        "什么是9IM模型",
        "空间拓扑关系",
        "点集拓扑",
        "三维空间拓扑",
        "拓扑关系的类型",
        "Egenhofer 4IM 9IM"
    ]

    print("\n" + "=" * 70)
    print("🔍 RAG 检索测试（带相似度分数）")
    print("=" * 70)
    print(f"配置: search_type={RETRIEVAL_SEARCH_TYPE}, top_k={RETRIEVAL_TOP_K}")
    if RETRIEVAL_SEARCH_TYPE == "mmr":
        print(f"      fetch_k={RETRIEVAL_FETCH_K}, lambda_mult={RETRIEVAL_LAMBDA_MULT}")
    print("=" * 70)

    # 存储所有结果用于分析
    all_results = {}

    for query in test_queries:
        try:
            results = search_with_scores(query)
            all_results[query] = results
            print_results(query, results, show_full=False)
        except Exception as e:
            print(f"\n❌ 查询 '{query}' 失败: {e}")

    # 统计分析
    print("\n" + "=" * 70)
    print("📊 检索统计")
    print("=" * 70)

    scores = []
    for query, results in all_results.items():
        for doc, score in results:
            scores.append(score)

    if scores:
        print(f"  总检索结果数: {len(scores)}")
        print(f"  最高相似度: {max(scores):.4f}")
        print(f"  最低相似度: {min(scores):.4f}")
        print(f"  平均相似度: {sum(scores) / len(scores):.4f}")

        # 分数分布
        print(f"\n  分数分布:")
        ranges = [(0.9, 1.0, "0.90-1.00"), (0.8, 0.9, "0.80-0.89"),
                  (0.7, 0.8, "0.70-0.79"), (0.6, 0.7, "0.60-0.69"),
                  (0.5, 0.6, "0.50-0.59"), (0.0, 0.5, "0.00-0.49")]
        for low, high, label in ranges:
            count = sum(1 for s in scores if low <= s < high)
            if count > 0:
                print(f"    {label}: {count} 个")
    else:
        print("  没有检索结果")

    print("=" * 70)


def test_threshold_comparison():
    """测试不同阈值的检索效果"""
    print("\n" + "=" * 70)
    print("🎯 阈值对比测试")
    print("=" * 70)

    store = get_vector_store()
    query = "9IM模型"

    thresholds = [0.0, 0.3, 0.5, 0.7, 0.9]

    for threshold in thresholds:
        try:
            if RETRIEVAL_SEARCH_TYPE == "similarity_score_threshold":
                results = store.similarity_search_with_relevance_scores(
                    query, k=RETRIEVAL_TOP_K, score_threshold=threshold
                )
            else:
                results = store.similarity_search_with_score(query, k=RETRIEVAL_TOP_K)

            print(f"\n  阈值 {threshold:.1f}: 返回 {len(results)} 个结果")
            if results:
                avg_score = sum(score for _, score in results) / len(results)
                print(f"    平均分数: {avg_score:.4f}")
                for doc, score in results[:2]:
                    print(f"      {doc.metadata.get('source')[:30]}... ({score:.4f})")
        except Exception as e:
            print(f"\n  阈值 {threshold:.1f}: 错误 - {e}")


def test_mmr_vs_similarity():
    """对比 MMR 和相似度检索"""
    print("\n" + "=" * 70)
    print("📊 MMR vs 相似度检索对比")
    print("=" * 70)

    store = get_vector_store()
    query = "拓扑关系"

    # 1. 纯相似度
    print("\n  1. 纯相似度检索:")
    sim_results = store.similarity_search_with_score(query, k=5)
    for i, (doc, score) in enumerate(sim_results[:3], 1):
        print(f"    {i}. {doc.metadata.get('source')[:30]}... (score: {score:.4f})")

    # 2. MMR（如果配置是 mmr）
    if RETRIEVAL_SEARCH_TYPE == "mmr":
        print("\n  2. MMR 检索 (当前配置):")
        mmr_results = store.similarity_search_with_score(query, k=RETRIEVAL_TOP_K)
        for i, (doc, score) in enumerate(mmr_results[:3], 1):
            print(f"    {i}. {doc.metadata.get('source')[:30]}... (score: {score:.4f})")

    # 显示来源多样性
    print("\n  来源多样性分析:")
    all_sources = []
    for doc, _ in sim_results:
        all_sources.append(doc.metadata.get('source', '未知'))
    unique_sources = set(all_sources)
    print(f"    相似度检索: {len(unique_sources)} 个不同来源 / {len(all_sources)} 个结果")

    if RETRIEVAL_SEARCH_TYPE == "mmr":
        mmr_sources = []
        for doc, _ in mmr_results:
            mmr_sources.append(doc.metadata.get('source', '未知'))
        unique_mmr = set(mmr_sources)
        print(f"    MMR 检索: {len(unique_mmr)} 个不同来源 / {len(mmr_results)} 个结果")


if __name__ == "__main__":
    # 运行主测试
    test_retrieval()

    # 可选：额外测试
    print("\n" + "=" * 70)
    print("🔧 额外测试")
    print("=" * 70)

    # 测试特定查询
    specific_queries = [
        "三维空间拓扑关系完备性",
        "点集拓扑在GIS中的应用",
        "Egenhofer 的研究贡献"
    ]

    for q in specific_queries:
        print(f"\n❓ '{q}'")
        results = search_with_scores(q, k=5)
        for doc, score in results[:3]:
            print(f"  [{score:.4f}] {doc.metadata.get('source')[:40]}...")

    # 取消注释以运行额外测试
    # test_threshold_comparison()
    # test_mmr_vs_similarity()