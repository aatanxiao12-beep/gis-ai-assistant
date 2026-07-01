"""
RAG 管道

两个核心函数:
    build_kb()   — 从 PDF 论文构建向量知识库
    search()     — 检索 + 重排序

用法:
    from rag.pipeline import build_kb, search
    build_kb()
    result = search("三维空间中线与体的拓扑关系有哪些？")
"""

from langchain_core.documents import Document

from rag import config
from rag.loader import build_survey_knowledge_base, clear_md5_store
from rag.embedding import get_embeddings
from rag.vectordb import get_vector_store, add_documents, clear_collection
from rag.retriever import retrieve, rerank
from utils.logger_handle import logger


def build_kb(clear_existing: bool = False) -> int:
    """
    构建向量知识库。

    1. loader 扫描 data/ 目录，MD5 去重，加载新文件
    2. 逐块向量化写入 ChromaDB

    Args:
        clear_existing: 清空已有集合后全量重建（忽略 MD5 记录）

    Returns:
        知识库中的文档总数
    """
    emb = get_embeddings()
    store = get_vector_store(embeddings=emb)

    if clear_existing:
        clear_collection(store)
        clear_md5_store()
        store = get_vector_store(embeddings=emb)

    logger.info("扫描数据文件（MD5 去重）...")
    documents = build_survey_knowledge_base()

    if not documents:
        logger.warning("未找到需要入库的新文件")
        return store._collection.count()

    add_documents(store, documents)
    count = store._collection.count()
    logger.info("知识库构建完成 | 文档总数: %d | 集合: %s", count, config.CHROMA_COLLECTION_NAME)
    return count


def search(question: str, top_k: int | None = None) -> list[Document]:
    """
    端到端查询：检索 → 重排序。

    Args:
        question: 用户问题
        top_k: 检索文档数

    Returns:
        重排序后的文档列表
    """
    store = get_vector_store()
    docs = retrieve(store, question, top_k=top_k)

    if docs and config.RERANK_METHOD != "none":
        docs = rerank(question, docs)

    return docs


def search_with_context(question: str, top_k: int | None = None) -> str:
    """
    查询并拼接为 LLM 可用的上下文字符串。

    适合直接拼入对话模型的 prompt。
    """
    docs = search(question, top_k=top_k)

    parts = []
    for i, doc in enumerate(docs):
        src = doc.metadata.get("source", "")
        author = doc.metadata.get("author", "")
        year = doc.metadata.get("year", "")
        header = f"【参考 {i + 1}】{author} ({year}) — {src}"
        parts.append(f"{header}\n{doc.page_content}")

    context = "\n\n---\n\n".join(parts)
    logger.info("上下文拼接完成 | %d 个片段", len(parts))
    return context


# ============================================================
# 命令行入口
# ============================================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法:")
        print("  python -m rag.pipeline build      # 构建知识库")
        print("  python -m rag.pipeline search     # 交互式检索")
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == "build":
        n = build_kb(clear_existing=True)
        print(f"构建完成，共 {n} 个文档块。")

    elif cmd == "search":
        print("RAG 知识库检索 (输入 quit 退出)\n")
        while True:
            q = input(">>> ").strip()
            if q.lower() in ("quit", "exit", "q"):
                break
            if not q:
                continue
            docs = search(q)
            for i, doc in enumerate(docs):
                print(f"\n── 结果 {i + 1} ──")
                print(f"来源: {doc.metadata.get('source', '?')}")
                print(doc.page_content[:500])
            print(f"\n共返回 {len(docs)} 个文档\n")
