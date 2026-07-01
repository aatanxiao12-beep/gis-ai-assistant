"""
ChromaDB 向量库

用法:
    from rag.vectordb import get_vector_store, add_documents
    store = get_vector_store()
    add_documents(store, docs)
"""

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag import config
from rag.embedding import get_embeddings
from utils.path_tool import get_abs_path
from utils.logger_handle import logger

# 单例缓存
_store: Chroma | None = None


def get_vector_store(embeddings: Embeddings | None = None) -> Chroma:
    """获取 Chroma 向量库单例"""
    global _store
    if _store is not None:
        return _store

    emb = embeddings or get_embeddings()
    persist_dir = get_abs_path(config.CHROMA_PERSIST_DIR)

    logger.info(
        "加载 Chroma | collection=%s | path=%s | metric=%s",
        config.CHROMA_COLLECTION_NAME, persist_dir, config.CHROMA_DISTANCE_METRIC,
    )

    _store = Chroma(
        collection_name=config.CHROMA_COLLECTION_NAME,
        embedding_function=emb,
        persist_directory=persist_dir,
        collection_metadata={"hnsw:space": config.CHROMA_DISTANCE_METRIC},
    )
    return _store


def reset_vector_store() -> None:
    """重置单例"""
    global _store
    _store = None


def add_documents(store: Chroma, documents: list[Document]) -> list[str]:
    """将文档列表写入向量库，返回文档 ID 列表"""
    ids = store.add_documents(documents)
    logger.info("写入 %d 个文档到集合 %s", len(documents), config.CHROMA_COLLECTION_NAME)
    return ids


def clear_collection(store: Chroma) -> None:
    """清空集合"""
    store.delete_collection()
    reset_vector_store()
    logger.warning("集合 %s 已清空", config.CHROMA_COLLECTION_NAME)


def get_existing_file_hashes(store: Chroma) -> dict[str, str]:
    """返回 {文件名: file_hash} 映射"""
    try:
        results = store.get()
        if not results or not results["metadatas"]:
            return {}

        seen = {}
        for meta in results["metadatas"]:
            src = meta.get("source", "")
            fhash = meta.get("file_hash", "")
            if src and fhash and src not in seen:
                seen[src] = fhash
        return seen
    except Exception:
        return {}


def delete_by_source(store: Chroma, source: str) -> int:
    """删除指定来源的所有文档块"""
    results = store.get(where={"source": source})
    if not results or not results["ids"]:
        return 0

    ids = results["ids"]
    store.delete(ids=ids)
    logger.info("删除旧文档: %s (%d 块)", source, len(ids))
    return len(ids)
