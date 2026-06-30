"""
嵌入模型工厂

用法:
    from rag.embedding import get_embeddings
    emb = get_embeddings()
    vectors = emb.embed_documents(["文本1", "文本2"])
"""

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.embeddings import Embeddings

from rag import config


class EmbeddingsFactory:
    """嵌入模型工厂"""

    def generator(self) -> Embeddings:
        return DashScopeEmbeddings(
            model=config.EMBEDDING_MODEL_NAME,
            dashscope_api_key=config.EMBEDDING_API_KEY,
        )


def get_embeddings() -> Embeddings:
    """便捷函数，供其他模块调用"""
    return EmbeddingsFactory().generator()
