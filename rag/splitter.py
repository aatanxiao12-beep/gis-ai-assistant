"""
文本分割器

提供两种常用分割方式：
- split_text(): 递归字符分割（适合纯文本）
- split_by_markdown(): 按 H1/H2/H3 标题分割（适合 Markdown）

配置项 CHUNK_SIZE / CHUNK_OVERLAP 在 rag.config 中。
"""

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)

from rag import config
from utils.logger_handle import logger


def split_text(
    text: str,
    metadata: dict | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """
    将长文本按字符递归分割为语义块。

    Args:
        text: 原始文本
        metadata: 附加到每个块的元数据
        chunk_size: 块大小，默认 config.CHUNK_SIZE
        chunk_overlap: 块重叠，默认 config.CHUNK_OVERLAP
    """
    cs = chunk_size or config.CHUNK_SIZE
    co = chunk_overlap or config.CHUNK_OVERLAP

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cs,
        chunk_overlap=co,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", ".", "!", "?", ";", ",", " "],
        keep_separator=True,
    )

    chunks = splitter.create_documents([text], metadatas=[metadata or {}])
    logger.info("字符分割完成: %d 个块 (size=%d, overlap=%d)", len(chunks), cs, co)
    return chunks


def split_by_markdown(text: str) -> list[Document]:
    """按 Markdown 标题层级（H1/H2/H3）分割文本"""
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
        ],
        strip_headers=False,
    )
    chunks = splitter.split_text(text)
    logger.info("Markdown 标题分割完成: %d 个块", len(chunks))
    return chunks
