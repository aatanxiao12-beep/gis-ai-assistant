"""
文档加载与分割

支持三类文件:
- OGC 标准 PDF  →  条款级清洗 → 按 Clause 编号切块
- XSD Schema    →  XML 解析 → 按 complexType/simpleType/element 切块
- TXT 纯文本    →  递归字符分割

同时提供 MD5 文件级去重。
"""

import hashlib
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pymupdf4llm

from rag import config
from utils.path_tool import get_abs_path

DATA_DIR = Path(get_abs_path(config.DATA_DIR))


# ============================================================
# 通用分割器
# ============================================================

def split_text(
    text: str,
    metadata: dict | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """递归字符分割，适合纯文本"""
    cs = chunk_size or config.CHUNK_SIZE
    co = chunk_overlap or config.CHUNK_OVERLAP
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cs, chunk_overlap=co,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", ".", "!", "?", ";", ",", " "],
        keep_separator=True,
    )
    return splitter.create_documents([text], metadatas=[metadata or {}])


# ============================================================
# OGC 标准 PDF 清洗
# ============================================================

def clean_ogc_standard(raw_text: str) -> str:
    """
    OGC 标准文档清洗:
    - 从 "1 Scope" 开始保留正文，砍掉前言/目录/版权
    - 抚平英文硬换行
    - 精准剔除页脚版权、文档编号、孤立页码
    - 保留 XML/XSD 代码块的原始缩进和换行
    """
    if not raw_text:
        return ""

    scope_match = re.search(r"\n1\s+Scope\s*\n", raw_text)
    if scope_match:
        raw_text = raw_text[scope_match.start():]

    lines = raw_text.split("\n")
    cleaned_lines = []
    current_paragraph = []

    for line in lines:
        line_strip = line.strip()
        if not line_strip:
            if current_paragraph:
                cleaned_lines.append("".join(current_paragraph))
                current_paragraph = []
            continue

        # 过滤页脚噪声
        if "Copyright" in line_strip and "Open Geospatial" in line_strip:
            continue
        if re.match(r"^OGC\s+\d{2}-\d+", line_strip):
            continue
        if re.match(r"^(?:[ivxlcdm]+|\d+)\s*$", line_strip, re.IGNORECASE):
            continue

        # 结构化行保留换行
        is_structural = (
            re.match(r"^\d+(?:\.\d+)*\s+[A-Z]", line_strip)  # 条款标题
            or line_strip.startswith("<")                      # XML 标签
            or line_strip.startswith("|")                      # 表格
        )

        if is_structural:
            if current_paragraph:
                cleaned_lines.append("".join(current_paragraph))
                current_paragraph = []
            cleaned_lines.append(line_strip)
        else:
            if current_paragraph:
                current_paragraph.append(" " + line_strip)
            else:
                current_paragraph.append(line_strip)

    if current_paragraph:
        cleaned_lines.append("".join(current_paragraph))

    return "\n\n".join([l for l in cleaned_lines if l])


def _split_by_clauses(cleaned_text: str, doc_name: str) -> list[Document]:
    """按 Clause 条款大标题切分"""
    chunks = []
    clause_pattern = r"\n(?=\d+\s+[A-Z][a-zA-Z0-9_\s\d\:]+)"
    sections = re.split(clause_pattern, cleaned_text)

    for section in sections:
        section = section.strip()
        if not section:
            continue
        first_line = section.split("\n")[0]
        clause_name = first_line if re.match(r"^\d+", first_line) else "正文引言"
        chunks.append(Document(
            page_content=section,
            metadata={
                "source": doc_name, "clause": clause_name,
                "category": "ogc_standard",
            },
        ))
    return chunks


# DashScope text-embedding-v4 硬限制 8192 tokens，保守取 ~10000 字符安全值
MAX_CHUNK_LENGTH = 10000


def _process_ogc_pdf(pdf_path: Path) -> list[Document]:
    """处理 OGC 标准 PDF"""
    raw_text = pymupdf4llm.to_markdown(str(pdf_path))
    cleaned = clean_ogc_standard(raw_text)
    chunks = _split_by_clauses(cleaned, pdf_path.name)

    final_chunks = []
    for chunk in chunks:
        if len(chunk.page_content) > MAX_CHUNK_LENGTH:
            final_chunks.extend(split_text(chunk.page_content, metadata=chunk.metadata))
        else:
            final_chunks.append(chunk)
    return final_chunks


# ============================================================
# XSD Schema 加载
# ============================================================

XSD_NS = {
    'xs': 'http://www.w3.org/2001/XMLSchema',
    'gml': 'http://www.opengis.net/gml/3.2',
}

COMPONENT_TAGS = {'complexType', 'simpleType', 'element'}


def _process_xsd(file_path: Path) -> list[Document]:
    """解析 XSD，按 complexType / simpleType / element 拆分为独立组件"""
    try:
        tree = ET.parse(str(file_path))
        root = tree.getroot()
        target_ns = root.attrib.get('targetNamespace', 'unknown')
        raw_chunks = []

        for child in root:
            tag_local = child.tag.split('}')[-1] if '}' in child.tag else child.tag
            if tag_local not in COMPONENT_TAGS:
                continue

            component_name = child.attrib.get('name')
            if not component_name:
                continue

            code = ET.tostring(child, encoding='unicode').strip()
            raw_chunks.append(Document(
                page_content=code,
                metadata={
                    "source": file_path.name,
                    "component_name": component_name,
                    "component_type": tag_local,
                    "target_namespace": target_ns,
                    "category": "xsd_schema",
                },
            ))

        chunks = []
        for chunk in raw_chunks:
            if len(chunk.page_content) > MAX_CHUNK_LENGTH:
                chunks.extend(split_text(chunk.page_content, metadata=chunk.metadata))
            else:
                chunks.append(chunk)

        print(f"  √ {file_path.name}: 拆解出 {len(raw_chunks)} 个组件 → {len(chunks)} 个文档块")
        return chunks
    except Exception as e:
        print(f"  × {file_path.name} 解析失败: {e}")
        return []


# ============================================================
# MD5 去重
# ============================================================

def _get_md5_store_path() -> str:
    path = get_abs_path(config.MD5_HEX_STORE)
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    return path


def check_md5_hex(md5_for_check: str) -> bool:
    store_path = _get_md5_store_path()
    if not os.path.exists(store_path):
        open(store_path, "w", encoding="utf-8").close()
        return False
    with open(store_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip() == md5_for_check:
                return True
    return False


def save_md5_hex(md5_for_check: str) -> None:
    with open(_get_md5_store_path(), "a", encoding="utf-8") as f:
        f.write(md5_for_check + "\n")


def clear_md5_store() -> None:
    open(_get_md5_store_path(), "w", encoding="utf-8").close()
    print("MD5 去重记录已清空")


# ============================================================
# 文件类型分发
# ============================================================

def txt_loader(file_path: str) -> list[Document]:
    """读取纯文本文件，递归字符分割"""
    path = Path(file_path)
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []
    return split_text(text, metadata={"source": path.name, "category": "text_file"})


def get_file_documents(file_path: str) -> list[Document]:
    """
    根据扩展名选择加载器:
    - .pdf  → OGC 标准管线
    - .xsd  → XSD Schema 组件拆解
    - .txt  → 纯文本分割
    """
    path = Path(file_path)

    if path.suffix == ".pdf":
        return _process_ogc_pdf(path)

    if path.suffix == ".xsd":
        return _process_xsd(path)

    if path.suffix == ".txt":
        return txt_loader(file_path)

    return []


# ============================================================
# 批量构建（MD5 去重）
# ============================================================

def build_survey_knowledge_base() -> list[Document]:
    """遍历 data/ 目录，加载所有支持的文件，MD5 去重"""
    all_chunks = []
    extensions = ["*.pdf", "*.txt", "*.xsd"]
    files = []
    for ext in extensions:
        files.extend(DATA_DIR.rglob(ext))

    print(f"扫描数据目录，共发现 {len(files)} 个文件")

    md5_store_name = os.path.basename(config.MD5_HEX_STORE)

    for file_path in files:
        if file_path.name == md5_store_name:
            continue

        md5 = hashlib.md5(file_path.read_bytes()).hexdigest()
        if check_md5_hex(md5):
            print(f"  → 跳过已处理: {file_path.name}")
            continue

        try:
            chunks = get_file_documents(str(file_path))
        except Exception as e:
            print(f"  × {file_path.name} 处理失败: {e}")
            continue

        if chunks:
            all_chunks.extend(chunks)
            save_md5_hex(md5)
            print(f"  √ {file_path.name}: {len(chunks)} 个文档块")

    print(f"本次新增 {len(all_chunks)} 个文档块")
    return all_chunks
