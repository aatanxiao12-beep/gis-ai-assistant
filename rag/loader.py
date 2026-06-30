import hashlib
import os
import re
from pathlib import Path
from langchain_core.documents import Document
import pymupdf4llm

from rag import config
from rag.splitter import split_by_markdown, split_text
from utils.path_tool import get_abs_path

DATA_DIR = Path(get_abs_path(config.DATA_DIR))


def clean_academic_markdown(raw_md: str) -> str:
    """极简核心清洗：熔断头尾噪声，抚平硬换行，消灭矩阵乱码"""
    if not raw_md:
        return ""

    # 1. 强力尾部熔断：切掉【参考文献】及其往后的所有内容
    raw_md = re.split(
        r"\n#+\s*(?:参考文献|References)\s*\n", raw_md, flags=re.IGNORECASE
    )[0]

    # 2. 强力头部熔断：从【摘要】开始保留，前面的封面噪声、中图分类号等一律丢弃
    abstract_match = re.search(
        r"\n#*\s*(?:摘要|Abstract)[:：\s]", raw_md, flags=re.IGNORECASE
    )
    if abstract_match:
        raw_md = raw_md[abstract_match.start() :]

    # 3. 行级精细清洗与段落展平
    lines = raw_md.split("\n")
    cleaned_blocks = []
    current_paragraph = []

    for line in lines:
        line_strip = line.strip()
        if not line_strip:
            if current_paragraph:
                # 拼合缓存的普通正文行
                cleaned_blocks.append("".join(current_paragraph))
                current_paragraph = []
            continue

        # 过滤 PDF 转换为 Markdown 时产生的矩阵碎屑和无效符号（如 ; ; ; 0 B @）
        if (
            line_strip.count(";") > 2
            or "0 B @" in line_strip
            or line_strip.count("_") > 5
        ):
            continue

        # 判断是否为结构化行（标题、表格、列表）
        is_structural = (
            line_strip.startswith("#")
            or line_strip.startswith("|")
            or line_strip.startswith("- ")
            or re.match(r"^\d+\.\s", line_strip)
        )

        if is_structural:
            if current_paragraph:
                cleaned_blocks.append("".join(current_paragraph))
                current_paragraph = []
            cleaned_blocks.append(line_strip)
        else:
            # 中英文混合硬换行智能贴合：英文间补空格，中文直接粘合
            if current_paragraph:
                last_char = current_paragraph[-1][-1]
                first_char = line_strip[0]
                if re.match(r"[A-Za-z0-9]", last_char) and re.match(
                    r"[A-Za-z0-9]", first_char
                ):
                    current_paragraph.append(" " + line_strip)
                else:
                    current_paragraph.append(line_strip)
            else:
                current_paragraph.append(line_strip)

    if current_paragraph:
        cleaned_blocks.append("".join(current_paragraph))

    # 4. 消除中文字符间由于排版残留的碎空格（循环替换直到干净）
    final_md = "\n\n".join([b for b in cleaned_blocks if b])
    while True:
        prev = final_md
        final_md = re.sub(
            r"([\u4e00-\u9fa5])\s+([\u4e00-\u9fa5])", r"\1\2", final_md
        )
        if final_md == prev:
            break

    return final_md


def process_single_paper(pdf_path: Path) -> list[Document]:
    """处理单篇论文：全文本清洗 -> 基于标题切块 -> 隐式元数据封装"""
    try:
        # 直接提取整本书的连续 Markdown 文本，排版极佳
        raw_md = pymupdf4llm.to_markdown(str(pdf_path))

        # 深度清洗
        cleaned_md = clean_academic_markdown(raw_md)

        # 提取基础元数据
        year_match = re.search(r"\[(\d{4})\]|【(\d{4})】", pdf_path.name)
        year = (
            year_match.group(1) or year_match.group(2)
            if year_match
            else "未知年份"
        )
        author_match = re.search(r"_([^_ \.]+)\.pdf$", pdf_path.name)
        author = author_match.group(1) if author_match else "未知作者"

        # 计算文件 MD5，用于去重
        file_hash = hashlib.md5(pdf_path.read_bytes()).hexdigest()

        # 按章节标题切块
        chunks = split_by_markdown(cleaned_md)

        # 过大的块进一步切分（嵌入模型单次输入上限约 33000 字符）
        final_chunks = []
        for chunk in chunks:
            if len(chunk.page_content) > 25000:
                sub_chunks = split_text(
                    chunk.page_content,
                    metadata=chunk.metadata,
                )
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)

        # 将元数据全部封装在 metadata 字典中，正文（page_content）保持绝对纯净
        for chunk in final_chunks:
            chunk.metadata["source"] = pdf_path.name
            chunk.metadata["author"] = author
            chunk.metadata["year"] = year
            chunk.metadata["file_hash"] = file_hash
            chunk.metadata["category"] = "paper_chunk"

        print(
            f"  √ {pdf_path.name}：清洗完成，切分成 {len(final_chunks)} 个文本块。"
        )
        return final_chunks
    except Exception as e:
        print(f"  × {pdf_path.name} 处理失败: {e}")
        return []


# ============================================================
# MD5 去重：用本地文件记录已处理过的文件
# ============================================================

def _get_md5_store_path() -> str:
    path = get_abs_path(config.MD5_HEX_STORE)
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    return path


def check_md5_hex(md5_for_check: str) -> bool:
    """检查 MD5 是否已经处理过（在记录文件中存在）"""
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
    """将已处理的 MD5 追加写入记录文件"""
    with open(_get_md5_store_path(), "a", encoding="utf-8") as f:
        f.write(md5_for_check + "\n")


def clear_md5_store() -> None:
    """清空 MD5 记录文件"""
    open(_get_md5_store_path(), "w", encoding="utf-8").close()
    print("MD5 去重记录已清空")


# ============================================================
# 文件加载器（按扩展名分发）
# ============================================================

def txt_loader(file_path: str) -> list[Document]:
    """加载 .txt 文件，按字符递归分割"""
    text = Path(file_path).read_text(encoding="utf-8")
    if not text.strip():
        return []
    chunks = split_text(text, metadata={"source": Path(file_path).name, "category": "text_file"})
    return chunks


def pdf_loader(file_path: str) -> list[Document]:
    """加载 .pdf 文件"""
    return process_single_paper(Path(file_path))


def get_file_documents(read_path: str) -> list[Document]:
    """
    根据文件扩展名选择加载器。

    支持: .txt / .pdf
    其他扩展名返回空列表。
    """
    if read_path.endswith(".txt"):
        return txt_loader(read_path)
    if read_path.endswith(".pdf"):
        return pdf_loader(read_path)
    return []


# ============================================================
# 批量构建（带 MD5 去重）
# ============================================================

def build_survey_knowledge_base() -> list[Document]:
    """
    遍历 data/ 目录，加载所有支持的文件并转为文档块。

    通过 MD5 去重：已处理过的文件直接跳过。
    """
    all_chunks = []
    extensions = ["*.pdf", "*.txt"]
    files = []
    for ext in extensions:
        files.extend(DATA_DIR.rglob(ext))

    print(f"扫描数据目录，共发现 {len(files)} 个文件")

    md5_store_name = os.path.basename(config.MD5_HEX_STORE)

    for file_path in files:
        if file_path.name == md5_store_name:
            continue   # 跳过 MD5 记录文件自身

        md5 = hashlib.md5(file_path.read_bytes()).hexdigest()

        if check_md5_hex(md5):
            print(f"  → 跳过已处理: {file_path.name}")
            continue

        chunks = get_file_documents(str(file_path))
        if chunks:
            all_chunks.extend(chunks)
            save_md5_hex(md5)
            print(f"  √ {file_path.name}: {len(chunks)} 个文档块")

    print(f"本次新增 {len(all_chunks)} 个文档块")
    return all_chunks


