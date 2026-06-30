"""
loader 模块测试

运行:
    python -m pytest tests/test_loader.py -v
    或
    python tests/test_loader.py
"""

import hashlib
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.loader import (
    clean_academic_markdown,
    check_md5_hex,
    save_md5_hex,
    get_file_documents,
    txt_loader,
)
from rag import config


# ============================================================
# clean_academic_markdown — 学术论文清洗
# ============================================================

def test_clean_cuts_after_references():
    """参考文献之后的内容被截断"""
    text = "# 摘要\n这是摘要内容。\n\n# 参考文献\n[1] 张三. 某论文.\n[2] 李四. 某书."
    cleaned = clean_academic_markdown(text)
    assert "参考文献" not in cleaned
    assert "张三" not in cleaned
    assert "摘要内容" in cleaned


def test_clean_cuts_after_english_references():
    """英文 References 也能截断"""
    text = "# Abstract\nContent here.\n\n# References\n[1] Author. Title."
    cleaned = clean_academic_markdown(text)
    assert "References" not in cleaned
    assert "Content here" in cleaned


def test_clean_drops_before_abstract():
    """摘要之前的封面噪声被丢弃"""
    text = "中图分类号: TP311\nDOI: 12345\n\n# 摘要\n这是正文内容。"
    cleaned = clean_academic_markdown(text)
    assert "TP311" not in cleaned
    assert "DOI" not in cleaned
    assert "正文内容" in cleaned


def test_clean_filters_matrix_debris():
    """PDF 转换产生的矩阵碎屑被过滤"""
    text = "# 摘要\n正常内容。\n; ; ; ; ;\n0 B @\n__________\n继续正文。"
    cleaned = clean_academic_markdown(text)
    assert ";" not in cleaned
    assert "0 B @" not in cleaned
    assert "正常内容" in cleaned
    assert "继续正文" in cleaned


def test_clean_merges_hard_line_breaks():
    """中英文混合硬换行被智能贴合"""
    text = "# 摘要\nThis is a very long\nsentence that wraps.\n中文段落也\n被断开了。"
    cleaned = clean_academic_markdown(text)
    # 英文间断开的地方被空格连接
    assert "very long sentence" in cleaned or "This is a very long" in cleaned


def test_clean_removes_spaces_between_chinese():
    """中文字间的碎空格被消除"""
    text = "# 摘要\n三 维 空 间 拓 扑 关 系"
    cleaned = clean_academic_markdown(text)
    assert "三维空间拓扑关系" in cleaned


def test_clean_empty_input():
    """空字符串返回空"""
    assert clean_academic_markdown("") == ""


# ============================================================
# check_md5_hex / save_md5_hex — MD5 去重
# ============================================================

def test_md5_check_and_save():
    """保存 MD5 后再次检查应返回 True"""
    # 用临时文件替代配置路径
    original_path = config.MD5_HEX_STORE
    try:
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            config.MD5_HEX_STORE = f.name

        test_md5 = hashlib.md5(b"test_file_content").hexdigest()
        assert check_md5_hex(test_md5) is False   # 第一次检查，不存在
        save_md5_hex(test_md5)
        assert check_md5_hex(test_md5) is True    # 保存后应找到
    finally:
        config.MD5_HEX_STORE = original_path


def test_md5_not_found_for_unknown():
    """未保存的 MD5 返回 False"""
    assert check_md5_hex("a" * 32) is False


# ============================================================
# txt_loader — 文本文件加载
# ============================================================

def test_txt_loader_reads_file():
    """正常读取 .txt 文件并分割"""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write("第一段内容在这里。\n\n第二段内容更多一些文字。\n\n第三段。")
        tmp_path = f.name

    try:
        docs = txt_loader(tmp_path)
        assert len(docs) >= 1
        assert docs[0].metadata["source"].endswith(".txt")
        assert docs[0].metadata["category"] == "text_file"
        combined = "".join(d.page_content for d in docs)
        assert "第一段" in combined
    finally:
        os.unlink(tmp_path)


def test_txt_loader_empty_file():
    """空文件返回空列表"""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write("")
        tmp_path = f.name

    try:
        docs = txt_loader(tmp_path)
        assert docs == []
    finally:
        os.unlink(tmp_path)


# ============================================================
# get_file_documents — 文件类型分发
# ============================================================

def test_get_file_documents_txt():
    """识别 .txt 文件并调用 txt_loader"""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write("测试内容。")
        tmp_path = f.name

    try:
        docs = get_file_documents(tmp_path)
        assert len(docs) >= 1
        assert docs[0].metadata["category"] == "text_file"
    finally:
        os.unlink(tmp_path)


def test_get_file_documents_unknown_extension():
    """不支持的扩展名返回空列表"""
    docs = get_file_documents("test.docx")
    assert docs == []


# ============================================================
# 直接运行
# ============================================================
if __name__ == "__main__":
    tests = [
        test_clean_cuts_after_references,
        test_clean_cuts_after_english_references,
        test_clean_drops_before_abstract,
        test_clean_filters_matrix_debris,
        test_clean_merges_hard_line_breaks,
        test_clean_removes_spaces_between_chinese,
        test_clean_empty_input,
        test_md5_check_and_save,
        test_md5_not_found_for_unknown,
        test_txt_loader_reads_file,
        test_txt_loader_empty_file,
        test_get_file_documents_txt,
        test_get_file_documents_unknown_extension,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  OK  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} 通过")
