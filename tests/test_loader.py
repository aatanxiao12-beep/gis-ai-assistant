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
    clean_ogc_standard,
    check_md5_hex,
    save_md5_hex,
    get_file_documents,
    txt_loader,
)
from rag import config


# ============================================================
# clean_ogc_standard — OGC 标准清洗
# ============================================================

SAMPLE_OGC = """OGC 06-103r4

Copyright © 2007 Open Geospatial Consortium, Inc. All Rights Reserved.

i

Contents

1 Scope

This standard defines the Geography Markup Language.

2 Conformance

The conformance requirements are as follows.

3 Normative References

The following normative documents contain provisions.
"""


def test_ogc_clean_cuts_before_scope():
    """从 "1 Scope" 开始保留正文，砍掉前置的目录/版权"""
    cleaned = clean_ogc_standard(SAMPLE_OGC)
    assert "Copyright" not in cleaned
    assert "OGC 06-103r4" not in cleaned
    assert "Contents" not in cleaned
    assert "This standard defines" in cleaned


def test_ogc_clean_preserves_xml():
    """XML/XSD 代码块保留原始缩进"""
    text = "1 Scope\nScope text here.\n<xs:element name=\"test\">\n  <xs:annotation>\n    <xs:documentation>desc</xs:documentation>\n  </xs:annotation>\n</xs:element>"
    cleaned = clean_ogc_standard(text)
    assert "<xs:element name=\"test\">" in cleaned


def test_ogc_clean_filters_roman_page_numbers():
    """孤立罗马数字页码被过滤"""
    text = "1 Scope\nScope text.\niv\n\n2 Conformance\nConformance text.\nviii"
    cleaned = clean_ogc_standard(text)
    assert "iv" not in cleaned
    assert "viii" not in cleaned


def test_ogc_clean_empty_input():
    """空字符串返回空"""
    assert clean_ogc_standard("") == ""


# ============================================================
# check_md5_hex / save_md5_hex — MD5 去重
# ============================================================

def test_md5_check_and_save():
    """保存 MD5 后再次检查应返回 True"""
    original_path = config.MD5_HEX_STORE
    try:
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            config.MD5_HEX_STORE = f.name

        test_md5 = hashlib.md5(b"test_file_content").hexdigest()
        assert check_md5_hex(test_md5) is False
        save_md5_hex(test_md5)
        assert check_md5_hex(test_md5) is True
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
    """识别 .txt 文件"""
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
        test_ogc_clean_cuts_before_scope,
        test_ogc_clean_preserves_xml,
        test_ogc_clean_filters_roman_page_numbers,
        test_ogc_clean_empty_input,
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
