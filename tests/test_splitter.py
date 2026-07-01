"""
splitter 模块测试

运行:
    python -m pytest tests/test_splitter.py -v
    或
    python tests/test_splitter.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.loader import split_text


# ============================================================
# split_text — 递归字符分割
# ============================================================

def test_split_text_basic():
    """基本分割：长中文文本被切成多块"""
    text = "第一段落。\n\n第二段落，包含一些内容。\n\n第三段落，继续延伸。"
    chunks = split_text(text, chunk_size=15, chunk_overlap=3)
    assert len(chunks) >= 2, f"期望至少2块，实际{len(chunks)}块"
    # 每块应该都在原文本中有对应
    combined = "".join(c.page_content for c in chunks)
    assert "第一段落" in combined


def test_split_text_short():
    """短文本不会被切碎"""
    text = "就一句话。"
    chunks = split_text(text, chunk_size=500, chunk_overlap=50)
    assert len(chunks) == 1
    assert chunks[0].page_content == text


def test_split_text_metadata_passed():
    """metadata 被正确传递到每个块"""
    text = "内容A。\n\n内容B。"
    meta = {"source": "测试.txt", "author": "张三"}
    chunks = split_text(text, metadata=meta, chunk_size=10, chunk_overlap=2)
    for chunk in chunks:
        assert chunk.metadata["source"] == "测试.txt"
        assert chunk.metadata["author"] == "张三"


def test_split_text_english():
    """英文文本按句子分割"""
    text = "Hello world. This is a test. Another sentence here."
    chunks = split_text(text, chunk_size=30, chunk_overlap=5)
    assert len(chunks) >= 1
    for chunk in chunks:
        assert isinstance(chunk.page_content, str)
        assert len(chunk.page_content) > 0


# ============================================================
# 直接运行
# ============================================================
if __name__ == "__main__":
    tests = [
        test_split_text_basic,
        test_split_text_short,
        test_split_text_metadata_passed,
        test_split_text_english,
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
