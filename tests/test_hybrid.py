"""
混合检索测试（OGC GML 场景）

运行:
    python tests/test_hybrid.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document
from rag.hybrid_retriever import BM25Index, reciprocal_rank_fusion


def test_bm25_gml():
    """OGC GML 标准片段：验证 BM25 关键词命中"""
    docs = [
        Document(
            page_content=(
                '<xs:complexType name="PointType" abstract="false">'
                "gml:Point implements a single coordinate tuple. "
                "The point is defined by a direct position."
                "</xs:complexType>"
            ),
            metadata={"source": "geometryBasic0d1d.xsd", "component_type": "complexType"},
        ),
        Document(
            page_content=(
                '<xs:complexType name="CurveType" abstract="false">'
                "gml:Curve is a 1-dimensional primitive. "
                "Curves are continuous lines composed of curve segments."
                "</xs:complexType>"
            ),
            metadata={"source": "geometryBasic0d1d.xsd", "component_type": "complexType"},
        ),
        Document(
            page_content=(
                '<xs:complexType name="SurfaceType" abstract="false">'
                "gml:Surface is a 2-dimensional primitive. "
                "A Surface is composed of surface patches."
                "</xs:complexType>"
            ),
            metadata={"source": "geometryPrimitives.xsd", "component_type": "complexType"},
        ),
        Document(
            page_content=(
                '<xs:element name="AbstractFeature" type="gml:AbstractFeatureType" abstract="true">'
                "The basic feature model is based on the gml:AbstractFeatureType. "
                "All GML features derive from this abstract type."
                "</xs:element>"
            ),
            metadata={"source": "feature.xsd", "component_type": "element"},
        ),
        Document(
            page_content=(
                "GML topology defines spatial relationships. "
                "TopoPoint, TopoCurve, TopoSurface, and TopoSolid represent "
                "topological primitives that describe adjacency and connectivity."
            ),
            metadata={"source": "topology.xsd", "component_type": "complexType"},
        ),
    ]

    bm25 = BM25Index()
    bm25.build(docs)

    # 查询1: 几何类型
    results = bm25.search("Curve segment", top_k=3)
    assert len(results) >= 1
    top = [doc.page_content[:30] for _, doc in results[:3]]
    print(f"  'Curve segment' → {[d[:40] for d in top]}")

    # 查询2: 要素模型
    results2 = bm25.search("AbstractFeatureType", top_k=2)
    assert len(results2) >= 1
    assert "AbstractFeatureType" in results2[0][1].page_content
    print(f"  'AbstractFeatureType' → {results2[0][1].metadata['source']}")

    # 查询3: 拓扑
    results3 = bm25.search("topology primitive adjacency", top_k=3)
    assert len(results3) >= 1
    top3_doc = results3[0][1]
    assert "topology" in top3_doc.page_content.lower()
    print(f"  'topology primitive adjacency' → {top3_doc.metadata['source']}")

    print("  BM25 全部断言通过")


def test_rrf_fusion():
    """RRF 融合：稠密偏向语义、稀疏偏向关键词"""
    docs = [
        Document(page_content="gml:Point implements a single coordinate tuple.", metadata={"source": "gml_point"}),
        Document(page_content="gml:Curve is composed of curve segments.", metadata={"source": "gml_curve"}),
        Document(page_content="gml:Surface is composed of surface patches.", metadata={"source": "gml_surface"}),
        Document(page_content="TopoSurface represents topological surface primitives.", metadata={"source": "gml_topo"}),
    ]

    # 模拟稠密检索排位（语义相似度：Curve 和 Surface 都涉及几何原语）
    dense = [docs[1], docs[2], docs[0]]

    # 模拟稀疏检索排位（关键词命中：topology 查询命中 TopoSurface）
    sparse = [docs[3], docs[2], docs[1]]

    fused = reciprocal_rank_fusion([dense, sparse], k=60)

    # RRF 融合后，两个列表中排位靠前的文档都会出现
    sources = {d.metadata["source"] for d in fused}
    assert "gml_curve" in sources       # dense 中排第一
    assert "gml_topo" in sources        # sparse 中排第一
    print(f"  RRF 融合: dense={[d.metadata['source'] for d in dense]} + sparse={[d.metadata['source'] for d in sparse]}")
    print(f"  → fused={[d.metadata['source'] for d in fused]}")
    print("  RRF 全部断言通过")


if __name__ == "__main__":
    test_bm25_gml()
    test_rrf_fusion()
    print("\n测试通过")
