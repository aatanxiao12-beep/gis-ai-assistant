"""
RAG 模块配置

直接修改这里的变量即可切换模型、调整参数。
环境变量优先级更高，方便部署时覆盖。
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# 嵌入模型（阿里百炼 DashScope）
# ============================================================

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-v2")
EMBEDDING_API_KEY = os.getenv("DASHSCOPE_API_KEY", os.getenv("DEEPSEEK_API_KEY", ""))

# ============================================================
# 文本分割
# ============================================================

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ============================================================
# ChromaDB 向量库
# ============================================================

CHROMA_PERSIST_DIR = "vector_store/chroma_db"
CHROMA_COLLECTION_NAME = "gis_topology_knowledge"
CHROMA_DISTANCE_METRIC = "cosine"                 # cosine / l2 / ip

# ============================================================
# 文件处理 & 去重
# ============================================================

DATA_DIR = "data"                                  # 数据文件根目录
MD5_HEX_STORE = "data/md5_processed.txt"           # 记录已处理文件的 MD5

# ============================================================
# 检索
# ============================================================

RETRIEVAL_TOP_K = 10                              # 返回文档数
RETRIEVAL_SEARCH_TYPE = "similarity_score_threshold"  # similarity / mmr / similarity_score_threshold
RETRIEVAL_FETCH_K = 20                            # MMR 初选数
RETRIEVAL_LAMBDA_MULT = 0.6                       # MMR 多样性 (0→多样, 1→相似)
RETRIEVAL_SCORE_THRESHOLD = 0.5                  # 相似度阈值，低于此分数的文档被过滤

# ============================================================
# 重排序
# ============================================================

RERANK_METHOD = "cross_encoder"                  # cross_encoder / none
RERANK_MODEL_NAME = "Qwen/Qwen3-Reranker-0.6B"
RERANK_TOP_N = 5                                  # 重排后保留数
