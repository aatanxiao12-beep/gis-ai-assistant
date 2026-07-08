"""
Server 配置

环境变量优先级更高，方便部署时覆盖。
"""

import os
from dotenv import load_dotenv

load_dotenv()

HOST = os.getenv("SERVER_HOST", "0.0.0.0")
PORT = int(os.getenv("SERVER_PORT", "8000"))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")
