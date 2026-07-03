"""
GIS AI Standard Assistant — HTTP 服务启动入口

用法:
    python run_server.py
"""

import uvicorn
from server.config import HOST, PORT

if __name__ == "__main__":
    uvicorn.run("server.main:app", host=HOST, port=PORT, reload=True)
