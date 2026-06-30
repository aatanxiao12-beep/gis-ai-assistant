import logging
import os
from datetime import datetime
from utils.path_tool import get_abs_path


class LoggerFactory:
    """简单的日志工厂类，负责统一管理和生产项目日志"""

    # 统一的日志保存根目录
    LOG_ROOT = get_abs_path('log')
    os.makedirs(LOG_ROOT, exist_ok=True)

    # 统一的标准日志格式
    LOG_FORMAT = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )

    # 缓存已生成的 Logger，确保同一个名字不重复创建 Handler
    _loggers = {}

    @classmethod
    def get_logger(
            cls,
            name: str = "agent",
            console_level: int = logging.INFO,
            file_level: int = logging.DEBUG,
            log_file: str = None
    ) -> logging.Logger:
        """
        工厂核心方法：根据名称获取或创建 Logger。
        相同名字的 Logger 再次调用时会直接复用，防止日志重复打印。
        """
        # 1. 如果缓存里有且已经配置过，直接返回
        if name in cls._loggers and cls._loggers[name].handlers:
            return cls._loggers[name]

        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)  # 总门槛设为最低，具体过滤由 Handler 决定

        # 2. 双重防御：防止底层 logging 模块原生缓存导致的 Handler 重复添加
        if logger.handlers:
            cls._loggers[name] = logger
            return logger

        # 3. 配置控制台 Handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_handler.setFormatter(cls.LOG_FORMAT)
        logger.addHandler(console_handler)

        # 4. 配置标准文件 Handler
        if not log_file:
            current_date = datetime.now().strftime('%Y%m%d')
            log_file = os.path.join(cls.LOG_ROOT, f"{name}_{current_date}.log")

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(file_level)
        file_handler.setFormatter(cls.LOG_FORMAT)
        logger.addHandler(file_handler)

        # 5. 注册到工厂缓存
        cls._loggers[name] = logger
        return logger


# --- 默认导出的常用实例 ---
logger = LoggerFactory.get_logger("agent")

if __name__ == '__main__':
    # 测试默认导出的 agent 日志
    logger.info("这是一条普通的常规日志")
    logger.error("这是一条错误级别的日志")

    # 测试工厂模式创建新模块日志，名字不同即完全隔离
    rag_logger = LoggerFactory.get_logger("rag", console_level=logging.DEBUG)
    rag_logger.debug("这是 RAG 模块的专属调试信息")