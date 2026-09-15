# -*- coding: utf-8 -*-
"""
utils/logger.py —— 统一日志模块
================================
职责：
  1. 同时向 控制台 与 日志文件 输出日志（文件带 UTF-8 编码，中文不乱码）；
  2. 全局单例：同名 logger 重复获取不会重复添加 handler；
  3. 训练、评估、UI 各模块统一调用，方便课程验收时追溯运行记录。
"""

import logging
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

_LOCK = threading.Lock()
_CONFIGURED = set()

# 日志格式：时间 | 级别 | 模块名 | 内容
_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name="paddle_cifar", log_file=None, level=logging.INFO):
    """获取（或创建）一个同时写控制台与文件的 logger。

    Args:
        name (str): logger 名称，不同模块可用不同名称。
        log_file (str|None): 日志文件路径；None 时默认写到
            outputs/logs/run.log。传 False 表示只写控制台不写文件
            （供 pytest 等场景使用，避免测试产生日志文件）。
        level (int): 日志级别。

    Returns:
        logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    with _LOCK:
        # 同一 logger 已配置过则直接复用，防止 handler 重复导致的重复输出
        if name in _CONFIGURED:
            return logger

        fmt = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

        # --- 控制台输出（强制 UTF-8，避免 Windows GBK 控制台中文乱码） ---
        console = logging.StreamHandler(stream=sys.stdout)
        console.setFormatter(fmt)
        try:
            console.stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001 非 TextIO 流可能不支持 reconfigure
            pass
        logger.addHandler(console)

        # --- 文件输出 ---
        if log_file is not False:
            if log_file is None:
                log_file = os.path.join(config.LOG_DIR, "run.log")
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)

        _CONFIGURED.add(name)
        return logger


if __name__ == "__main__":
    log = get_logger("selftest")
    log.info("中文日志测试：训练启动")
    log.warning("这是一条警告")
    log.error("这是一条错误")
    print("日志文件位置:", os.path.join(config.LOG_DIR, "run.log"))
