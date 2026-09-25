"""progress.py —— 统一中文进度：阶段立即显示，耗时阶段每 20 秒报告仍在等待。
只由命令行入口启用；业务模块不配置全局日志，不记录密钥、请求地址或用户资料。
后台线程只报告时间，不执行任务；离开阶段时停止线程，异常仍交给原调用方处理。
"""
import logging
import sys
import threading
import time
from contextlib import contextmanager

logger = logging.getLogger("rag_agent.progress")
logger.addHandler(logging.NullHandler())
logger.propagate = False


def configure_progress():
    """为三个命令行入口启用同一格式；重复调用不会重复添加输出处理器。"""
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def report(message):
    """立即输出已确认的状态；调用方只能传入可公开的阶段、文件名和数量。"""
    logger.info(message)


@contextmanager
def stage(label, interval=20):
    """显示开始、周期等待与完成耗时；异常只显示阶段失败，不打印异常内的敏感信息。"""
    started = time.monotonic()
    stop = threading.Event()

    def heartbeat():
        """等待可中断事件，避免任务完成后线程仍持续打印。"""
        while not stop.wait(interval):
            report(f"[等待] {label}，已用时 {time.monotonic() - started:.0f} 秒")

    report(f"[开始] {label}")
    worker = None
    if logger.isEnabledFor(logging.INFO):
        worker = threading.Thread(target=heartbeat, daemon=True)
        worker.start()
    try:
        yield
    except BaseException:
        stop.set()
        if worker is not None:
            worker.join()
        report(f"[未完成] {label}，用时 {time.monotonic() - started:.1f} 秒")
        raise
    else:
        stop.set()
        if worker is not None:
            worker.join()
        report(f"[完成] {label}，用时 {time.monotonic() - started:.1f} 秒")
