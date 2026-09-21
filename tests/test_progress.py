"""进度提示离线测试：验证等待可见、线程会退出，并且不会输出异常携带的秘密。"""
import io
import logging
import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rag_agent.common import progress


class ProgressTests(unittest.TestCase):
    """只测试输出与线程生命周期，不调用云端或真实模型。"""

    def test_wait_and_completion_stop_worker(self):
        """阶段等待可被观察到；退出后后台提示线程已停止。"""
        messages = []
        observed = threading.Event()
        workers = []

        def record(message):
            """收集提示，并用事件等待第一条心跳，避免依赖固定睡眠。"""
            messages.append(message)
            if message.startswith("[等待]"):
                workers.append(threading.current_thread())
                observed.set()

        with patch.object(progress.logger, "isEnabledFor", return_value=True), patch.object(progress, "report", side_effect=record):
            with progress.stage("测试阶段", interval=0.01):
                self.assertTrue(observed.wait(2))
        self.assertTrue(messages[0].startswith("[开始]"))
        self.assertTrue(messages[-1].startswith("[完成]"))
        self.assertTrue(all(not worker.is_alive() for worker in workers))

    def test_failure_does_not_log_exception_details(self):
        """失败仍向上传递原异常，但进度提示不会打印异常里的密钥或地址。"""
        with patch.object(progress, "report") as output:
            with self.assertRaisesRegex(ValueError, "secret"):
                with progress.stage("生成回答"):
                    raise ValueError("secret https://private.example")
        messages = " ".join(call.args[0] for call in output.call_args_list)
        self.assertIn("[未完成]", messages)
        self.assertNotIn("secret", messages)
        self.assertNotIn("[完成]", messages)

    def test_configuration_does_not_duplicate_handlers(self):
        """重复初始化入口日志时，每条消息仍只输出一次。"""
        stream = io.StringIO()
        with patch.object(progress.logger, "handlers", []), patch.object(progress.logger, "level", logging.WARNING), patch.object(sys, "stdout", stream):
            progress.configure_progress()
            progress.configure_progress()
            progress.report("唯一提示")
        self.assertEqual(stream.getvalue().count("唯一提示"), 1)
