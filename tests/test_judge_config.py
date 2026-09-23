"""独立裁判配置测试：验证三项服务配置与两项代码默认值。"""
import importlib
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from devtools.evaluation import judge_config
from rag_agent import config as rag_config


class JudgeConfigTests(unittest.TestCase):
    def tearDown(self):
        """恢复当前项目配置，避免模块级常量影响后续测试。"""
        importlib.reload(judge_config)

    def test_reads_three_service_values_from_project_env(self):
        """三个服务参数读取 `.env`，生成策略使用代码默认值。"""
        file_values = {
            "JUDGE_LLM_API_KEY": "file-key",
            "JUDGE_LLM_BASE_URL": "https://judge.example/v1",
            "JUDGE_LLM_MODEL": "strong-judge",
        }
        with patch.dict(os.environ, {}, clear=True), patch("dotenv.dotenv_values", return_value=file_values):
            importlib.reload(judge_config)
        self.assertEqual(judge_config.JUDGE_LLM_API_KEY, "file-key")
        self.assertEqual(judge_config.JUDGE_LLM_BASE_URL, "https://judge.example/v1")
        self.assertEqual(judge_config.JUDGE_LLM_MODEL, "strong-judge")
        self.assertEqual(judge_config.JUDGE_LLM_TEMPERATURE, 0)
        self.assertEqual(judge_config.JUDGE_LLM_MAX_TOKENS, 800)
        self.assertFalse(any(name.startswith("JUDGE_LLM_") for name in dir(rag_config)))

    def test_system_environment_wins_without_overriding_code_defaults(self):
        """系统环境变量覆盖三项服务配置，但不开放温度和输出长度环境变量。"""
        values = {
            "JUDGE_LLM_API_KEY": "env-key",
            "JUDGE_LLM_BASE_URL": "https://env.example/v1",
            "JUDGE_LLM_MODEL": "env-model",
            "JUDGE_LLM_TEMPERATURE": "0.9",
            "JUDGE_LLM_MAX_TOKENS": "9999",
        }
        with patch.dict(os.environ, values, clear=True), patch("dotenv.dotenv_values", return_value={}):
            importlib.reload(judge_config)
        self.assertEqual(judge_config.JUDGE_LLM_API_KEY, "env-key")
        self.assertEqual(judge_config.JUDGE_LLM_BASE_URL, "https://env.example/v1")
        self.assertEqual(judge_config.JUDGE_LLM_MODEL, "env-model")
        self.assertEqual(judge_config.JUDGE_LLM_TEMPERATURE, 0)
        self.assertEqual(judge_config.JUDGE_LLM_MAX_TOKENS, 800)


if __name__ == "__main__":
    unittest.main()
