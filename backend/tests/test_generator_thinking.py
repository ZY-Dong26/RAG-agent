"""回答模型思考参数的离线测试：仅核对构造出的请求，不访问云端。"""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_agent import config
from rag_agent.qa.generator import Generator


class FakeCompletions:
    def __init__(self):
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="回答"))],
                               usage=None)


class GeneratorThinkingTests(unittest.TestCase):
    def test_dashscope_qwen_disables_thinking_by_default(self):
        """百炼 Qwen3.x 显式传关闭思考，避免服务端默认启用。"""
        completions = FakeCompletions()
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        with patch.object(config, "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"), \
             patch.object(config, "LLM_ENABLE_THINKING", False):
            generator = Generator(client=client, model="qwen3.8-27b")
            generator.generate("问题", [])
            self.assertEqual(completions.requests[0]["extra_body"], {"enable_thinking": False})

    def test_thinking_can_be_enabled_explicitly(self):
        """配置为 true 时向百炼传开启思考。"""
        completions = FakeCompletions()
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        with patch.object(config, "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"):
            Generator(client=client, model="qwen3.8-27b", enable_thinking=True).generate("问题", [])
        self.assertEqual(completions.requests[0]["extra_body"], {"enable_thinking": True})

    def test_other_provider_receives_no_dashscope_parameter(self):
        """非百炼兼容接口不发送供应商扩展参数。"""
        completions = FakeCompletions()
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        with patch.object(config, "LLM_BASE_URL", "https://api.example.com/v1"):
            Generator(client=client, model="qwen3.8-27b").generate("问题", [])
        self.assertNotIn("extra_body", completions.requests[0])


if __name__ == "__main__":
    unittest.main()
