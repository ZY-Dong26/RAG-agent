"""问答应用服务测试：验证未来前端与命令行共用的稳定入口，不调用真实模型。"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_agent.qa.service import RAGService


class FakeRetriever:
    """返回测试预设命中，代替本地 Embedding 和 FAISS。"""
    def __init__(self, hits):
        """保存预设检索记录，后续直接返回。"""
        self.hits = hits

    def retrieve(self, question):
        """返回预设命中，隔离真实向量模型与检索索引。"""
        return self.hits


class FakeGenerator:
    """记录生成调用，代替云端 LLM。"""
    def __init__(self):
        """初始化调用记录，供测试断言生成器是否被调用。"""
        self.calls = []

    def generate(self, question, hits):
        """记录问题与命中资料，返回固定答案，隔离真实 LLM 调用。"""
        self.calls.append((question, hits))
        return "测试回答"


class RAGServiceTests(unittest.TestCase):
    """验证统一问答入口是否正确连接检索与生成，并保留引用信息。"""
    def test_answer_returns_text_and_sources(self):
        """有检索结果时返回生成答案，并把同一份引用记录交给调用界面。"""
        hits = [{"text": "证据", "metadata": {"source": "a.pdf", "page": 1}}]
        generator = FakeGenerator()
        service = RAGService(retriever=FakeRetriever(hits), generator=generator)
        result = service.ask("问题")
        self.assertEqual(result.answer, "测试回答")
        self.assertEqual(result.hits, hits)
        self.assertEqual(generator.calls, [("问题", hits)])

    def test_empty_retrieval_does_not_call_llm(self):
        """没有资料时返回空答案，避免前端请求触发无依据的 LLM 调用。"""
        generator = FakeGenerator()
        result = RAGService(retriever=FakeRetriever([]), generator=generator).ask("问题")
        self.assertIsNone(result.answer)
        self.assertEqual(result.hits, [])
        self.assertEqual(generator.calls, [])


if __name__ == "__main__":
    unittest.main()
