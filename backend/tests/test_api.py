"""FastAPI 适配层离线测试：假服务替代模型和云端客户端。"""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from api.app import create_app


class FakeService:
    def __init__(self):
        self.prepared = 0
        self.questions = []
        self.fail = False
        self.retriever = SimpleNamespace(store=SimpleNamespace(
            chunks=[{}, {}], index_path=Path("data/vector_db/generations/version-a/index.faiss")))
        self.reranker = SimpleNamespace(device="cuda")

    def prepare(self):
        self.prepared += 1

    def ask(self, question):
        self.questions.append(question)
        if self.fail:
            raise RuntimeError("secret-token https://private.example/signed")
        return SimpleNamespace(
            answer="回答[1]", answerable=True, reason="threshold_not_calibrated",
            evidence_status="answerable", threshold_calibrated=False,
            hits=[{"text": "资料", "metadata": {
                "rank": 1, "source": "C:/private/docs/a.pdf", "page": 3,
                "rerank_score": .8, "fusion_score": .03, "internal_path": "C:/secret",
            }}],
            timing={"recall_seconds": .1, "rerank_seconds": .2,
                    "retrieval_seconds": .3, "generation_seconds": .4,
                    "total_seconds": .7},
        )


class ApiTests(unittest.TestCase):
    def test_startup_status_and_ask_reuse_one_service(self):
        """启动只预热一次；接口返回公开引用和各阶段耗时。"""
        service = FakeService()
        calls = []
        def factory():
            calls.append(True)
            return service

        app = create_app(factory)
        with TestClient(app) as client:
            status = client.get("/api/v1/status")
            self.assertEqual(status.status_code, 200)
            self.assertEqual(status.json(), {
                "ready": True, "vector_count": 2, "index_version": "version-a",
                "reranker_enabled": True, "reranker_device": "cuda",
            })
            response = client.post("/api/v1/ask", json={"question": "  问题  "})
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(service.questions, ["问题"])
            self.assertEqual(data["answer"], "回答[1]")
            self.assertEqual(data["citations"][0]["source"], "a.pdf")
            self.assertEqual(data["citations"][0]["rank"], 1)
            self.assertNotIn("internal_path", data["citations"][0])
            self.assertEqual(data["timing"]["rerank_seconds"], .2)
        self.assertEqual(len(calls), 1)
        self.assertEqual(service.prepared, 1)

    def test_blank_question_busy_and_error_are_safe(self):
        """无效输入不调用服务；忙碌与模型错误不泄漏内部异常。"""
        service = FakeService()
        app = create_app(lambda: service)
        with TestClient(app) as client:
            self.assertEqual(client.post("/api/v1/ask", json={"question": "   "}).status_code, 422)
            self.assertEqual(service.questions, [])
            self.assertTrue(app.state.ask_lock.acquire(blocking=False))
            try:
                busy = client.post("/api/v1/ask", json={"question": "问题"})
                self.assertEqual(busy.status_code, 429)
            finally:
                app.state.ask_lock.release()
            service.fail = True
            failed = client.post("/api/v1/ask", json={"question": "问题"})
            self.assertEqual(failed.status_code, 500)
            self.assertNotIn("secret-token", failed.text)
            self.assertNotIn("https://private.example", failed.text)

    def test_startup_failure_does_not_expose_exception(self):
        """初始化失败不接收请求，也不输出异常原文。"""
        def factory():
            raise RuntimeError("secret-token")
        app = create_app(factory)
        with self.assertRaisesRegex(RuntimeError, "RAG 服务初始化失败"):
            with TestClient(app):
                pass


if __name__ == "__main__":
    unittest.main()
