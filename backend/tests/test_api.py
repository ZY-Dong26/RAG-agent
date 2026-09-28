"""FastAPI 适配层离线测试：假服务替代模型和云端客户端。"""
import sys
import json
import tempfile
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
        self.fail_after_delta = False
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

    def ask_stream(self, question):
        """模拟检索和两块生成；失败时先输出增量再报错，验证不会保存半截。"""
        result = self.ask(question)
        yield "status", {"stage": "retrieving", "message": "正在检索"}
        yield "retrieval", {"hits": result.hits, "timing": result.timing}
        yield "delta", {"text": "回答"}
        if self.fail_after_delta:
            raise RuntimeError("secret-token https://private.example/signed")
        yield "delta", {"text": "[1]"}
        yield "done", result


class ApiTests(unittest.TestCase):
    def setUp(self):
        """所有 API 测试使用临时 SQLite 文件，不写 backend/data/ 正式聊天记录。"""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.history_path = Path(self.temp_dir.name) / "chat.sqlite3"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_startup_status_and_ask_reuse_one_service(self):
        """启动只预热一次；接口返回公开引用和各阶段耗时。"""
        service = FakeService()
        calls = []
        def factory():
            calls.append(True)
            return service

        app = create_app(factory, history_path=self.history_path)
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
            self.assertTrue(data["conversation_id"])
            self.assertEqual(data["citations"][0]["source"], "a.pdf")
            self.assertEqual(data["citations"][0]["rank"], 1)
            self.assertNotIn("internal_path", data["citations"][0])
            self.assertEqual(data["timing"]["rerank_seconds"], .2)
        self.assertEqual(len(calls), 1)
        self.assertEqual(service.prepared, 1)

    def test_stream_events_save_only_complete_answer(self):
        """检索和增量按序到达；完成后才有会话 ID 和持久化消息。"""
        service = FakeService()
        with TestClient(create_app(lambda: service, history_path=self.history_path)) as client:
            response = client.post("/api/v1/ask/stream", json={"question": "流式问题"})
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.headers["content-type"].startswith("text/event-stream"))
            events = []
            for frame in response.text.strip().split("\n\n"):
                lines = frame.splitlines()
                events.append((lines[0][7:], json.loads(lines[1][6:])))
            self.assertEqual([name for name, _ in events],
                             ["status", "retrieval", "delta", "delta", "done"])
            self.assertEqual(events[1][1]["citations"][0]["source"], "a.pdf")
            self.assertNotIn("internal_path", events[1][1]["citations"][0])
            conversation_id = events[-1][1]["conversation_id"]
            detail = client.get(f"/api/v1/conversations/{conversation_id}").json()
            self.assertEqual([item["text"] for item in detail["messages"]],
                             ["流式问题", "回答[1]"])

    def test_stream_failure_does_not_save_partial_answer_or_expose_error(self):
        """模型在已输出增量后失败时，发送安全错误并释放忙碌锁。"""
        service = FakeService()
        service.fail_after_delta = True
        with TestClient(create_app(lambda: service, history_path=self.history_path)) as client:
            response = client.post("/api/v1/ask/stream", json={"question": "失败"})
            self.assertIn("event: delta", response.text)
            self.assertIn("event: error", response.text)
            self.assertNotIn("secret-token", response.text)
            self.assertEqual(client.get("/api/v1/conversations").json(), [])
            service.fail_after_delta = False
            self.assertEqual(client.post("/api/v1/ask", json={"question": "再次"}).status_code, 200)

    def test_blank_question_busy_and_error_are_safe(self):
        """无效输入不调用服务；忙碌与模型错误不泄漏内部异常。"""
        service = FakeService()
        app = create_app(lambda: service, history_path=self.history_path)
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
            self.assertEqual(client.get("/api/v1/conversations").json(), [])

    def test_history_survives_app_restart_and_delete_cascades(self):
        """首问创建、同会话续问、重建应用后读取快照，以及删除后的 404。"""
        first_service = FakeService()
        with TestClient(create_app(lambda: first_service, history_path=self.history_path)) as client:
            first = client.post("/api/v1/ask", json={"question": "第一题"}).json()
            conversation_id = first["conversation_id"]
            second = client.post("/api/v1/ask", json={
                "question": "第二题", "conversation_id": conversation_id,
            }).json()
            self.assertEqual(second["conversation_id"], conversation_id)
            summaries = client.get("/api/v1/conversations").json()
            self.assertEqual(len(summaries), 1)
            self.assertEqual(summaries[0]["title"], "第一题")

        # 新应用实例模拟页面刷新和后端重启，读取同一临时数据库。
        with TestClient(create_app(FakeService, history_path=self.history_path)) as client:
            detail = client.get(f"/api/v1/conversations/{conversation_id}")
            self.assertEqual(detail.status_code, 200)
            messages = detail.json()["messages"]
            self.assertEqual([message["role"] for message in messages],
                             ["user", "assistant", "user", "assistant"])
            self.assertEqual(messages[0]["text"], "第一题")
            self.assertIsNone(messages[0]["citations"])
            self.assertEqual(messages[1]["citations"][0]["source"], "a.pdf")
            self.assertEqual(messages[1]["timing"]["rerank_seconds"], .2)
            self.assertFalse(messages[1]["threshold_calibrated"])
            self.assertEqual(client.delete(f"/api/v1/conversations/{conversation_id}").status_code, 204)
            self.assertEqual(client.get("/api/v1/conversations").json(), [])
            self.assertEqual(client.get(f"/api/v1/conversations/{conversation_id}").status_code, 404)
            self.assertEqual(client.delete(f"/api/v1/conversations/{conversation_id}").status_code, 404)

    def test_failed_ask_and_unknown_conversation_do_not_save_messages(self):
        """模型失败不建空会话；旧会话失败不留下半组消息。"""
        service = FakeService()
        with TestClient(create_app(lambda: service, history_path=self.history_path)) as client:
            unknown = client.post("/api/v1/ask", json={
                "question": "问题", "conversation_id": "missing",
            })
            self.assertEqual(unknown.status_code, 404)
            self.assertEqual(service.questions, [])
            service.fail = True
            self.assertEqual(client.post("/api/v1/ask", json={"question": "失败"}).status_code, 500)
            self.assertEqual(client.get("/api/v1/conversations").json(), [])
            service.fail = False
            conversation_id = client.post("/api/v1/ask", json={"question": "成功"}).json()["conversation_id"]
            service.fail = True
            failed = client.post("/api/v1/ask", json={
                "question": "第二轮失败", "conversation_id": conversation_id,
            })
            self.assertEqual(failed.status_code, 500)
            messages = client.get(f"/api/v1/conversations/{conversation_id}").json()["messages"]
            self.assertEqual(len(messages), 2)

    def test_delete_waits_for_active_ask(self):
        """删除与问答共用锁，忙碌时保留聊天记录并返回可重试状态。"""
        with TestClient(create_app(FakeService, history_path=self.history_path)) as client:
            conversation_id = client.post("/api/v1/ask", json={"question": "问题"}).json()["conversation_id"]
            app = client.app
            self.assertTrue(app.state.ask_lock.acquire(blocking=False))
            try:
                response = client.delete(f"/api/v1/conversations/{conversation_id}")
                self.assertEqual(response.status_code, 429)
            finally:
                app.state.ask_lock.release()
            self.assertEqual(len(client.get("/api/v1/conversations").json()), 1)

    def test_startup_failure_does_not_expose_exception(self):
        """初始化失败不接收请求，也不输出异常原文。"""
        def factory():
            raise RuntimeError("secret-token")
        app = create_app(factory, history_path=self.history_path)
        with self.assertRaisesRegex(RuntimeError, "RAG 服务初始化失败"):
            with TestClient(app):
                pass


if __name__ == "__main__":
    unittest.main()
