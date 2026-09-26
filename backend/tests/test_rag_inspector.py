"""问答诊断工具的离线测试：验证请求记录、相邻片段定位与报告导出。"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import sys

# 调试实现与核心包并列放在 src，直接运行测试时也能找到它们。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from devtools.rag_inspector import RecordingClient, TraceRecorder, load_neighbor_chunks, write_trace_report


class _FakeCompletions:
    def create(self, **kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=" 测试回答[1] "))]
        )


class RagInspectorTests(unittest.TestCase):
    def test_recording_client_captures_exact_request_without_changing_response(self):
        delegate = SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions()))
        recorder = TraceRecorder("问题")
        client = RecordingClient(delegate, recorder)
        response = client.chat.completions.create(
            model="test-model",
            messages=[{"role": "user", "content": "资料"}],
            temperature=0.3,
        )
        self.assertEqual(response.choices[0].message.content.strip(), "测试回答[1]")
        self.assertEqual(recorder.request["model"], "test-model")
        self.assertEqual(recorder.request["messages"][0]["content"], "资料")
        self.assertEqual(recorder.answer, "测试回答[1]")

    def test_report_and_neighbor_lookup(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            generation = "a" * 32
            directory = root / "data" / "vector_db" / "generations" / generation
            directory.mkdir(parents=True)
            (root / "data" / "vector_db" / "current.json").write_text(
                json.dumps({"generation": generation}), encoding="utf-8"
            )
            chunks = [
                {"text": f"内容{i}", "metadata": {"source": "a.pdf", "document_id": "doc", "segment_index": i, "page": 1}}
                for i in range(3)
            ]
            (directory / "chunks_meta.json").write_text(
                json.dumps(chunks, ensure_ascii=False), encoding="utf-8"
            )
            hit = {"text": "内容1", "metadata": {**chunks[1]["metadata"], "rank": 1, "score": 0.8}}
            recorder = TraceRecorder("测试问题")
            recorder.record_retrieval([hit], 0.2)
            recorder.record_request({"model": "demo", "messages": [{"role": "user", "content": "[1] 内容1"}]})
            recorder.record_answer("回答", 0.4)
            recorder.neighbors = load_neighbor_chunks(root, [hit], radius=1)
            json_path, html_path = write_trace_report(recorder, root / "data/outputs/debug")

            self.assertEqual([item["text"] for item in recorder.neighbors["1"]], ["内容0", "内容2"])
            self.assertTrue(json_path.exists())
            self.assertTrue(html_path.exists())
            self.assertIn("测试问题", html_path.read_text(encoding="utf-8"))
            self.assertIn("已发送", html_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
