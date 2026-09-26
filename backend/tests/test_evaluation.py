"""评测离线测试：验证指标、标签隔离、恢复、防混跑和失败记录。"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from devtools.evaluation.runner import load_dataset, retrieval_metrics, run_evaluation
from rag_agent.qa.generator import Generator


class EvaluationTests(unittest.TestCase):
    """用假检索与假生成器复现批量流程，不加载模型或访问 API。"""

    def test_metrics_count_unique_documents_and_chunk_rank(self):
        """重复文档只算一次召回；MRR 使用第一个正确片段的排名。"""
        hits = [{"metadata": {"source": s}} for s in ["x", "a", "a", "b"]]
        result = retrieval_metrics(hits, ["a", "b"], 3)
        self.assertEqual(result, {"hit_at_k": 1, "recall_at_k": .5, "reciprocal_rank_at_k": .5})
        self.assertIsNone(retrieval_metrics(hits, [], 3)["hit_at_k"])

    def test_resume_exports_and_label_isolation(self):
        """两题分批执行，恢复不重复生成；答案标签不会混入模型输入。"""
        calls = []
        hits = [{"text": "检索证据", "metadata": {"source": "a.pdf", "rank": 1, "page": 2, "score": .8}}]
        retriever = SimpleNamespace(store=SimpleNamespace(chunks=hits), retrieve=lambda q, top_k: hits)
        def generate(question, evidence):
            calls.append((question, evidence))
            return "真实回答"
        generator = SimpleNamespace(generate=generate, build_prompt=Generator.build_prompt, last_usage=None)
        rows = [{"question_id": i, "question": f"问题{i}", "question_type": "事实", "difficulty": "简单",
                 "ground_truth": "秘密答案", "eval_criteria": "秘密规则", "reference_context_ids": ["文档1"]} for i in [1, 2]]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_evaluation(rows, {"文档1": "a.pdf"}, retriever, generator, root, {"v": 1}, limit=1)
            summary = run_evaluation(rows, {"文档1": "a.pdf"}, retriever, generator, root, {"v": 1})
            self.assertEqual(len(calls), 2)
            self.assertEqual(summary["answered"], 2)
            data = [json.loads(line) for line in (root / "results.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertNotIn("秘密", str(data[0]["messages"]))
            self.assertTrue((root / "results.csv").read_bytes().startswith(b"\xef\xbb\xbf"))
            self.assertTrue((root / "report.md").exists())
            self.assertTrue((root / "judge_summary.md").exists())
            with self.assertRaises(ValueError):
                run_evaluation(rows, {"文档1": "a.pdf"}, retriever, generator, root, {"v": 2})

    def test_failed_generation_preserves_evidence_and_redacts(self):
        """生成失败不丢检索指标，也不把异常里的秘密写入结果。"""
        hits = [{"text": "正文", "metadata": {"source": "a", "rank": 1}}]
        def fail(*args):
            raise RuntimeError("secret-token https://private")
        ret = SimpleNamespace(store=SimpleNamespace(chunks=hits), retrieve=lambda q, top_k: hits)
        gen = SimpleNamespace(generate=fail, build_prompt=Generator.build_prompt)
        rows = [{"question_id": 1, "question": "问题", "question_type": "事实", "difficulty": "简单",
                 "ground_truth": "答案", "eval_criteria": "规则", "reference_context_ids": ["文档1"]}]
        with tempfile.TemporaryDirectory() as tmp:
            summary = run_evaluation(rows, {"文档1": "a"}, ret, gen, tmp, {})
            self.assertEqual(summary["document_hit_at_k"], 1)
            self.assertEqual(summary["answered"], 0)
            text = (Path(tmp) / "results.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("secret-token", text)
            self.assertNotIn("https://private", text)

    def test_rerank_latency_is_recorded_separately(self):
        """逐题和汇总记录均区分召回、重排、检索合计。"""
        hits = [{"text": "证据", "metadata": {"source": "a", "rank": 1}}]
        retriever = SimpleNamespace(store=SimpleNamespace(chunks=hits), retrieve=lambda q, top_k: hits)
        reranker = SimpleNamespace(rerank=lambda q, candidates, top_k: candidates)
        generator = SimpleNamespace(generate=lambda q, evidence: "回答",
                                    build_prompt=Generator.build_prompt, last_usage=None)
        row = {"question_id": 1, "question": "问题", "question_type": "事实", "difficulty": "简单",
               "ground_truth": "答案", "eval_criteria": "规则", "reference_context_ids": ["文档1"]}
        with tempfile.TemporaryDirectory() as tmp:
            summary = run_evaluation([row], {"文档1": "a"}, retriever, generator, tmp, {},
                                     reranker=reranker)
            item = json.loads((Path(tmp) / "results.jsonl").read_text(encoding="utf-8"))
            self.assertIsNotNone(item["recall_seconds"])
            self.assertIsNotNone(item["rerank_seconds"])
            self.assertGreaterEqual(item["retrieval_seconds"], item["recall_seconds"])
            self.assertGreaterEqual(item["retrieval_seconds"], item["rerank_seconds"])
            self.assertIsNotNone(summary["mean_rerank_seconds"])
            self.assertIn("rerank_seconds", (Path(tmp) / "results.csv").read_text(encoding="utf-8-sig"))

    def test_invalid_mapping_rejected(self):
        """缺少目标文档映射时，在任何模型调用之前拒绝评测。"""
        row = {"question_id": 1, "question": "问题", "question_type": "事实", "difficulty": "简单",
               "ground_truth": "答案", "eval_criteria": "规则", "reference_context_ids": ["文档1"]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "questions.json"
            path.write_text(json.dumps([row]), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_dataset(path, {})
