"""自动判分与独立导出的离线测试：不访问检索器、回答模型或真实裁判接口。"""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from devtools.evaluation.exporter import JUDGE_COLUMNS, export_results, load_items
from devtools.evaluation.judger import build_judge_messages, judge_directory, parse_judge_json


def sample_item(qid, judged=False):
    """生成包含检索指标的最小逐题记录，结构与 evaluate.py 保存结果一致。"""
    item = {
        "question_id": qid, "question_type": "事实题", "difficulty": "简单",
        "question": f"问题{qid}", "ground_truth": "参考答案", "eval_criteria": "覆盖关键事实",
        "reference_context_ids": ["文档1"], "targets": ["a.pdf"], "missing_sources": [],
        "hits": [{"text": "检索证据", "metadata": {"rank": 1, "source": "a.pdf", "page": 2, "score": .8}}],
        "answer": "RAG 回答", "usage": None,
        "metrics": {"hit_at_k": 1, "recall_at_k": 1.0, "reciprocal_rank_at_k": 1.0},
        "retrieval_seconds": .1, "generation_seconds": .2, "total_seconds": .3,
        "error": None, "status": "answered",
    }
    if judged:
        item["judge"] = {"scores": {"correctness": 3, "completeness": 2, "relevance": 3},
                         "reason": "已有判分"}
    return item


class FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = '```json\n{"correctness":3,"completeness":2,"relevance":3,"reason":"证据支持结论"}\n```'
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=20, total_tokens=120)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))], usage=usage)


class JudgingTests(unittest.TestCase):
    def test_exporter_reads_items_and_reserves_judge_columns(self):
        """纯本地导出读取 items，CSV 末尾预留四列，并生成未判分汇总表。"""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "items").mkdir()
            (root / "items" / "one.json").write_text(json.dumps(sample_item(1), ensure_ascii=False), encoding="utf-8")
            rows = load_items(root)
            export_results(root, rows, total=2)
            with (root / "results.csv").open(encoding="utf-8-sig", newline="") as stream:
                records = list(csv.DictReader(stream))
                self.assertEqual(records[0][JUDGE_COLUMNS[0]], "")
                self.assertEqual(list(records[0])[-4:], JUDGE_COLUMNS)
            self.assertEqual(json.loads((root / "summary.json").read_text(encoding="utf-8"))["remaining"], 1)
            table = (root / "judge_summary.md").read_text(encoding="utf-8")
            self.assertIn("| 事实题 | 1 | 100.0%", table)
            self.assertIn("|  |  |  |", table)

    def test_judge_writes_item_then_resume_skips_it_and_refreshes_exports(self):
        """每题判完立刻写 judge；重跑跳过已有分数，CSV、报告与汇总同步更新。"""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            items = root / "items"
            items.mkdir()
            (items / "one.json").write_text(json.dumps(sample_item(1), ensure_ascii=False), encoding="utf-8")
            (items / "two.json").write_text(json.dumps(sample_item(2, judged=True), ensure_ascii=False), encoding="utf-8")
            completions = FakeCompletions()
            client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

            result = judge_directory(root, client, "judge-model", "https://judge.example/v1",
                                     temperature=0.4, max_tokens=321)
            self.assertEqual((result["completed"], result["skipped"], result["failed"]), (1, 1, 0))
            self.assertEqual(len(completions.calls), 1)
            self.assertEqual(completions.calls[0]["temperature"], 0.4)
            self.assertEqual(completions.calls[0]["max_tokens"], 321)
            self.assertEqual(completions.calls[0]["extra_body"], {"thinking": {"type": "disabled"}})
            self.assertEqual(load_items(root)[0]["judge"]["scores"]["correctness"], 3)
            self.assertEqual(load_items(root)[0]["judge"]["temperature"], 0.4)
            self.assertEqual(load_items(root)[0]["judge"]["max_tokens"], 321)
            self.assertNotIn("https://judge.example/v1", json.dumps(load_items(root), ensure_ascii=False))
            self.assertIn("证据支持结论", (root / "report.md").read_text(encoding="utf-8"))
            self.assertIn("| 事实题 | 2 | 100.0% | 100.0% | 100.0% | 0.30s | 3.00 | 2.00 | 3.00 |",
                          (root / "judge_summary.md").read_text(encoding="utf-8"))

            again = judge_directory(root, client, "judge-model", "https://judge.example/v1",
                                    temperature=0.4, max_tokens=321)
            self.assertEqual((again["completed"], again["skipped"]), (0, 2))
            self.assertEqual(len(completions.calls), 1)

    def test_prompt_and_json_validation(self):
        """裁判能看到要求的五类信息；非法或越界 JSON 明确失败。"""
        messages = build_judge_messages(sample_item(1))
        system = messages[0]["content"]
        user = messages[1]["content"]
        self.assertIn("空回答或表示未找到相关内容", system)
        self.assertIn("reason 字段控制在 50 字以内", system)
        for expected in ("问题1", "参考答案", "覆盖关键事实", "RAG 回答", "检索证据"):
            self.assertIn(expected, user)
        scores, reason = parse_judge_json('{"correctness":1,"completeness":2,"relevance":3,"reason":"理由"}')
        self.assertEqual(scores, {"correctness": 1, "completeness": 2, "relevance": 3})
        self.assertEqual(reason, "理由")
        with self.assertRaises(ValueError):
            parse_judge_json('{"correctness":4,"completeness":2,"relevance":3,"reason":"理由"}')


if __name__ == "__main__":
    unittest.main()
