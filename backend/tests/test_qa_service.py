"""服务层离线测试：验证证据门控严格位于重排和生成之间。"""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_agent.retrieval.evidence_policy import EvidencePolicy, REFUSAL_ANSWER
from rag_agent.chat.service import RAGService
from rag_agent import config
from rag_agent.qa.generator import Generator


class FakeRetriever:
    def __init__(self, hits):
        self.hits = hits

    def retrieve(self, _question):
        return self.hits


class FakeReranker:
    def __init__(self, score=.8):
        self.score = score

    def rerank(self, _question, hits):
        return [{"text": hit["text"], "metadata": {
            **hit["metadata"], "rerank_score": self.score, "rank": rank,
        }} for rank, hit in enumerate(hits[:5], 1)]


class FakeGenerator:
    def __init__(self, answer="测试回答"):
        self.calls = []
        self.answer = answer

    def generate(self, question, hits):
        self.calls.append((question, hits))
        return self.answer

    def stream(self, question, hits):
        self.calls.append((question, hits))
        for part in (self.answer[:2], self.answer[2:]):
            yield part


def evidence():
    return [{"text": "证据", "metadata": {"chunk_id": "c1", "source": "a.pdf", "page": 1}}]


class RAGServiceTests(unittest.TestCase):
    def test_prepare_runs_before_first_question_and_timing_excludes_generation(self):
        """预热由入口触发；检索合计不包含生成，分项耗时可供诊断报告使用。"""
        from time import sleep

        class PreparedReranker(FakeReranker):
            def __init__(self):
                super().__init__()
                self.prepares = 0
                self.device = "cuda"

            def prepare(self):
                self.prepares += 1

        class SlowGenerator(FakeGenerator):
            def generate(self, question, hits):
                sleep(.02)
                return super().generate(question, hits)

        reranker = PreparedReranker()
        service = RAGService(FakeRetriever(evidence()), reranker, EvidencePolicy(None), SlowGenerator())
        service.prepare()
        self.assertEqual(reranker.prepares, 1)
        result = service.ask("问题")
        self.assertGreaterEqual(result.timing["retrieval_seconds"], result.timing["recall_seconds"])
        self.assertGreaterEqual(result.timing["retrieval_seconds"], result.timing["rerank_seconds"])
        self.assertGreater(result.timing["generation_seconds"], result.timing["retrieval_seconds"])
        self.assertGreater(result.timing["total_seconds"], result.timing["generation_seconds"])

    def test_threshold_none_allows_answer_and_marks_uncalibrated(self):
        """阈值为 None 时不硬拒答，但结果明确标记尚未校准。"""
        generator = FakeGenerator()
        service = RAGService(FakeRetriever(evidence()), FakeReranker(), EvidencePolicy(None), generator)
        result = service.ask("问题")
        self.assertTrue(result.answerable)
        self.assertFalse(result.threshold_calibrated)
        self.assertEqual(result.reason, "threshold_not_calibrated")
        self.assertEqual(result.answer, "测试回答")
        self.assertEqual(len(generator.calls), 1)

    def test_model_refusal_after_evidence_gate_is_final_refusal(self):
        """重排候选通过后，模型拒答仍是最终拒答；尾随引用编号不影响判断。"""
        generator = FakeGenerator("资料中没有找到相关内容。\n\n[1][2][3]")
        service = RAGService(FakeRetriever(evidence()), FakeReranker(), EvidencePolicy(None), generator)
        result = service.ask("资料范围外的问题")
        self.assertFalse(result.answerable)
        self.assertEqual(result.reason, "generation_refused")
        self.assertEqual(result.evidence_status, "answerable")
        self.assertFalse(result.threshold_calibrated)
        self.assertEqual(len(generator.calls), 1)

    def test_empty_retrieval_rejects_without_generator(self):
        """没有候选时直接拒答，不调用重排器的模型路径或生成器。"""
        generator = FakeGenerator()
        service = RAGService(FakeRetriever([]), FakeReranker(), EvidencePolicy(None), generator)
        result = service.ask("问题")
        self.assertFalse(result.answerable)
        self.assertEqual(result.evidence_status, "no_candidates")
        self.assertEqual(result.answer, REFUSAL_ANSWER)
        self.assertEqual(generator.calls, [])

    def test_below_threshold_rejects_without_generator(self):
        """最终 Top-1 低于已校准阈值时统一拒答且不调用生成器。"""
        generator = FakeGenerator()
        service = RAGService(FakeRetriever(evidence()), FakeReranker(.4), EvidencePolicy(.6), generator)
        result = service.ask("问题")
        self.assertFalse(result.answerable)
        self.assertEqual(result.evidence_status, "below_threshold")
        self.assertEqual(generator.calls, [])

    def test_at_or_above_threshold_calls_generator(self):
        """Top-1 达标时正常生成并返回最终重排引用。"""
        generator = FakeGenerator()
        service = RAGService(FakeRetriever(evidence()), FakeReranker(.6), EvidencePolicy(.6), generator)
        result = service.ask("问题")
        self.assertTrue(result.answerable)
        self.assertTrue(result.threshold_calibrated)
        self.assertIsNone(result.reason)
        self.assertEqual(len(generator.calls), 1)

    def test_stream_uses_same_gate_and_final_refusal_rule(self):
        """流式生成沿用证据门控与最终拒答判定。"""
        generator = FakeGenerator("资料中没有找到相关内容。")
        service = RAGService(FakeRetriever(evidence()), FakeReranker(), EvidencePolicy(None), generator)
        events = list(service.ask_stream("问题"))
        self.assertEqual([name for name, _ in events],
                         ["status", "retrieval", "status", "delta", "delta", "done"])
        self.assertEqual("".join(data["text"] for name, data in events if name == "delta"),
                         generator.answer)
        self.assertFalse(events[-1][1].answerable)
        self.assertEqual(events[-1][1].reason, "generation_refused")

        blocked = RAGService(FakeRetriever([]), FakeReranker(), EvidencePolicy(None), generator)
        blocked_events = list(blocked.ask_stream("问题"))
        self.assertEqual([name for name, _ in blocked_events], ["status", "retrieval", "done"])
        self.assertEqual(len(generator.calls), 1)

    def test_prompt_budget_and_final_citations_match_in_both_modes(self):
        """只有实际送入模型且被答案引用的片段能进入最终结果与流式引用。"""
        hits = [
            {"text": "甲" * 10, "metadata": {"chunk_id": "c1", "source": "a.pdf", "page": 1}},
            {"text": "乙" * 10, "metadata": {"chunk_id": "c2", "source": "b.pdf", "page": 2}},
        ]
        generator = FakeGenerator("依据[1]，误引[2]。")
        service = RAGService(FakeRetriever(hits), FakeReranker(), EvidencePolicy(None), generator)
        with patch.object(config, "MAX_CONTEXT_CHARS", 14):
            result = service.ask("问题")
            events = list(service.ask_stream("问题"))
            prompt = Generator.build_prompt("问题", generator.calls[0][1])
        self.assertIn("[1] " + "甲" * 10, prompt[1]["content"])
        self.assertNotIn("乙", prompt[1]["content"])
        self.assertEqual([hit["metadata"]["rank"] for hit in generator.calls[0][1]], [1])
        self.assertEqual([hit["metadata"]["rank"] for hit in generator.calls[1][1]], [1])
        self.assertEqual(result.answer, "依据[1]，误引。")
        self.assertEqual([hit["metadata"]["rank"] for hit in result.hits], [1])
        self.assertEqual(len(result.retrieval_hits), 2)
        self.assertEqual([hit["metadata"]["rank"] for hit in events[1][1]["hits"]], [1])
        self.assertEqual(events[-1][1].answer, result.answer)
        self.assertEqual([hit["metadata"]["rank"] for hit in events[-1][1].hits], [1])

    def test_empty_prompt_budget_rejects_without_model(self):
        """首块证据放不进提示词时，不在空资料上生成，也不展示虚假引用。"""
        generator = FakeGenerator("回答[1]")
        service = RAGService(FakeRetriever(evidence()), FakeReranker(), EvidencePolicy(None), generator)
        with patch.object(config, "MAX_CONTEXT_CHARS", 3):
            result = service.ask("问题")
            events = list(service.ask_stream("问题"))
        self.assertEqual(generator.calls, [])
        self.assertEqual(result.reason, "context_budget_empty")
        self.assertFalse(result.answerable)
        self.assertEqual(result.hits, [])
        self.assertEqual(events[1][1]["hits"], [])
        self.assertEqual(events[-1][1].reason, "context_budget_empty")

    def test_uncited_prompt_hit_is_not_a_final_citation(self):
        """流式预告可包含已发送证据，最终结果只保存答案真正引用的编号。"""
        hits = [
            {"text": "证据甲", "metadata": {"chunk_id": "c1", "source": "a.pdf"}},
            {"text": "证据乙", "metadata": {"chunk_id": "c2", "source": "b.pdf"}},
        ]
        generator = FakeGenerator("根据资料[2]。")
        service = RAGService(FakeRetriever(hits), FakeReranker(), EvidencePolicy(None), generator)
        result = service.ask("问题")
        events = list(service.ask_stream("问题"))
        self.assertEqual([hit["metadata"]["rank"] for hit in generator.calls[0][1]], [1, 2])
        self.assertEqual([hit["metadata"]["rank"] for hit in events[1][1]["hits"]], [1, 2])
        self.assertEqual([hit["metadata"]["rank"] for hit in result.hits], [2])
        self.assertEqual([hit["metadata"]["rank"] for hit in events[-1][1].hits], [2])


if __name__ == "__main__":
    unittest.main()
