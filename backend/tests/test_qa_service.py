"""服务层离线测试：验证证据门控严格位于重排和生成之间。"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_agent.qa.evidence_policy import EvidencePolicy, REFUSAL_ANSWER
from rag_agent.qa.service import RAGService


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
    def __init__(self):
        self.calls = []

    def generate(self, question, hits):
        self.calls.append((question, hits))
        return "测试回答"


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


if __name__ == "__main__":
    unittest.main()
