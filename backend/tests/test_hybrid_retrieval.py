"""RRF 与 BGE 重排的纯离线测试，不加载真实 Embedding 或重排模型。"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_agent.qa.reranker import Reranker
from rag_agent.qa.retriever import reciprocal_rank_fusion


def hit(chunk_id, route, rank, score):
    return {"text": f"正文-{chunk_id}", "metadata": {
        "chunk_id": chunk_id, f"{route}_rank": rank, f"{route}_score": score,
    }}


class FusionTests(unittest.TestCase):
    def test_rrf_keeps_single_route_and_deduplicates(self):
        """重复 ID 只保留一条，两路各自独有的候选也都进入融合。"""
        dense = [hit("shared", "dense", 1, .8), hit("dense-only", "dense", 2, .7)]
        sparse = [hit("shared", "bm25", 1, 8.0), hit("bm25-only", "bm25", 2, 6.0)]
        result = reciprocal_rank_fusion(dense, sparse, top_k=10, rrf_k=60)
        self.assertEqual([row["metadata"]["chunk_id"] for row in result],
                         ["shared", "bm25-only", "dense-only"])
        self.assertAlmostEqual(result[0]["metadata"]["fusion_score"], 2 / 61)
        self.assertNotIn("bm25_rank", result[2]["metadata"])
        self.assertNotIn("dense_rank", result[1]["metadata"])


class RerankerTests(unittest.TestCase):
    def test_prepare_is_idempotent_with_injected_scorer(self):
        """假评分器在准备阶段不加载本地模型，重复预热无额外动作。"""
        reranker = Reranker(scorer=lambda _q, texts: [0.0 for _ in texts])
        reranker.prepare()
        reranker.prepare()
        self.assertTrue(reranker._prepared)
        self.assertIsNone(reranker.model)

    def test_fake_scorer_changes_order_and_limits_top_five(self):
        """注入评分器可改变 RRF 排名，并且最终最多返回五条。"""
        candidates = [{"text": str(i), "metadata": {
            "chunk_id": str(i), "fusion_rank": i + 1,
        }} for i in range(8)]
        reranker = Reranker(scorer=lambda _q, texts: [float(text) for text in texts], top_k=5)
        result = reranker.rerank("问题", candidates)
        self.assertEqual([row["text"] for row in result], ["7", "6", "5", "4", "3"])
        self.assertEqual([row["metadata"]["rerank_rank"] for row in result], [1, 2, 3, 4, 5])

    def test_empty_candidates_do_not_call_scorer(self):
        """空候选直接返回，不调用假评分器，更不会加载真实模型。"""
        calls = []
        reranker = Reranker(scorer=lambda *_: calls.append(True))
        self.assertEqual(reranker.rerank("问题", []), [])
        self.assertEqual(calls, [])

    def test_missing_local_model_has_clear_error(self):
        """有候选但本地模型不存在时明确报错，且不会尝试联网下载。"""
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing-model"
            reranker = Reranker(model_name=str(missing))
            with self.assertRaisesRegex(FileNotFoundError, "不会自动下载"):
                reranker.rerank("问题", [{"text": "正文", "metadata": {"chunk_id": "c1"}}])


if __name__ == "__main__":
    unittest.main()
