"""检索模型路径配置测试。"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_agent import config


class RetrievalConfigTests(unittest.TestCase):
    def test_relative_and_absolute_model_paths(self):
        """相对路径基于项目根目录，绝对路径保持不变。"""
        relative = Path(config._model_path("models/reranker", "unused"))
        self.assertEqual(relative, config.BASE_DIR / "models" / "reranker")
        absolute = config.BASE_DIR / "external" / "reranker"
        self.assertEqual(Path(config._model_path(str(absolute), "unused")), absolute)

    def test_default_retrieval_values(self):
        """默认候选规模与未校准阈值符合混合检索设计。"""
        self.assertEqual((config.DENSE_TOP_K, config.BM25_TOP_K), (30, 30))
        self.assertEqual((config.FUSION_TOP_K, config.RERANK_TOP_K), (20, 5))
        self.assertIsNone(config.RERANK_REJECT_THRESHOLD)
        self.assertTrue(Path(config.RERANKER_MODEL).is_absolute())


if __name__ == "__main__":
    unittest.main()
