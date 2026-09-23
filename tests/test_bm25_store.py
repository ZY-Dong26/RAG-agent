"""中文 BM25 的离线测试：只使用临时目录和微型语料，不读取正式索引。"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_agent.indexing.bm25_store import BM25Store, tokenize


def chunks():
    """返回含稳定 ID 的最小中英文语料。"""
    return [
        {"text": "北京使用 Qwen3-Embedding-0.6B，准确率 95.5%（2024）", "metadata": {"chunk_id": "a"}},
        {"text": "上海数据库介绍 B+Tree 与 BM25", "metadata": {"chunk_id": "b"}},
    ]


class TokenizerTests(unittest.TestCase):
    def test_chinese_english_number_and_model_tokens(self):
        """搜索分词保留中文、英文小写、型号、年份和百分比，并过滤纯标点。"""
        values = tokenize("北京 Qwen3-Embedding-0.6B 2024 95.5% !!!")
        self.assertIn("北京", values)
        self.assertIn("qwen3-embedding-0.6b", values)
        self.assertIn("2024", values)
        self.assertIn("95.5%", values)
        self.assertEqual(tokenize("，。！？"), [])


class BM25StoreTests(unittest.TestCase):
    def test_build_query_save_and_load(self):
        """BM25 可查询并完整回读，返回结构带排名与原始分数。"""
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "bm25"
            store = BM25Store().build(chunks())
            self.assertEqual(store.search("北京 Qwen3", 1)[0]["metadata"]["chunk_id"], "a")
            store.save(directory)
            loaded = BM25Store.load(directory, chunks())
            hit = loaded.search("数据库 B+Tree", 1)[0]
            self.assertEqual(hit["metadata"]["chunk_id"], "b")
            self.assertEqual(hit["metadata"]["bm25_rank"], 1)
            self.assertIsInstance(hit["metadata"]["bm25_score"], float)

    def test_corruption_and_chunk_mismatch_are_rejected(self):
        """任一索引文件损坏或 chunk ID 顺序不一致时拒绝加载。"""
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "bm25"
            BM25Store().build(chunks()).save(directory)
            with self.assertRaisesRegex(ValueError, "数量或 ID"):
                BM25Store.load(directory, list(reversed(chunks())))
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            damaged = directory / next(iter(manifest["artifacts"]))
            damaged.write_bytes(b"damaged")
            with self.assertRaisesRegex(ValueError, "校验失败"):
                BM25Store.load(directory, chunks())

    def test_legacy_generation_has_clear_upgrade_message(self):
        """旧代没有 BM25 时明确提示重建，不静默退化。"""
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "不含 BM25"):
                BM25Store.load(Path(tmp) / "bm25", chunks())


if __name__ == "__main__":
    unittest.main()
