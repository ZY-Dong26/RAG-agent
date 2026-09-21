# 离线回归测试 —— 检查版本化发布、按文档向量缓存、部分成功和严格模式
# 所有建库场景均使用 TemporaryDirectory、假解析结果和假 Embedding，
# 不读取真实 PDF 内容、不请求 MinerU、不加载本地大模型，也不会改动正式 vector_db。
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from types import SimpleNamespace
from unittest.mock import patch

import faiss

from rag_agent import config
from rag_agent.indexing import chunker, faiss_store
from rag_agent.indexing import builder
from rag_agent.indexing.artifact_store import load_artifact, save_artifact
from rag_agent.common.files import atomic_json, read_json


class PublicationTests(unittest.TestCase):
    """验证索引完整性与原子发布；所有文件写入临时目录，不触碰正式知识库。"""
    def chunks(self, text="old"):
        """生成带稳定 ID 的最小 chunk，测试向量与文本仍然一一对应。"""
        return [{"text": text, "metadata": {
            "source": "doc.pdf", "page": 1, "document_id": "doc", "chunk_id": "chunk-" + text,
        }}]

    def test_failed_publication_preserves_previous_generation(self):
        """切换 current.json 失败时旧版本仍可读；随后成功发布，旧版本目录仍保留。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = faiss_store.VectorStore(root)
            old.add(self.chunks(), [[1.0, 0.0]])
            old.save({"signature": "old"})
            pointer = read_json(root / "current.json")
            new = faiss_store.VectorStore(root)
            new.add(self.chunks("new"), [[0.0, 1.0]])

            def fail_pointer(path, value):
                """仅在切换版本指针时模拟磁盘失败，验证新文件写入后旧版本仍可使用。"""
                if Path(path).name == "current.json":
                    raise OSError("simulated disk error")
                atomic_json(path, value)

            with patch.object(faiss_store, "atomic_json", side_effect=fail_pointer):
                with self.assertRaises(OSError):
                    new.save({"signature": "new"})
            self.assertEqual(read_json(root / "current.json"), pointer)
            self.assertEqual(faiss_store.VectorStore(root).load().chunks, self.chunks())
            new.save({"signature": "new"})
            self.assertEqual(faiss_store.VectorStore(root).load().chunks, self.chunks("new"))
            self.assertTrue((root / "generations" / pointer["generation"] / "index.faiss").exists())

    def test_legacy_index_kept_and_invalid_vectors_rejected(self):
        """旧格式索引仍可读取和保留，同时拒绝数量不匹配以及 NaN 向量。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = faiss_store.VectorStore(root)
            store.add(self.chunks(), [[1.0, 0.0]])
            faiss.write_index(store.index, str(root / "index.faiss"))
            atomic_json(root / "chunks_meta.json", store.chunks)
            original = (root / "index.faiss").read_bytes()
            self.assertEqual(faiss_store.VectorStore(root).load().chunks, self.chunks())
            store.save()
            self.assertEqual((root / "index.faiss").read_bytes(), original)
            with self.assertRaises(ValueError):
                store.add(self.chunks(), [[1.0, 0.0], [0.0, 1.0]])
            with self.assertRaises(ValueError):
                store.add(self.chunks(), [[float("nan"), 0.0]])

    def test_document_artifact_roundtrip_and_hash_validation(self):
        """按文档产物可以安全回读；任一数据文件被修改后必须拒绝复用。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            key = "a" * 64
            documents = [{"text": "正文", "metadata": {"source": "a.pdf", "page": 1}}]
            chunks = self.chunks("正文")
            saved = save_artifact(root, key, documents, chunks, [[1.0, 0.0]],
                                  {"document_id": "doc", "pages": 1})
            self.assertEqual(saved["chunks"], chunks)
            self.assertEqual(saved["vectors"].shape, (1, 2))
            artifact_dir = root / "document_artifacts" / key
            (artifact_dir / "chunks.json").write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "校验失败"):
                load_artifact(root, key)

    def test_metadata_and_bounded_chunks(self):
        """切块保留引用信息，长正文和表格切块均受字符预算约束。"""
        splitter = chunker.TextSplitter(80, 10)
        meta = {"source": "a.pdf", "page": 2, "block_type": "text",
                "block_id": "p2_b1", "parser": "mineru_cloud"}
        chunks = splitter.split_documents([{"text": "中" * 200, "metadata": meta}])
        self.assertTrue(all(len(chunk["text"]) <= 80 for chunk in chunks))
        self.assertTrue(all(chunk["metadata"]["block_id"] == "p2_b1" for chunk in chunks))
        meta["block_type"] = "table"
        chunks = splitter.split_documents([{
            "text": "header\n" + "\n".join(["a | b " * 10] * 10), "metadata": meta,
        }])
        self.assertTrue(all(len(chunk["text"]) <= 80 for chunk in chunks))
        self.assertTrue(all(chunk["text"].startswith("header") for chunk in chunks))


class IncrementalBuildTests(unittest.TestCase):
    """用临时文件验证按文档状态转换，测试中不会触碰真实项目数据。"""

    def environment(self, root):
        """返回本测试统一使用的配置补丁；模型目录为空也能生成稳定测试签名。"""
        raw = root / "raw"
        raw.mkdir()
        model = root / "model"
        model.mkdir()
        return raw, patch.multiple(
            config, RAW_PDF_DIR=raw, VECTOR_DB_DIR=root / "db", PROCESSED_DIR=root / "processed",
            CHUNKS_FILE=root / "processed" / "chunks.json", EMBEDDING_MODEL=str(model),
        )

    @staticmethod
    def document(source, text):
        """构造与 MinerU 适配器输出一致的最小文档记录。"""
        return {"text": text, "metadata": {
            "source": source, "page": 1, "block_id": "p1_b0", "block_type": "text",
            "parser": "mineru_cloud", "adapter_version": "test",
        }}

    def fake_embedding_module(self, calls):
        """创建确定性的二维假向量；calls 用来断言实际向量化了多少批文档。"""
        class FakeEmbedding:
            """测试用向量模型，只返回确定性的二维数据，不加载真实模型。"""
            def embed_texts(self, texts):
                """记录本次输入并生成测试向量，用调用次数验证是否正确复用缓存。"""
                calls.append(list(texts))
                return [[1.0, float(index + 1)] for index, _ in enumerate(texts)]
        return SimpleNamespace(Embedding=FakeEmbedding)

    def test_partial_build_reuses_success_and_skips_new_failure(self):
        """默认模式发布成功文档；再次运行只重试失败文档，不重新向量化成功文档。"""
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw, config_patch = self.environment(root)
            (raw / "ok.pdf").write_bytes(b"ok")
            (raw / "bad.pdf").write_bytes(b"bad")

            def parse(paths, **_):
                """模拟一份解析成功、一份解析失败，检查默认建库能够发布可用内容。"""
                self.assertEqual([path.name for path in paths], ["bad.pdf", "ok.pdf"])
                return ({"bad.pdf": [], "ok.pdf": [self.document("ok.pdf", "可用正文")]}, [
                    {"source": "bad.pdf", "status": "failed", "stage": "parse_failed", "error": "云端失败"},
                    {"source": "ok.pdf", "status": "done", "parsed_pages": 1},
                ])

            with config_patch, patch.object(builder, "load_settings",
                    return_value=SimpleNamespace(parameters=lambda: {"model": "test"})), \
                    patch.object(builder, "collect_parse_results", side_effect=parse), \
                    patch.dict(sys.modules, {"rag_agent.indexing.embedder": self.fake_embedding_module(calls)}):
                builder.build_index()
                store = faiss_store.VectorStore(root / "db").load()
                self.assertEqual([chunk["metadata"]["source"] for chunk in store.chunks], ["ok.pdf"])
                self.assertEqual(read_json(root / "processed" / "build_report.json")["status"], "partial")
                self.assertEqual(len(calls), 1)

                # 第二次只有 bad.pdf 需要解析；ok.pdf 的文档产物和向量直接复用。
                def retry(paths, **_):
                    """模拟再次处理失败文档；断言成功文档没有被重复送入解析流程。"""
                    self.assertEqual([path.name for path in paths], ["bad.pdf"])
                    return ({"bad.pdf": []}, [{"source": "bad.pdf", "status": "failed",
                                                "stage": "parse_failed", "error": "仍然失败"}])
                with patch.object(builder, "collect_parse_results", side_effect=retry):
                    builder.build_index()
                self.assertEqual(len(calls), 1)

    def test_strict_failure_does_not_publish(self):
        """严格模式中只要一份当前 PDF 失败，就不计算向量也不创建 current.json。"""
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw, config_patch = self.environment(root)
            (raw / "ok.pdf").write_bytes(b"ok")
            (raw / "bad.pdf").write_bytes(b"bad")
            grouped = {"ok.pdf": [self.document("ok.pdf", "正文")], "bad.pdf": []}
            report = [{"source": "ok.pdf", "status": "done", "parsed_pages": 1},
                      {"source": "bad.pdf", "status": "failed", "stage": "parse_failed", "error": "失败"}]
            with config_patch, patch.object(builder, "load_settings",
                    return_value=SimpleNamespace(parameters=lambda: {})), \
                    patch.object(builder, "collect_parse_results", return_value=(grouped, report)), \
                    patch.dict(sys.modules, {"rag_agent.indexing.embedder": self.fake_embedding_module(calls)}):
                with self.assertRaisesRegex(RuntimeError, "严格模式"):
                    builder.build_index(strict=True)
            self.assertFalse((root / "db" / "current.json").exists())
            self.assertEqual(calls, [])

    def test_failed_update_keeps_old_then_success_replaces_it(self):
        """已入库 PDF 更新失败时保留旧正文；后来成功时只计算该文档并替换旧版。"""
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw, config_patch = self.environment(root)
            pdf = raw / "report.pdf"
            pdf.write_bytes(b"version-one")
            settings = SimpleNamespace(parameters=lambda: {})
            first = ({"report.pdf": [self.document("report.pdf", "旧正文")]},
                     [{"source": "report.pdf", "status": "done", "parsed_pages": 1}])
            with config_patch, patch.object(builder, "load_settings", return_value=settings), \
                    patch.object(builder, "collect_parse_results", return_value=first), \
                    patch.dict(sys.modules, {"rag_agent.indexing.embedder": self.fake_embedding_module(calls)}):
                builder.build_index()
                pdf.write_bytes(b"version-two")
                failed = ({"report.pdf": []}, [{"source": "report.pdf", "status": "failed",
                                                  "stage": "parse_failed", "error": "新版解析失败"}])
                with patch.object(builder, "collect_parse_results", return_value=failed):
                    builder.build_index()
                stale = faiss_store.VectorStore(root / "db").load()
                self.assertEqual(stale.chunks[0]["text"], "旧正文")
                self.assertEqual(read_json(root / "processed" / "build_report.json")["stale_documents"], 1)

                success = ({"report.pdf": [self.document("report.pdf", "新正文")]},
                           [{"source": "report.pdf", "status": "done", "parsed_pages": 1}])
                with patch.object(builder, "collect_parse_results", return_value=success):
                    builder.build_index()
                current = faiss_store.VectorStore(root / "db").load()
                self.assertEqual(current.chunks[0]["text"], "新正文")
                self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
