"""MinerU 规则后处理 V1 的纯合成测试：不访问网络、不读取正式数据、不加载模型。"""
import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_agent.indexing import builder
from rag_agent.indexing.chunker import TextSplitter
from rag_agent.ingestion import postprocessor
from rag_agent.ingestion.mineru_client import MinerUCloud
from rag_agent.ingestion.mineru_settings import MinerUSettings
from rag_agent.ingestion.postprocessor import POSTPROCESSOR_VERSION, postprocess_document


def block(text, page, block_id, kind="text", bbox=None, title_level=None, **metadata):
    """构造与 MinerU 适配器输出一致的最小块，并允许注入版面与未知字段。"""
    values = {"source": "sample.pdf", "page": page, "block_id": block_id,
              "block_type": kind, **metadata}
    if bbox is not None:
        values["bbox"] = bbox
    if title_level is not None:
        values["title_level"] = title_level
    return {"text": text, "metadata": values}


class MarginRuleTests(unittest.TestCase):
    def test_repeated_headers_variable_footers_and_body_duplicate(self):
        """稳定页边位置会被排除；正文中偶然同文不会仅凭内容被排除。"""
        raw = []
        for page in range(1, 6):
            raw.extend([
                block("RAG 学习报告", page, f"h{page}", "header", [0, 0.01, 1, 0.08]),
                block(f"第 {page} 页正文。", page, f"b{page}", bbox=[0, 0.4, 1, 0.5]),
                block(f"- {page} -", page, f"f{page}", "footer", [0, 0.9, 1, 0.98]),
            ])
        raw.insert(4, block("RAG 学习报告", 2, "body-duplicate", bbox=[0, 0.5, 1, 0.6]))

        refined, report = postprocess_document(raw, "sample.pdf")
        by_id = {row["metadata"]["block_id"]: row for row in refined}
        self.assertTrue(all(by_id[f"h{page}"]["metadata"]["excluded_reason"] == "repeated_header"
                            for page in range(1, 6)))
        self.assertTrue(all(by_id[f"f{page}"]["metadata"]["excluded_reason"] == "page_number"
                            for page in range(1, 6)))
        self.assertFalse(by_id["body-duplicate"]["metadata"]["excluded_from_retrieval"])
        self.assertEqual((report["excluded_headers"], report["excluded_page_numbers"]), (5, 5))

    def test_two_pages_do_not_supply_enough_repetition_evidence(self):
        """两页文档即使页眉相同也不排除，宁可保留噪声而不误删正文。"""
        raw = [block("短文页眉", page, f"h{page}", "header") for page in (1, 2)]
        refined, report = postprocess_document(raw)
        self.assertFalse(any(row["metadata"]["excluded_from_retrieval"] for row in refined))
        self.assertEqual(report["excluded_headers"], 0)


class CrossPageMergeTests(unittest.TestCase):
    def _merge(self, left, right):
        return postprocess_document([block(left, 1, "left"), block(right, 2, "right")])[0]

    def test_english_hyphen_lowercase_and_chinese_continuations(self):
        """三类明确续接分别去连字符、补空格或直接连接，并保留来源 ID 与跨页范围。"""
        cases = [
            ("retriev-", "al improves quality", "retrieval improves quality"),
            ("Retrieval uses dense", "and sparse signals", "Retrieval uses dense and sparse signals"),
            ("该方法同时使用稠密检索，", "并结合稀疏召回提高效果", "该方法同时使用稠密检索，并结合稀疏召回提高效果"),
        ]
        for left, right, expected in cases:
            with self.subTest(expected=expected):
                refined = self._merge(left, right)
                self.assertEqual(len(refined), 1)
                self.assertEqual(refined[0]["text"], expected)
                self.assertEqual(refined[0]["metadata"]["source_block_ids"], ["left", "right"])
                self.assertEqual((refined[0]["metadata"]["page_start"],
                                  refined[0]["metadata"]["page_end"]), (1, 2))

    def test_terminator_and_special_structures_never_merge(self):
        """完整句和标题、表格、公式、脚注、参考文献等结构都阻断跨页正文合并。"""
        self.assertEqual(len(self._merge("Complete sentence.", "next begins lower")), 2)
        for kind in ("title", "table", "equation", "page_footnote", "ref_text", "list", "caption", "image"):
            with self.subTest(kind=kind):
                raw = [block("previous without stop", 1, "left"), block("next content", 2, "right", kind)]
                refined, _ = postprocess_document(raw)
                self.assertEqual(len(refined), 2)


class TitleAndSectionTests(unittest.TestCase):
    def test_arabic_title_levels_and_section_path(self):
        """阿拉伯层级编号决定标题等级，正文继承完整章节路径。"""
        raw = [block("1.RAG技术", 1, "t1"), block("1.2 检索增强", 1, "t2"),
               block("1.2.1 查询重写", 1, "t3"), block("正文内容。", 1, "body")]
        refined, report = postprocess_document(raw)
        self.assertEqual([row["metadata"]["title_level"] for row in refined[:3]], [1, 2, 3])
        self.assertEqual(refined[-1]["metadata"]["section_path"],
                         ["1.RAG技术", "1.2 检索增强", "1.2.1 查询重写"])
        self.assertEqual(report["promoted_titles"], 3)

    def test_chinese_levels_but_lists_and_references_are_not_titles(self):
        """中文章节目次可分级；列表与参考文献编号由类型保护，不会误提升。"""
        raw = [block("第一章 基础", 1, "c1"), block("第一节 检索", 1, "c2"),
               block("一、总体方案", 1, "c3"), block("（一）实现细节", 1, "c4"),
               block("1）普通列表项", 1, "list", "list"),
               block("[1] Author. Paper.", 1, "ref", "ref_text")]
        refined, _ = postprocess_document(raw)
        self.assertEqual([row["metadata"]["title_level"] for row in refined[:4]], [1, 2, 1, 2])
        self.assertEqual(refined[4]["metadata"]["block_type"], "list")
        self.assertEqual(refined[5]["metadata"]["block_type"], "ref_text")


class IntegrityAndBoundaryTests(unittest.TestCase):
    def test_special_content_unknown_fields_and_chunk_exclusion(self):
        """特殊原文、未知字段和未知类型保留；切片器只跳过显式排除块。"""
        raw = [
            block("<table>  A  </table>", 1, "table", "table", mineru_block={"custom": 1}),
            block(r"E = mc^2  \\alpha", 1, "formula", "equation"),
            block("脚注", 1, "footnote", "page_footnote"),
            block("参考文献", 1, "reference", "ref_text"),
            block("未知结构", 1, "unknown", "new_cloud_type", custom_field="kept"),
        ]
        refined, report = postprocess_document(raw)
        self.assertEqual([row["text"] for row in refined[:2]], [raw[0]["text"], raw[1]["text"]])
        self.assertEqual(refined[0]["metadata"]["mineru_block"], {"custom": 1})
        self.assertEqual(refined[4]["metadata"]["custom_field"], "kept")
        self.assertTrue(any(row["type"] == "unknown_block_type" for row in report["warnings"]))

        excluded = copy.deepcopy(refined[4])
        excluded["metadata"]["excluded_from_retrieval"] = True
        chunks = TextSplitter(100, 10).split_documents([refined[2], excluded])
        self.assertEqual([row["text"] for row in chunks], ["脚注"])

    def test_deterministic_does_not_mutate_and_fail_open_returns_original(self):
        """同一输入结果稳定且不修改调用方对象；校验异常时返回未加工深拷贝。"""
        raw = [block("  正文   内容  ", 1, "body")]
        original = copy.deepcopy(raw)
        first, first_report = postprocess_document(raw)
        second, second_report = postprocess_document(raw)
        self.assertEqual((first, first_report), (second, second_report))
        self.assertEqual(raw, original)

        with patch.object(postprocessor, "_validate_document", side_effect=ValueError("secret URL")):
            fallback, report = postprocess_document(raw)
        self.assertEqual(fallback, original)
        self.assertIsNot(fallback, raw)
        self.assertTrue(report["fail_open"])
        self.assertNotIn("secret", str(report))

    def test_postprocessor_version_changes_local_pipeline_not_mineru_cache_key(self):
        """规则版本只改变本地 pipeline signature，不改变 PDF 的 MinerU 云端解析缓存键。"""
        settings = MinerUSettings(token="test")
        with tempfile.TemporaryDirectory() as temp:
            pdf = Path(temp) / "sample.pdf"
            pdf.write_bytes(b"same-pdf")
            parser = MinerUCloud(settings, Path(temp) / "cache", client=object())
            cloud_key = parser.cache_key(pdf)
            before = builder.pipeline_signature(settings, "embedding")
            with patch.object(builder, "POSTPROCESSOR_VERSION", "rules-v2"):
                after = builder.pipeline_signature(settings, "embedding")
                self.assertEqual(parser.cache_key(pdf), cloud_key)
            self.assertNotEqual(before, after)
            self.assertEqual(POSTPROCESSOR_VERSION, "rules-v1")


if __name__ == "__main__":
    unittest.main()
