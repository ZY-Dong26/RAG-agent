"""章节感知切块器的纯本地测试，不读取正式数据，也不调用任何模型。"""
import copy
import unittest

from rag_agent.indexing import chunker
from rag_agent.ingestion.postprocessor import postprocess_document


def block(text, block_id, path=None, kind="text", page=1, source="a.pdf", document_id="doc-a"):
    """构造已经过后处理的最小统一记录。"""
    return {"text": text, "metadata": {
        "source": source, "document_id": document_id, "page": page,
        "page_start": page, "page_end": page, "block_id": block_id,
        "source_block_ids": [block_id], "block_type": kind,
        "section_path": list(path or []), "atomic": kind in {"table", "equation", "formula"},
    }}


class SectionChunkerTests(unittest.TestCase):
    def setUp(self):
        self.splitter = chunker.TextSplitter(1000, 150, section_aware=True)

    def test_01_continuous_same_section_is_packed_together(self):
        docs = [block("甲" * 420, "a", ["第一章"]), block("乙" * 420, "b", ["第一章"])]
        chunks = self.splitter.split_documents(docs)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["metadata"]["source_block_ids"], ["a", "b"])

    def test_02_same_named_sections_are_not_global_or_cross_document(self):
        docs = [block("甲" * 300, "a", ["同名"]),
                block("乙" * 300, "b", ["同名"], source="b.pdf", document_id="doc-b")]
        chunks = self.splitter.split_documents(docs)
        self.assertEqual(len(chunks), 2)
        self.assertNotEqual(chunks[0]["metadata"]["document_id"],
                            chunks[1]["metadata"]["document_id"])

    def test_03_small_sibling_sections_merge(self):
        docs = [block("甲" * 450, "a", ["章", "1.1"]),
                block("乙" * 450, "b", ["章", "1.2"])]
        chunk = self.splitter.split_documents(docs)[0]
        self.assertEqual(chunk["metadata"]["section_paths"], [["章", "1.1"], ["章", "1.2"]])
        self.assertEqual(chunk["metadata"]["section_path"], ["章"])

    def test_04_hierarchy_rise_stops_merge(self):
        docs = [block("甲" * 300, "a", ["章", "节"]), block("乙" * 300, "b", ["下一章"])]
        self.assertEqual(len(self.splitter.split_documents(docs)), 2)

    def test_05_major_title_boundary_stops_merge(self):
        docs = [block("甲" * 300, "a", ["第一章", "节"]),
                block("乙" * 300, "b", ["第二章", "节"])]
        self.assertEqual(len(self.splitter.split_documents(docs)), 2)

    def test_06_long_text_recursively_splits_under_hard_limit(self):
        docs = [block(("一段正文。" * 400), "a", ["长章节"])]
        chunks = self.splitter.split_documents(docs)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(item["text"]) <= chunker.HARD_MAX_CHARS for item in chunks))

    def test_07_formula_is_never_split(self):
        formula = "E=" + "x" * 1300
        chunks = self.splitter.split_documents([block(formula, "f", ["公式"], "equation")])
        self.assertEqual(len(chunks), 1)
        self.assertIn(formula, chunks[0]["text"])
        self.assertTrue(chunks[0]["metadata"]["atomic"])

    def test_08_formula_stays_with_short_explanation(self):
        docs = [block("公式解释" * 40, "t", ["公式"]),
                block("x^2+y^2=z^2", "f", ["公式"], "equation")]
        chunks = self.splitter.split_documents(docs)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["metadata"]["source_block_ids"], ["t", "f"])

    def test_09_oversized_atomic_formula_is_reported(self):
        self.splitter.split_documents([block("F" * 1700, "f", ["公式"], "formula")])
        self.assertEqual(self.splitter.last_report["above_hard_maximum"], 1)
        self.assertIn("formula", self.splitter.last_report["above_hard_maximum_types"])

    def test_10_small_table_is_atomic(self):
        chunk = self.splitter.split_documents([block("列A | 列B\n1 | 2", "tb", ["表"], "table")])[0]
        self.assertEqual(chunk["metadata"]["table_part_count"], 1)
        self.assertTrue(chunk["metadata"]["atomic"])

    def test_11_long_table_splits_by_rows_and_repeats_header(self):
        table = "列A | 列B\n" + "\n".join(f"第{i}行 | " + "值" * 110 for i in range(30))
        chunks = self.splitter.split_documents([block(table, "tb", ["表"], "table")])
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all("列A | 列B" in item["text"] for item in chunks))
        self.assertEqual([item["metadata"]["table_part_index"] for item in chunks],
                         list(range(1, len(chunks) + 1)))
        self.assertTrue(all(item["metadata"]["table_id"] == "tb" for item in chunks))

    def test_12_caption_binds_to_table(self):
        docs = [block("表 1：指标", "cap", ["表"], "table_caption"),
                block("名称 | 数值\nA | 1", "tb", ["表"], "table")]
        chunk = self.splitter.split_documents(docs)[0]
        self.assertIn("表 1：指标", chunk["text"])
        self.assertEqual(chunk["metadata"]["source_block_ids"], ["cap", "tb"])

    def test_13_metadata_contains_trace_and_page_range(self):
        docs = [block("甲" * 500, "a", ["章"], page=2),
                block("乙" * 500, "b", ["章"], page=3)]
        meta = self.splitter.split_documents(docs)[0]["metadata"]
        for key in ("source", "section_path", "section_paths", "page_start", "page_end",
                    "source_block_ids", "block_types"):
            self.assertIn(key, meta)
        self.assertEqual((meta["page"], meta["page_start"], meta["page_end"]), (2, 2, 3))

    def test_14_legacy_strategy_remains_available(self):
        legacy = chunker.TextSplitter(80, 10, section_aware=False)
        chunks = legacy.split_documents([block("中" * 200, "a")])
        self.assertTrue(all(len(item["text"]) <= 80 for item in chunks))
        self.assertEqual(legacy.last_report["strategy"], "legacy")

    def test_15_output_and_ids_are_stable_and_input_is_not_mutated(self):
        docs = [block("稳定内容。" * 100, "a", ["章"])]
        original = copy.deepcopy(docs)
        first = self.splitter.split_documents(docs)
        second = self.splitter.split_documents(docs)
        self.assertEqual(first, second)
        self.assertEqual(docs, original)

    def test_16_excluded_blocks_are_skipped_without_losing_valid_ids(self):
        excluded = block("页眉", "noise", ["章"])
        excluded["metadata"]["excluded_from_retrieval"] = True
        chunks = self.splitter.split_documents([block("有效正文", "body", ["章"]), excluded])
        self.assertEqual(chunks[0]["metadata"]["source_block_ids"], ["body"])
        self.assertTrue(self.splitter.last_report["integrity"]["passed"])

    def test_16a_multiple_formulas_remain_complete_and_ordered(self):
        docs = [block("f_1=" + "x" * 500, "f1", ["公式"], "equation"),
                block("f_2=" + "y" * 500, "f2", ["公式"], "equation")]
        chunks = self.splitter.split_documents(docs)
        combined = "\n".join(item["text"] for item in chunks)
        self.assertIn("f_1=" + "x" * 500, combined)
        self.assertIn("f_2=" + "y" * 500, combined)
        self.assertLess(combined.index("f_1="), combined.index("f_2="))

    def test_16b_deep_long_section_prefix_counts_toward_hard_limit(self):
        path = [str(index) + "级" * 45 for index in range(6)]
        chunks = self.splitter.split_documents([block("正文。" * 600, "body", path)])
        self.assertTrue(all(len(item["text"]) <= chunker.HARD_MAX_CHARS for item in chunks))

    def test_17_postprocessor_marks_only_independent_formula_and_table_atomic(self):
        raw = [block("行内公式 x+y 仍是正文", "t", kind="text"),
               block("x+y=z", "f", kind="equation"), block("A | B", "tb", kind="table")]
        refined, report = postprocess_document(raw)
        self.assertFalse(report["fail_open"])
        self.assertEqual([item["metadata"]["atomic"] for item in refined], [False, True, True])

    def test_18_audit_statistics_and_content_markers(self):
        docs = [block("唯一甲" + "甲" * 700, "a", ["章", "一"]),
                block("唯一乙" + "乙" * 700, "b", ["章", "二"])]
        chunks = self.splitter.split_documents(docs)
        combined = "\n".join(item["text"] for item in chunks)
        self.assertIn("唯一甲", combined)
        self.assertIn("唯一乙", combined)
        report = self.splitter.last_report
        self.assertEqual(report["original_blocks"], 2)
        self.assertEqual(report["chunks"], len(chunks))
        self.assertTrue(report["integrity"]["passed"])


if __name__ == "__main__":
    unittest.main()
