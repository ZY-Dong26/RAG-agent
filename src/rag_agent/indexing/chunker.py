"""
chunker.py —— 章节感知的本地切块层

按连续章节组织内容，小章节仅和兼容兄弟合并；正文递归切分，公式保持原子性，
长表格只按行拆分。本模块不解析 PDF、不调用模型、不计算向量，也不跨文档合并。
"""
import hashlib
import json
import math
import re
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter


CHUNKER_VERSION = "section-rules-v1"
SECTION_AWARE_CHUNKING = True  # 简单回退开关：False 恢复旧的逐块切分。
MIN_CHUNK_CHARS = 800
TARGET_CHUNK_CHARS = 1000
SOFT_MAX_CHARS = 1200
HARD_MAX_CHARS = 1500
SECTION_SEPARATORS = ["\n\n", "。", "；", "\n", "！", "？", "，", " ", ""]
FORMULA_TYPES = {"formula", "equation", "interline_equation"}
TABLE_TYPES = {"table"}
CAPTION_TYPES = {"caption", "table_caption", "figure_caption", "image_caption"}
REFERENCE_TYPES = {"ref_text", "reference", "references", "bibliography"}
HARD_BOUNDARY_TYPES = TABLE_TYPES | REFERENCE_TYPES | {"code"}
OBJECT_TYPES = TABLE_TYPES | FORMULA_TYPES | {"image", "chart"}
_APPENDIX_RE = re.compile(r"^(?:附录|appendix)\b", re.I)


def chunking_signature():
    """返回所有会改变 chunk 边界的规则参数，供建库流水线签名使用。"""
    return {"version": CHUNKER_VERSION, "enabled": SECTION_AWARE_CHUNKING,
            "min": MIN_CHUNK_CHARS, "target": TARGET_CHUNK_CHARS,
            "soft_max": SOFT_MAX_CHARS, "hard_max": HARD_MAX_CHARS,
            "separators": SECTION_SEPARATORS}


def _ordered(values):
    """按首次出现顺序去重，使元数据和审计报告可重复。"""
    return list(dict.fromkeys(value for value in values if value is not None))


def _percentile(values, fraction):
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def _common_path(paths):
    paths = [list(path) for path in paths if path]
    if not paths:
        return []
    common = []
    for values in zip(*paths):
        if len(set(values)) != 1:
            break
        common.append(values[0])
    return common or list(paths[0])


def _composite_id(source_ids):
    payload = json.dumps(source_ids, ensure_ascii=False, separators=(",", ":"))
    return "chunk_blocks_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


class TextSplitter:
    """兼容旧接口的切分器；主建库显式启用 section_aware。"""

    def __init__(self, chunk_size=TARGET_CHUNK_CHARS, chunk_overlap=150,
                 section_aware=False):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.section_aware = section_aware
        self.last_report = {}
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", "：", "，", " ", ""])
        # 章节策略先切为无重叠中间单位，再按章节装箱；这样审计不会因重叠产生正文副本。
        self.section_splitter = RecursiveCharacterTextSplitter(
            chunk_size=SOFT_MAX_CHARS, chunk_overlap=0, separators=SECTION_SEPARATORS)

    def split_documents(self, documents):
        """输入统一文档块，返回可向量化 chunk；排除块不会进入任一策略。"""
        if not self.section_aware:
            chunks = self._split_legacy(documents)
            self.last_report = self._report(documents, chunks, [], "legacy")
            return chunks
        return self._split_by_section(documents)

    def _split_legacy(self, documents):
        """原来的逐块切分行为，保留作为明确回退路径。"""
        chunks = []
        for doc in documents:
            text, metadata = doc["text"], doc["metadata"]
            if metadata.get("excluded_from_retrieval"):
                continue
            parts = (self._split_table_legacy(text) if metadata.get("block_type") == "table"
                     else self.splitter.split_text(text))
            for part in parts:
                chunks.append({"text": part, "metadata": {
                    **{key: value for key, value in metadata.items() if key != "mineru_block"},
                    "chunk_id": len(chunks)}})
        return chunks

    def _split_by_section(self, documents):
        valid = [doc for doc in documents
                 if not doc.get("metadata", {}).get("excluded_from_retrieval")
                 and str(doc.get("text", "")).strip()]
        sections = self._continuous_sections(valid)
        regions, unmergeable = self._merge_small_sections(sections)
        chunks = []
        for region in regions:
            chunks.extend(self._pack_region(self._units(region["blocks"]),
                                            region["section_paths"]))
        for index, chunk in enumerate(chunks):
            chunk["metadata"]["chunk_id"] = index
        self.last_report = self._report(documents, chunks, unmergeable, "section", len(sections))
        if not self.last_report["integrity"]["passed"]:
            raise ValueError("章节切块完整性校验失败，拒绝生成不完整索引")
        return chunks

    @staticmethod
    def _continuous_sections(documents):
        """同名章节只有在输入中连续才组合，绝不做跨文档的全局聚合。"""
        result = []
        for doc in documents:
            meta = doc["metadata"]
            key = (meta.get("document_id", meta.get("source")),
                   tuple(meta.get("section_path") or []))
            if not result or result[-1]["key"] != key:
                result.append({"key": key, "path": list(key[1]), "blocks": []})
            result[-1]["blocks"].append(doc)
        for section in result:
            section["characters"] = sum(len(block["text"]) for block in section["blocks"])
            types = {block["metadata"].get("block_type", "unknown") for block in section["blocks"]}
            section["hard"] = bool(types & HARD_BOUNDARY_TYPES or any(
                _APPENDIX_RE.match(str(value).strip()) for value in section["path"]))
        return result

    @staticmethod
    def _compatible(left, right):
        """仅允许相邻同级兄弟合并；层级上升、主标题变化、特殊区域都是硬边界。"""
        if left["hard"] or right["hard"] or left["key"][0] != right["key"][0]:
            return False
        a, b = left["path"], right["path"]
        if len(b) < len(a) or (a and b and a[0] != b[0]):
            return False
        return (not a and not b) or (len(a) == len(b) and a[:-1] == b[:-1])

    def _merge_small_sections(self, sections):
        """把不足 800 字的小节向后并入兼容兄弟，达到目标长度或遇边界即停止。"""
        regions, unmergeable, index = [], [], 0
        while index < len(sections):
            selected, size = [sections[index]], sections[index]["characters"]
            while index + 1 < len(sections):
                candidate = sections[index + 1]
                if not (size < MIN_CHUNK_CHARS or candidate["characters"] < MIN_CHUNK_CHARS):
                    break
                if not self._compatible(selected[-1], candidate):
                    break
                if size + candidate["characters"] > HARD_MAX_CHARS:
                    break
                selected.append(candidate)
                size += candidate["characters"]
                index += 1
                if size >= TARGET_CHUNK_CHARS:
                    break
            if size < MIN_CHUNK_CHARS:
                unmergeable.append({"section_path": selected[0]["path"], "characters": size,
                                    "reason": "hard_boundary" if selected[-1]["hard"]
                                    else "no_compatible_sibling"})
            regions.append({"blocks": [block for section in selected for block in section["blocks"]],
                            "section_paths": _ordered(tuple(section["path"]) for section in selected)})
            index += 1
        return regions, unmergeable

    def _units(self, blocks):
        """生成装箱单位：图注绑定相邻对象，公式/表格/图片不做字符级拆分。"""
        units, index = [], 0
        while index < len(blocks):
            current = blocks[index]
            kind = current["metadata"].get("block_type", "unknown")
            bound = [current]
            if kind in CAPTION_TYPES and index + 1 < len(blocks):
                if blocks[index + 1]["metadata"].get("block_type") in OBJECT_TYPES:
                    bound.append(blocks[index + 1]); index += 1
            elif kind in OBJECT_TYPES and index + 1 < len(blocks):
                if blocks[index + 1]["metadata"].get("block_type") in CAPTION_TYPES:
                    bound.append(blocks[index + 1]); index += 1
            unit = self._make_unit(bound)
            if "table" in unit["block_types"]:
                units.extend(self._split_table_unit(unit))
            elif unit["atomic"] or len(unit["text"]) <= SOFT_MAX_CHARS:
                units.append(unit)
            else:
                units.extend({**unit, "text": part, "atomic": False}
                             for part in self.section_splitter.split_text(unit["text"]))
            index += 1
        return units

    @staticmethod
    def _make_unit(blocks):
        meta = blocks[0]["metadata"]
        types = _ordered(block["metadata"].get("block_type", "unknown") for block in blocks)
        ids = _ordered(source_id for block in blocks for source_id in block["metadata"].get(
            "source_block_ids", [block["metadata"].get("block_id")]))
        table_block = next((block for block in blocks
                            if block["metadata"].get("block_type") == "table"), None)
        table_id = table_block["metadata"].get("block_id") if table_block else None
        return {"text": "\n".join(block["text"] for block in blocks if block["text"].strip()),
                "metadata": meta, "source_block_ids": ids, "block_types": types,
                "pages": [(block["metadata"].get("page_start", block["metadata"].get("page", 1)),
                           block["metadata"].get("page_end", block["metadata"].get("page", 1)))
                          for block in blocks],
                "atomic": bool(meta.get("atomic")) or bool(set(types) & OBJECT_TYPES),
                "table_id": table_id,
                "table_metadata": table_block["metadata"] if table_block else {}}

    @staticmethod
    def _split_table_unit(unit):
        """长表按行打包并重复图注/表头；无法识别行时整表保留并允许超过软上限。"""
        table_id = unit["table_id"]
        if len(unit["text"]) <= SOFT_MAX_CHARS:
            return [{**unit, "table_id": table_id, "table_part_index": 1,
                     "table_part_count": 1}]
        lines = [line.strip() for line in unit["text"].splitlines() if line.strip()]
        if len(lines) < 2:
            return [{**unit, "table_id": table_id, "table_part_index": 1,
                     "table_part_count": 1}]
        # 优先从 MinerU 原始表格结构识别图注行数，再额外重复一行表头。
        raw = unit["table_metadata"].get("mineru_block")
        caption_count = 1 if set(unit["block_types"]) & CAPTION_TYPES else 0
        if isinstance(raw, dict):
            caption = raw.get("table_caption")
            if isinstance(caption, str) and caption.strip():
                caption_count = max(caption_count, len(caption.splitlines()))
            elif isinstance(caption, list):
                caption_count = max(caption_count, sum(bool(str(value).strip()) for value in caption))
            children = raw.get("blocks")
            if isinstance(children, list):
                caption_count = max(caption_count, sum(
                    isinstance(child, dict) and child.get("type") == "table_caption"
                    for child in children))
        prefix_count = min(max(1, caption_count + 1), len(lines) - 1)
        prefix, parts, current = lines[:prefix_count], [], []
        for row in lines[prefix_count:]:
            if current and len("\n".join([*prefix, *current, row])) > SOFT_MAX_CHARS:
                parts.append("\n".join([*prefix, *current])); current = []
            current.append(row)
        if current:
            parts.append("\n".join([*prefix, *current]))
        return [{**unit, "text": text, "table_id": table_id,
                 "table_part_index": index, "table_part_count": len(parts)}
                for index, text in enumerate(parts, 1)]

    def _pack_region(self, units, section_paths):
        """按来源顺序装箱；表格独占边界，公式可与前后解释共同装箱。"""
        main = _common_path([list(path) for path in section_paths])
        prefix_length = len("章节：" + (" > ".join(str(value) for value in main)
                                     if main else "未分章节")) + 1
        soft_limit = max(256, SOFT_MAX_CHARS - prefix_length)
        hard_limit = max(soft_limit, HARD_MAX_CHARS - prefix_length)
        # 深层长标题也计入总字符预算；只对普通文本中间单位再次细分。
        adjusted = []
        local_splitter = RecursiveCharacterTextSplitter(
            chunk_size=soft_limit, chunk_overlap=0, separators=SECTION_SEPARATORS)
        for unit in units:
            if not unit["atomic"] and len(unit["text"]) > soft_limit:
                adjusted.extend({**unit, "text": part} for part in local_splitter.split_text(unit["text"]))
            else:
                adjusted.append(unit)
        units = adjusted
        chunks, current = [], []
        for unit in units:
            is_table = "table" in unit["block_types"]
            if is_table and current:
                chunks.append(self._emit(current, section_paths)); current = []
            candidate = "\n\n".join(item["text"] for item in [*current, unit])
            if current and len(candidate) > soft_limit:
                if (len(candidate) <= hard_limit
                        and (len(self._join(current)) < MIN_CHUNK_CHARS
                             or len(unit["text"]) < MIN_CHUNK_CHARS)):
                    current.append(unit)
                else:
                    chunks.append(self._emit(current, section_paths)); current = [unit]
            else:
                current.append(unit)
            if is_table:
                chunks.append(self._emit(current, section_paths)); current = []
        if current:
            chunks.append(self._emit(current, section_paths))
        return chunks

    @staticmethod
    def _join(units):
        return "\n\n".join(unit["text"] for unit in units)

    def _emit(self, units, section_paths):
        first = units[0]["metadata"]
        paths = [list(path) for path in section_paths]
        main = _common_path(paths)
        prefix = "章节：" + (" > ".join(str(value) for value in main) if main else "未分章节")
        ids = _ordered(value for unit in units for value in unit["source_block_ids"])
        types = _ordered(value for unit in units for value in unit["block_types"])
        pages = [page for unit in units for pair in unit["pages"] for page in pair]
        metadata = {key: value for key, value in first.items()
                    if key not in {"mineru_block", "block_id", "source_block_ids", "section_path",
                                   "title_level", "excluded_from_retrieval", "excluded_reason"}}
        metadata.update({"block_id": _composite_id(ids), "source_block_ids": ids,
                         "block_types": types, "block_type": types[0] if len(types) == 1 else "mixed",
                         "section_path": main, "section_paths": paths,
                         "page": min(pages), "page_start": min(pages), "page_end": max(pages),
                         "atomic": len(units) == 1 and units[0]["atomic"]})
        if len(units) == 1 and units[0].get("table_id"):
            metadata.update({key: units[0][key] for key in
                             ("table_id", "table_part_index", "table_part_count")})
        return {"text": prefix + "\n" + self._join(units), "metadata": metadata}

    def _report(self, documents, chunks, unmergeable, strategy, sections=None):
        """生成可落盘审计统计，并以 source_block_ids 验证所有有效输入均可追溯。"""
        lengths = [len(chunk["text"]) for chunk in chunks]
        expected = _ordered(source_id for doc in documents if str(doc.get("text", "")).strip()
                            and not doc.get("metadata", {}).get("excluded_from_retrieval")
                            for source_id in doc["metadata"].get(
                                "source_block_ids", [doc["metadata"].get("block_id")]))
        actual = _ordered(source_id for chunk in chunks for source_id in
                          chunk["metadata"].get("source_block_ids", []))
        short_reasons = {}
        for chunk in chunks:
            if len(chunk["text"]) < MIN_CHUNK_CHARS:
                types = set(chunk["metadata"].get("block_types", []))
                reason = "atomic_or_hard_boundary" if types & (OBJECT_TYPES | HARD_BOUNDARY_TYPES) else "section_end"
                short_reasons[reason] = short_reasons.get(reason, 0) + 1
        return {"chunker_version": CHUNKER_VERSION, "strategy": strategy,
                "original_blocks": len(documents),
                "retrieval_blocks": sum(bool(str(doc.get("text", "")).strip())
                                        and not doc.get("metadata", {}).get("excluded_from_retrieval")
                                        for doc in documents),
                "continuous_sections": sections,
                "chunks": len(chunks),
                "lengths": {"min": min(lengths, default=0), "p50": _percentile(lengths, .5),
                            "p95": _percentile(lengths, .95), "max": max(lengths, default=0)},
                "below_minimum": sum(value < MIN_CHUNK_CHARS for value in lengths),
                "below_minimum_reasons": short_reasons,
                "above_hard_maximum": sum(value > HARD_MAX_CHARS for value in lengths),
                "above_hard_maximum_types": _ordered(
                    kind for chunk in chunks if len(chunk["text"]) > HARD_MAX_CHARS
                    for kind in chunk["metadata"].get("block_types", [])),
                "formula_blocks": sum(doc["metadata"].get("block_type") in FORMULA_TYPES
                                      for doc in documents),
                "table_blocks": sum(doc["metadata"].get("block_type") == "table"
                                    for doc in documents),
                "table_parts": sum("table" in chunk["metadata"].get("block_types", [])
                                   for chunk in chunks),
                "unmergeable_short_sections": unmergeable,
                "integrity": {"passed": expected == actual,
                              "expected_source_block_ids": expected,
                              "actual_source_block_ids": actual,
                              "missing_source_block_ids": [value for value in expected if value not in actual]}}

    def _split_table_legacy(self, text):
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) < 2 or len(lines[0]) >= self.chunk_size // 2:
            return self.splitter.split_text(text)
        prefix, result, current = lines[0], [], lines[0]
        for line in lines[1:]:
            if len(current) + len(line) + 1 <= self.chunk_size:
                current += "\n" + line; continue
            if current != prefix:
                result.append(current); current = prefix
            if len(prefix) + len(line) + 1 > self.chunk_size:
                budget = self.chunk_size - len(prefix) - 1
                result.extend(prefix + "\n" + line[start:start + budget]
                              for start in range(0, len(line), budget))
            else:
                current += "\n" + line
        if current != prefix:
            result.append(current)
        return result

    @staticmethod
    def save_chunks(chunks, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as stream:
            json.dump(chunks, stream, ensure_ascii=False, indent=2)
        print(f"已保存 {len(chunks)} 个 chunk → {output_path}")


if __name__ == "__main__":
    from rag_agent import config
    from rag_agent.ingestion import pypdf_fallback

    docs = pypdf_fallback.load_all_pdfs(config.RAW_PDF_DIR)
    splitter = TextSplitter(config.CHUNK_SIZE, config.CHUNK_OVERLAP,
                            section_aware=SECTION_AWARE_CHUNKING)
    result = splitter.split_documents(docs)
    TextSplitter.save_chunks(result, config.CHUNKS_FILE)
    print(f"总共生成chunk数量：{len(result)}")
