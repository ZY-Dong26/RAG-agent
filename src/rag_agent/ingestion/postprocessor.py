"""
postprocessor.py —— MinerU 统一文档记录的本地规则后处理层

职责：
    1. 在 MinerU 适配完成后，对块字段和普通文本做确定性标准化。
    2. 保守识别重复页眉、页脚、页码，以及高置信度跨页正文续接。
    3. 用集中维护的编号规则识别标题等级，并为后续块构建章节路径。
    4. 返回可审计报告；任一意外异常或完整性校验失败时 fail-open，返回原输入深拷贝。

边界：
    - 本模块完全本地，不调用 LLM，也不导入向量库、Embedding、BM25、检索器或回答模型。
    - 不进行跨页表格合并，不改变表格、公式、图片、代码等特殊内容的原文。
    - 不负责切片；切片器只消费本模块输出，并跳过 excluded_from_retrieval=true 的块。
"""
import copy
import hashlib
import json
import re

from rag_agent.common.progress import report as notify


POSTPROCESSOR_VERSION = "rules-v1"

# 规则参数集中在模块内，V1 不增加环境变量。页眉页脚至少需要三页文档才有足够证据。
MIN_REPEAT_PAGES = 3
REPEAT_PAGE_RATIO = 0.60
MAX_TITLE_LENGTH = 100

# 表格、公式、图片和代码的字符串可能是 HTML、Markdown 或 LaTeX，基础标准化不得改写它们。
PROTECTED_TEXT_TYPES = {"table", "formula", "equation", "interline_equation", "image", "chart", "code"}
NON_PARAGRAPH_TYPES = PROTECTED_TEXT_TYPES | {
    "title", "caption", "list", "page_footnote", "footnote", "ref_text", "reference",
    "header", "footer", "page_number", "discarded",
}
PRESERVED_COUNT_TYPES = {
    "table", "formula", "equation", "interline_equation", "page_footnote", "footnote",
    "ref_text", "reference",
}
KNOWN_TYPES = NON_PARAGRAPH_TYPES | {"text", "paragraph"}

# 标题编号规则按用途集中维护，避免判断散落在处理流程中。
_CHINESE_NUMBER = r"[一二三四五六七八九十百千零〇两]+"
_TITLE_RULES = (
    ("chinese_chapter", re.compile(rf"^第(?:\d+|{_CHINESE_NUMBER})章(?:\s*[:：、.]?\s*.*)?$"), 1),
    ("chinese_section", re.compile(rf"^第(?:\d+|{_CHINESE_NUMBER})节(?:\s*[:：、.]?\s*.*)?$"), 2),
    ("arabic_hierarchy", re.compile(r"^(\d+(?:\.\d+){0,5})(?:[.、])?\s+\S.+$"), None),
    ("arabic_compact", re.compile(r"^(\d+(?:\.\d+){0,5})[.、]([A-Z\u3400-\u9fff]\S*)$"), None),
    ("chinese_major", re.compile(rf"^{_CHINESE_NUMBER}、\s*\S.+$"), 1),
    ("chinese_parenthesized", re.compile(rf"^[（(]{_CHINESE_NUMBER}[）)]\s*\S.+$"), 2),
)
_WEAK_TITLE_RULES = (
    ("arabic_parenthesized", re.compile(r"^(?:[（(]\d+[）)]|\d+[）)])\s*\S.+$"), 3),
    ("circled_number", re.compile(r"^[①②③④⑤⑥⑦⑧⑨⑩]\s*\S.+$"), 3),
)
_HEADING_PREFIX = re.compile(
    rf"^(?:第(?:\d+|{_CHINESE_NUMBER})[章节]|\d+(?:\.\d+)*[.、]?\s+|" +
    rf"\d+(?:\.\d+)*[.、][A-Z\u3400-\u9fff]|{_CHINESE_NUMBER}、|"
    rf"[（(](?:\d+|{_CHINESE_NUMBER})[）)]|\d+[）)]|[①②③④⑤⑥⑦⑧⑨⑩])"
)
_CAPTION_PREFIX = re.compile(r"^(?:图|表|Figure|Table)\s*[A-Za-z0-9一二三四五六七八九十.-]+", re.I)
_REFERENCE_PREFIX = re.compile(r"^\s*(?:\[\d+\]|\d+[.)、])\s*")
_TERMINATORS = tuple("。！？.!?；;：:")
_PAGE_ONLY = re.compile(r"^\s*(?:[-—–]\s*)?(?:\d+|[ivxlcdm]+)(?:\s*[-—–])?\s*$", re.I)
_ROMAN_TOKEN = re.compile(r"(?<![A-Za-z])[ivxlcdm]+(?![A-Za-z])", re.I)
_ARABIC_TOKEN = re.compile(r"(?<![A-Za-z0-9])\d+(?![A-Za-z0-9])")
_CJK = re.compile(r"[\u3400-\u9fff]")
_CONTINUATION_PREFIXES = ("的", "了", "和", "与", "及", "或", "而", "并", "但", "则", "其", "这", "该", "其中", "以及", "同时")


def _new_report(count):
    """创建字段稳定的审计报告；失败分支也返回同一结构，方便调用方和人工比较。"""
    return {
        "schema_version": 1,
        "postprocessor_version": POSTPROCESSOR_VERSION,
        "input_blocks": count,
        "output_blocks": count,
        "normalized_blocks": 0,
        "excluded_headers": 0,
        "excluded_footers": 0,
        "excluded_page_numbers": 0,
        "merged_cross_page_paragraphs": 0,
        "promoted_titles": 0,
        "assigned_title_levels": 0,
        "warnings": [],
        "operations": [],
        "fail_open": False,
    }


def _operation(report, block, operation, rule, reason, input_ids=None):
    """同时写块级操作名和文档级审计记录，不在报告中复制正文。"""
    metadata = block["metadata"]
    metadata.setdefault("postprocess_operations", []).append(operation)
    report["operations"].append({
        "operation": operation,
        "input_block_ids": list(input_ids or metadata["source_block_ids"]),
        "output_block_id": metadata["block_id"],
        "page_start": metadata["page_start"],
        "page_end": metadata["page_end"],
        "rule": rule,
        "reason": reason,
    })


def _stable_block_id(source, page, index, text):
    """旧记录缺少 block_id 时生成确定性 ID；相同输入重复运行得到相同结果。"""
    payload = json.dumps([source, page, index, text], ensure_ascii=False, separators=(",", ":"))
    return "block_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _prepare_blocks(raw_document, report):
    """深拷贝输入并补齐当前数据模型需要的元数据，不删除任何未知顶层或 metadata 字段。"""
    blocks = copy.deepcopy(raw_document)
    for index, block in enumerate(blocks):
        if not isinstance(block, dict) or not isinstance(block.get("metadata", {}), dict):
            raise ValueError("文档块或 metadata 格式无效")
        metadata = block.setdefault("metadata", {})
        text = block.get("text", "")
        if not isinstance(text, str):
            raise ValueError("文档块 text 必须是字符串")
        page = metadata.get("page_start", metadata.get("page", block.get("page_start", 1)))
        if type(page) is not int:
            raise ValueError("文档块页码必须是整数")
        block_id = metadata.get("block_id") or _stable_block_id(
            metadata.get("source", ""), page, index, text)
        block_type = str(metadata.get("block_type", block.get("type", "text")) or "unknown")
        source_ids = metadata.get("source_block_ids")
        if not isinstance(source_ids, list) or not source_ids:
            source_ids = [block_id]
        metadata.update({
            "block_id": block_id,
            "source_block_ids": list(source_ids),
            "block_type": block_type,
            "page_start": page,
            "page_end": metadata.get("page_end", page),
            "title_level": metadata.get("title_level", metadata.get("text_level")),
            "section_path": list(metadata.get("section_path") or []),
            "excluded_from_retrieval": bool(metadata.get("excluded_from_retrieval", False)),
            "excluded_reason": metadata.get("excluded_reason"),
            "postprocess_operations": list(metadata.get("postprocess_operations") or []),
        })
        if block_type not in KNOWN_TYPES:
            report["warnings"].append({"type": "unknown_block_type", "block_id": block_id,
                                       "block_type": block_type})
    return blocks


def _normalize_plain_text(text):
    """压缩普通空格并保留换行；不改写词语、不合并段落，也不删除非空正文字符。"""
    text = text.strip()
    return "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines())


def _normalize_blocks(blocks, report):
    """只标准化普通文本；特殊内容原样保留，空普通文本仅标记排除而不物理删除。"""
    for block in blocks:
        metadata = block["metadata"]
        block_type = metadata["block_type"]
        if block_type not in PROTECTED_TEXT_TYPES:
            normalized = _normalize_plain_text(block["text"])
            if normalized != block["text"]:
                block["text"] = normalized
                report["normalized_blocks"] += 1
                _operation(report, block, "normalize_text", "collapse_plain_spaces",
                           "去除首尾空白并压缩普通空格")
        if not block["text"].strip() and block_type in {"text", "paragraph", "title", "header", "footer", "page_number"}:
            metadata["excluded_from_retrieval"] = True
            metadata["excluded_reason"] = metadata.get("excluded_reason") or "empty_text"


def _repeat_key(text):
    """生成页眉页脚模板键：折叠空白，并把独立阿拉伯/罗马页码统一成 <PAGE>。"""
    value = re.sub(r"\s+", " ", text).strip().casefold()
    value = _ROMAN_TOKEN.sub("<PAGE>", value)
    return _ARABIC_TOKEN.sub("<PAGE>", value)


def _bbox_position(metadata):
    """bbox 信息可靠时返回 header/footer；无法确认坐标尺度时返回 None 走页内顺序回退。"""
    bbox = metadata.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    try:
        y0, y1 = float(bbox[1]), float(bbox[3])
        height = metadata.get("page_height")
        if height is not None:
            y0, y1 = y0 / float(height), y1 / float(height)
        elif not (0 <= y0 <= 1 and 0 <= y1 <= 1):
            return None
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if y1 <= 0.20:
        return "header"
    if y0 >= 0.80:
        return "footer"
    # "body" 表示 bbox 可靠且明确位于正文区；与 None（缺少可靠 bbox）区分，
    # 防止正文块又回退为“每页前后两个块”的候选。
    return "body"


def _candidate_position(block, index, page_blocks):
    """优先使用明确类型/bbox；缺少可靠版面信息时，只取每页前后两个块作为保守候选。"""
    block_type = block["metadata"]["block_type"]
    if block_type == "header":
        return "header"
    if block_type in {"footer", "page_number"}:
        return "footer"
    positioned = _bbox_position(block["metadata"])
    if positioned in {"header", "footer"}:
        return positioned
    if positioned == "body":
        return None
    if index < 2:
        return "header"
    if index >= max(0, len(page_blocks) - 2):
        return "footer"
    return None


def _exclude_repeated_margins(blocks, report):
    """按不同页面统计稳定位置的重复模板，只排除实际命中位置的块，不按全文内容批量删除。"""
    pages = {}
    for block in blocks:
        pages.setdefault(block["metadata"]["page_start"], []).append(block)
    if len(pages) < MIN_REPEAT_PAGES:
        return

    candidates = []
    page_hits = {}
    for page, page_blocks in pages.items():
        for index, block in enumerate(page_blocks):
            if not block["text"].strip():
                continue
            position = _candidate_position(block, index, page_blocks)
            if not position:
                continue
            key = (position, _repeat_key(block["text"]))
            candidates.append((block, key))
            page_hits.setdefault(key, set()).add(page)

    accepted = {key for key, hit_pages in page_hits.items()
                if len(hit_pages) >= MIN_REPEAT_PAGES or len(hit_pages) / len(pages) >= REPEAT_PAGE_RATIO}
    for block, key in candidates:
        if key not in accepted:
            continue
        metadata = block["metadata"]
        if metadata["excluded_from_retrieval"] and metadata.get("excluded_reason") != "empty_text":
            continue
        if metadata["block_type"] == "page_number" or _PAGE_ONLY.fullmatch(block["text"]):
            reason, counter = "page_number", "excluded_page_numbers"
        elif key[0] == "header":
            reason, counter = "repeated_header", "excluded_headers"
        else:
            reason, counter = "repeated_footer", "excluded_footers"
        metadata["excluded_from_retrieval"] = True
        metadata["excluded_reason"] = reason
        report[counter] += 1
        _operation(report, block, "exclude_repeated_margin", "repeated_margin_template", reason)


def _title_level(text, original_title):
    """返回 (等级, 规则名)；弱编号只接受 MinerU 已标为标题的块，避免把列表项提升为标题。"""
    for name, pattern, fixed_level in _TITLE_RULES:
        match = pattern.fullmatch(text)
        if not match:
            continue
        if name in {"arabic_hierarchy", "arabic_compact"}:
            return min(6, match.group(1).count(".") + 1), name
        return fixed_level, name
    if original_title:
        for name, pattern, level in _WEAK_TITLE_RULES:
            if pattern.fullmatch(text):
                return level, name
    return None, None


def _classify_titles(blocks, report):
    """保留原有标题；只把编号明确、长度合理且不是图表注/参考文献/列表的普通文本提升为标题。"""
    for block in blocks:
        metadata, text = block["metadata"], block["text"].strip()
        block_type = metadata["block_type"]
        original_title = block_type == "title"
        if metadata["excluded_from_retrieval"] or not text or len(text) > MAX_TITLE_LENGTH:
            continue
        if (block_type in {"list", "caption", "ref_text", "reference", "page_footnote", "footnote"}
                or _CAPTION_PREFIX.match(text) or text.endswith(("。", ".", "！", "!", "？", "?"))):
            continue
        level, rule = _title_level(text, original_title)
        if level is None:
            # MinerU 已标标题但规则无法确认时，保留可信的 1～6 原等级；否则保持空值。
            if original_title and type(metadata.get("title_level")) is int and 1 <= metadata["title_level"] <= 6:
                continue
            if original_title:
                metadata["title_level"] = None
            continue
        if not original_title:
            # 参考文献编号和普通列表由类型先阻断；额外拒绝只有编号前缀、没有标题语义的短项。
            if _REFERENCE_PREFIX.match(text) and len(text) < 8 and not _HEADING_PREFIX.match(text):
                continue
            metadata["block_type"] = "title"
            report["promoted_titles"] += 1
            _operation(report, block, "promote_title", rule, "高置信度章节编号")
        if metadata.get("title_level") != level:
            metadata["title_level"] = level
            report["assigned_title_levels"] += 1
            _operation(report, block, "assign_title_level", rule, f"规则标题等级 {level}")


def _paragraph(block):
    """跨页合并只接受未排除的普通正文，未知类型和所有特殊结构一律不参与。"""
    metadata = block["metadata"]
    return (not metadata["excluded_from_retrieval"]
            and metadata["block_type"] in {"text", "paragraph"}
            and bool(block["text"].strip()))


def _merge_text(previous, following):
    """判断高置信度续接并返回 (合并文本, 规则名)；不确定时返回 (None, None)。"""
    left, right = previous.rstrip(), following.lstrip()
    if not left or not right or left.endswith(_TERMINATORS):
        return None, None
    if _HEADING_PREFIX.match(right) or (len(right) <= 12 and right.endswith(_TERMINATORS)):
        return None, None
    if re.search(r"[A-Za-z]+-$", left) and re.match(r"[A-Za-z]", right):
        return left[:-1] + right, "english_hyphenated_word"
    if re.search(r"[A-Za-z]$", left) and re.match(r"[a-z]", right):
        return left + " " + right, "english_lowercase_continuation"
    if (_CJK.search(left[-1:]) or left.endswith(("，", "、"))) and _CJK.match(right):
        if left.endswith(("，", "、")) or right.startswith(_CONTINUATION_PREFIXES):
            return left + right, "chinese_conservative_continuation"
    return None, None


def _merged_id(source_ids):
    """合并块 ID 只依赖有序来源 ID，保证重复执行结果一致。"""
    digest = hashlib.sha256("\0".join(source_ids).encode("utf-8")).hexdigest()[:20]
    return "merged_" + digest


def _merge_cross_page_paragraphs(blocks, report):
    """每对相邻页最多考察“上一页最后有效块 + 下一页第一有效块”，且原块最多参与一次。"""
    pages = {}
    for index, block in enumerate(blocks):
        pages.setdefault(block["metadata"]["page_start"], []).append((index, block))
    pairs = {}
    consumed = set()
    for page in sorted(pages):
        if page + 1 not in pages:
            continue
        left_valid = [(i, b) for i, b in pages[page] if not b["metadata"]["excluded_from_retrieval"]]
        right_valid = [(i, b) for i, b in pages[page + 1] if not b["metadata"]["excluded_from_retrieval"]]
        if not left_valid or not right_valid:
            continue
        left_index, left = left_valid[-1]
        right_index, right = right_valid[0]
        if left_index in consumed or right_index in consumed or not (_paragraph(left) and _paragraph(right)):
            continue
        merged_text, rule = _merge_text(left["text"], right["text"])
        if not merged_text:
            continue
        source_ids = list(dict.fromkeys(left["metadata"]["source_block_ids"]
                                        + right["metadata"]["source_block_ids"]))
        merged = copy.deepcopy(left)
        merged["text"] = merged_text
        merged["source_blocks"] = [copy.deepcopy(left), copy.deepcopy(right)]
        metadata = merged["metadata"]
        metadata["block_id"] = _merged_id(source_ids)
        metadata["source_block_ids"] = source_ids
        metadata["page_start"] = left["metadata"]["page_start"]
        metadata["page_end"] = right["metadata"]["page_end"]
        _operation(report, merged, "cross_page_paragraph_merge", rule,
                   "相邻页正文满足高置信度续接规则", input_ids=source_ids)
        pairs[left_index] = (right_index, merged)
        consumed.update((left_index, right_index))
        report["merged_cross_page_paragraphs"] += 1

    output = []
    skipped = {right for right, _ in pairs.values()}
    for index, block in enumerate(blocks):
        if index in pairs:
            output.append(pairs[index][1])
        elif index not in skipped:
            output.append(block)
    return output


def _assign_section_paths(blocks, report):
    """按标题等级维护章节栈；未知等级标题拥有可读路径，但不改变后续块的稳定章节上下文。"""
    stack = []
    for block in blocks:
        metadata = block["metadata"]
        if metadata["excluded_from_retrieval"]:
            metadata["section_path"] = list(stack)
            continue
        if metadata["block_type"] != "title":
            metadata["section_path"] = list(stack)
            continue
        level = metadata.get("title_level")
        if type(level) is not int:
            metadata["section_path"] = list(stack) + [block["text"]]
            continue
        if level > len(stack) + 1:
            corrected = min(6, len(stack) + 1)
            report["warnings"].append({"type": "title_level_gap", "block_id": metadata["block_id"],
                                       "original_level": level, "corrected_level": corrected})
            metadata["title_level"] = corrected
            level = corrected
        stack = stack[:level - 1]
        stack.append(block["text"])
        metadata["section_path"] = list(stack)


def _validate_document(input_blocks, output_blocks):
    """验证追溯、页码、标题等级、跨页合并和特殊内容数量；失败由公开入口统一 fail-open。"""
    input_ids = {block["metadata"]["block_id"] for block in input_blocks}
    output_ids = [block["metadata"]["block_id"] for block in output_blocks]
    if len(output_ids) != len(set(output_ids)):
        raise ValueError("输出 block_id 重复")
    traced = set()
    for block in output_blocks:
        metadata = block["metadata"]
        source_ids = metadata.get("source_block_ids")
        if not isinstance(source_ids, list) or not source_ids or not set(source_ids) <= input_ids:
            raise ValueError("source_block_ids 无法追溯")
        traced.update(source_ids)
        start, end = metadata.get("page_start"), metadata.get("page_end")
        if type(start) is not int or type(end) is not int or start < 1 or end < start:
            raise ValueError("页码范围无效")
        level = metadata.get("title_level")
        if level is not None and (type(level) is not int or not 1 <= level <= 6):
            raise ValueError("标题等级无效")
        if "cross_page_paragraph_merge" in metadata.get("postprocess_operations", []) and end != start + 1:
            raise ValueError("跨页段落只能合并相邻页")
    if traced != input_ids:
        raise ValueError("存在无法追溯的输入块")
    for block_type in PRESERVED_COUNT_TYPES:
        before = sum(block["metadata"]["block_type"] == block_type for block in input_blocks)
        after = sum(block["metadata"]["block_type"] == block_type for block in output_blocks)
        if before != after:
            raise ValueError(f"特殊内容数量变化：{block_type}")


def postprocess_document(raw_document, source=None):
    """
    对一份 MinerU 适配文档执行 V1 规则后处理。

    输入：同一 PDF 的统一文档记录列表；source 只用于简洁进度日志。
    输出：(refined_document, report)。成功时返回已校验的新结构；失败时返回原输入深拷贝，
    report.fail_open=true，且不暴露异常消息、URL 或正文。
    """
    original = copy.deepcopy(raw_document)
    audit = _new_report(len(original) if isinstance(original, list) else 0)
    label = str(source or "未命名文档")
    notify(f"[后处理] 文档 {label}：开始，共 {audit['input_blocks']} 个块")
    try:
        if not isinstance(raw_document, list):
            raise ValueError("文档必须是块列表")
        blocks = _prepare_blocks(raw_document, audit)
        _normalize_blocks(blocks, audit)
        _exclude_repeated_margins(blocks, audit)
        _classify_titles(blocks, audit)
        blocks = _merge_cross_page_paragraphs(blocks, audit)
        _assign_section_paths(blocks, audit)
        _validate_document(_prepare_blocks(raw_document, _new_report(len(raw_document))), blocks)
        audit["output_blocks"] = len(blocks)
        notify(f"[后处理] 页眉 {audit['excluded_headers']} 个，页脚 {audit['excluded_footers']} 个，"
               f"页码 {audit['excluded_page_numbers']} 个")
        notify(f"[后处理] 合并跨页段落 {audit['merged_cross_page_paragraphs']} 处，"
               f"标题分级 {audit['assigned_title_levels']} 个")
        notify(f"[后处理] 完成：输出 {len(blocks)} 个块")
        return blocks, audit
    except Exception as error:
        failed = _new_report(len(original) if isinstance(original, list) else 0)
        failed["fail_open"] = True
        failed["warnings"] = [{"type": "postprocess_failed", "error_type": type(error).__name__}]
        notify(f"[后处理失败] 文档 {label}：完整性校验失败，已回退原始解析结果")
        return original, failed
