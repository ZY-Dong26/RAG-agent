"""问答流程中的答案后处理规则。

输入是模型返回的完整答案及实际送入模型的证据，输出是最终正文和可展示的引用。
普通和流式生成结束后共用该规则；快照保存由同层 snapshot.py 负责，SQL 留在
storage/。本模块不决定检索证据是否足够，也不再次调用模型。
"""
import re


# 模型可能在拒答句后追加引用编号；只匹配整段，避免把正文中的引用误判为拒答。
_REFUSAL_PATTERN = re.compile(r"^资料中没有找到相关内容[。.!！]?(?:\s*\[\d+\])*\s*$")
_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def is_generation_refusal(answer: str) -> bool:
    """仅在完整答案等于约定拒答句（可带引用编号）时返回 True。"""
    return bool(_REFUSAL_PATTERN.fullmatch(answer))


def finalize_answer(answer: str, prompt_hits: list) -> tuple[str, list, int]:
    """整理最终正文，只返回模型见过且答案实际引用的证据。

    输入的 prompt_hits 已按提示词预算筛选；输出依次为最终正文、引用证据及被移除的
    无效编号数量。无效的 ``[数字]`` 不保留为看似可信的引用；不改写其他正文。
    拒答句不关联来源，即使模型在句尾附了编号，也清除这些编号。
    """
    if is_generation_refusal(answer):
        return _CITATION_PATTERN.sub("", answer).strip(), [], 0

    available = {str(hit["metadata"].get("rank")): hit for hit in prompt_hits
                 if hit["metadata"].get("rank") is not None}
    cited = set()
    invalid_count = 0

    def keep_valid(match):
        nonlocal invalid_count
        number = match.group(1)
        if number in available:
            cited.add(number)
            return match.group(0)
        invalid_count += 1
        return ""

    cleaned = _CITATION_PATTERN.sub(keep_valid, answer).strip()
    cited_hits = [hit for hit in prompt_hits if str(hit["metadata"].get("rank")) in cited]
    return cleaned, cited_hits, invalid_count
