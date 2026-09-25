"""
judger.py —— 调用独立裁判模型对已有评测批次逐题打分

职责：
    1. 读取批次目录下 items/*.json 中已有的题目记录（问题、参考答案、评分标准、RAG 回答、检索证据）。
    2. 把题目材料组装成裁判请求，调用独立裁判模型，按正确性、完整性、相关性三个维度各打 0–3 分。
    3. 每题判完立刻原子写回 item 文件；已判题自动跳过，支持中断后续跑。
    4. 判分结束后刷新批次的全部导出产物（CSV、报告、汇总表）。

设计原因：
    - 裁判模型是独立的开发工具服务，本模块不加载检索器、不调用回答模型，只消费已有 items。
    - 逐题原子写回，中断时已判题目不丢失；判分失败不写 judge 字段，下次运行自动重试该题。
    - 共用批次锁 .evaluation.lock，避免与 evaluate.py 同时改写同一批次。
    - 异常信息只保留类型，不写原文，防止鉴权 URL 或密钥落盘。
"""
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from devtools.evaluation.exporter import export_results, judge_summary_markdown, load_items
from rag_agent.common.files import atomic_json, exclusive_lock, read_json
from rag_agent.common.progress import report, stage


# 三个判分维度；裁判模型必须按此键名返回 JSON。
SCORE_KEYS = ("correctness", "completeness", "relevance")


def build_judge_messages(item):
    """
    把一道题的完整记录组装成裁判请求消息。

    输入：item 是 items/*.json 中的一条逐题记录。
    输出：OpenAI chat 接口需要的 messages 列表（system + user）。

    组装内容：问题、参考答案、评分标准、RAG 回答、检索证据原文。
    裁判只依据这些材料打分，不使用其内部知识。
    """
    evidence = [{"rank": hit.get("metadata", {}).get("rank"),
                 "source": hit.get("metadata", {}).get("source"),
                 "page": hit.get("metadata", {}).get("page"),
                 "score": hit.get("metadata", {}).get("score"),
                 "text": hit.get("text", "")} for hit in item.get("hits", [])]
    payload = {
        "question": item.get("question", ""),
        "reference_answer": item.get("ground_truth", ""),
        "evaluation_criteria": item.get("eval_criteria", ""),
        "rag_answer": item.get("answer") or "",
        "retrieval_evidence": evidence,
    }
    system = (
        "你是严格、可复核的 RAG 回答裁判。请只依据给出的参考答案、评分标准和检索证据评分。"
        "分别给出三个 0 到 3 的整数分数：correctness 衡量事实是否正确且有证据支持；"
        "completeness 衡量是否覆盖参考答案与评分标准要求的要点；"
        "relevance 衡量是否直接回答问题且没有明显无关内容。"
        "0 表示完全不满足，1 表示少量满足，2 表示大部分满足，3 表示充分满足。"
        "只输出一个 JSON 对象，不要 Markdown、代码围栏或额外文字。"
        "如果 RAG 回答是空回答或表示未找到相关内容，直接给 correctness=0、completeness=0，reason 不超过一句话。"
        "reason 字段控制在 50 字以内，只写结论，不要展开推理过程。"
    )
    user = ("请返回格式："
            '{"correctness":0,"completeness":0,"relevance":0,"reason":"简明、具体、可复核的理由"}'
            "\n\n待评分记录：\n" + json.dumps(payload, ensure_ascii=False, indent=2))
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_judge_json(content):
    """
    严格解析裁判模型返回的 JSON 字符串。

    输入：裁判模型返回的文本内容。
    输出：(scores 字典, reason 字符串)，scores 含 correctness/completeness/relevance 三个 0–3 整数。

    容错：允许模型用 ```json ... ``` 代码围栏包裹。
    拒绝：缺字段、非整数、越界分数、空理由都抛 ValueError，由调用方记录为失败并重试。
    """
    text = (content or "").strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("裁判返回值必须是 JSON 对象")
    scores = {}
    for key in SCORE_KEYS:
        score = value.get(key)
        if type(score) is not int or not 0 <= score <= 3:
            raise ValueError(f"裁判字段 {key} 必须是 0–3 的整数")
        scores[key] = score
    reason = value.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("裁判必须返回非空 reason")
    return scores, reason.strip()


def _usage(response):
    """从裁判响应中提取 token 用量；模型未返回 usage 时返回 None。"""
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    return {key: getattr(usage, key, None) for key in ("prompt_tokens", "completion_tokens", "total_tokens")}


def _completed_judge(value):
    """判断 item 中是否已有有效判分：三个维度都是 0–3 整数且理由非空。"""
    if not isinstance(value, dict) or not isinstance(value.get("scores"), dict):
        return False
    return all(type(value["scores"].get(key)) is int and 0 <= value["scores"][key] <= 3
               for key in SCORE_KEYS) and isinstance(value.get("reason"), str) and bool(value["reason"].strip())


def _item_entries(directory):
    """
    读取批次目录下所有 item 文件，并按题号稳定排序返回（文件路径 + 记录）。

    排序与 exporter 一致，保证续跑时日志顺序可预期。
    """
    rows = load_items(directory)
    paths = {}
    for path in (Path(directory) / "items").glob("*.json"):
        row = read_json(path)
        paths[str(row.get("question_id"))] = path
    return [(paths[str(row["question_id"])], row) for row in rows]


def judge_directory(directory, client, model, base_url, temperature, max_tokens, limit=None, force=False):
    """
    逐题调用裁判模型打分并原子写回 item。

    输入：
        directory: 批次目录路径。
        client: 已初始化的 OpenAI 兼容客户端。
        model / base_url / temperature / max_tokens: 裁判模型调用参数。
        limit: 本次最多新判几题（试跑用）；force: 忽略已有判分全部重判。

    输出：汇总字典（本次完成/跳过/失败数 + summary）。

    流程：
        1. 加批次锁，遍历 items。
        2. 已有有效判分且非 force → 跳过。
        3. 调裁判模型 → 解析 JSON → 原子写回 item 的 judge 字段。
        4. 失败不写 judge，下次运行自动重试。
        5. finally 中刷新全部导出产物并打印汇总表。
    """
    directory = Path(directory)
    if limit is not None and limit < 1:
        raise ValueError("limit 必须大于零")
    endpoint_fingerprint = hashlib.sha256(base_url.encode("utf-8")).hexdigest()
    completed = skipped = failed = attempted = 0
    summary = None
    # 与 evaluate.py 共用批次锁，避免跑题和判分同时改写同一批次。
    with exclusive_lock(directory / ".evaluation.lock"):
        entries = _item_entries(directory)
        try:
            for number, (path, item) in enumerate(entries, 1):
                if _completed_judge(item.get("judge")) and not force:
                    skipped += 1
                    continue
                if limit is not None and attempted >= limit:
                    break
                attempted += 1
                qid = str(item["question_id"])
                report(f"[判分 {number}/{len(entries)}] 题目 {qid}：{item.get('question_type', '未分类')}")
                started = time.perf_counter()
                try:
                    with stage("等待裁判模型返回 JSON"):
                        response = client.chat.completions.create(
                            model=model,
                            messages=build_judge_messages(item),
                            temperature=temperature,
                            max_tokens=max_tokens,
                            extra_body={"thinking": {"type": "disabled"}},
                        )
                    content = response.choices[0].message.content
                    scores, reason = parse_judge_json(content)
                    updated = {**item, "judge": {
                        "schema": 1, "model": model, "endpoint_fingerprint": endpoint_fingerprint,
                        "temperature": temperature, "max_tokens": max_tokens,
                        "scores": scores, "reason": reason,
                        "seconds": time.perf_counter() - started, "usage": _usage(response),
                        "judged_at": datetime.now(timezone.utc).isoformat(),
                    }}
                    atomic_json(path, updated)
                    completed += 1
                except Exception as error:
                    # 云端异常可能含鉴权 URL；不写异常原文，也不写 judge，下一次运行会重试本题。
                    failed += 1
                    report(f"[判分失败] 题目 {qid}：{type(error).__name__}；未写入 judge，可稍后续跑")
        finally:
            # 判分结束后统一刷新 CSV、报告和汇总表，保证产物与最新 judge 字段一致。
            rows = load_items(directory)
            summary = export_results(directory, rows)
            print("\n" + judge_summary_markdown(rows), flush=True)
    report(f"[判分结束] 本次完成 {completed} 题，跳过 {skipped} 题，失败 {failed} 题")
    report(f"结果目录：{directory}")
    return {"attempted": attempted, "completed": completed, "skipped": skipped, "failed": failed, "summary": summary}
