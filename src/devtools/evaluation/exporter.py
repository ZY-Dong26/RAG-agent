"""
exporter.py —— 从逐题 items 重新生成评测批次的全部汇总产物

职责：
    1. 读取批次目录下 items/*.json，生成 summary.json、results.jsonl、results.csv、report.md、judge_summary.md。
    2. 把 item 中的结构化 judge 字段映射为 CSV 的中文列，并在汇总表中计算各题型的判分均分。
    3. 供三个入口复用：evaluate.py 跑题结束后、judge.py 判分结束后、export_results.py 手动重导出。

设计原因：
    - 只读本地 JSON，不加载 Embedding、不调用回答模型或裁判模型，改导出格式后可对老批次零成本重建。
    - 三处共用同一套导出逻辑，避免 CSV 列名、Markdown 排版各自漂移。
    - CSV 列末尾预留四个判分列（正确性/完整性/相关性/理由），未判题时留空；人工评分列也保留。
"""
import csv
import json
import math
from collections import Counter
from pathlib import Path

from rag_agent.common.files import atomic_json, read_json


# CSV 末尾预留的四个判分列；judge.py 判完后填回。
JUDGE_COLUMNS = ["模型判分 - 正确性", "模型判分 - 完整性", "模型判分 - 相关性", "模型判分理由"]


def average(values):
    """忽略缺失数据；没有可统计样本时返回 None，而不是误报为零。"""
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def summarize(rows):
    """
    汇总一批逐题记录为指标字典。

    输入：同题型或同难度下的一组 item 记录。
    输出：含题数、回答成功率、检索 hit/recall/MRR、平均与 P50/P95 耗时、token 用量等指标。

    设计原因：检索成功但生成失败的题仍可参与检索指标统计；耗时样本用最近秩定义分位数。
    """
    metric_rows = [row for row in rows if row.get("metrics") is not None]
    durations = sorted(row["total_seconds"] for row in rows if row.get("total_seconds") is not None)

    def percentile(q):
        """采用最近秩定义，保证小样本时结果可解释。"""
        return durations[max(0, math.ceil(len(durations) * q) - 1)] if durations else None

    return {
        "attempted": len(rows), "answered": sum(row.get("status") == "answered" for row in rows),
        "success_rate": average([int(row.get("status") == "answered") for row in rows]),
        "status_counts": dict(Counter(row.get("status", "unknown") for row in rows)),
        "retrieval_evaluated": sum(row["metrics"].get("hit_at_k") is not None for row in metric_rows),
        "document_hit_at_k": average([row["metrics"].get("hit_at_k") for row in metric_rows]),
        "document_recall_at_k": average([row["metrics"].get("recall_at_k") for row in metric_rows]),
        "document_mrr_at_k": average([row["metrics"].get("reciprocal_rank_at_k") for row in metric_rows]),
        "mean_retrieval_seconds": average([row.get("retrieval_seconds") for row in rows]),
        "mean_generation_seconds": average([row.get("generation_seconds") for row in rows]),
        "mean_total_seconds": average(durations), "p50_seconds": percentile(.5), "p95_seconds": percentile(.95),
        "token_usage_reported_questions": sum(row.get("usage") is not None for row in rows),
        "reported_tokens": {key: sum((row.get("usage") or {}).get(key) or 0 for row in rows)
                            for key in ("prompt_tokens", "completion_tokens", "total_tokens")},
        "questions_missing_target_sources": sum(bool(row.get("missing_sources")) for row in rows),
    }


def _question_sort_key(row):
    """独立导出时按题号稳定排序；纯数字题号按数值排序，其余按文本排序。"""
    value = str(row.get("question_id", ""))
    return (0, int(value)) if value.isdigit() else (1, value.casefold())


def load_items(directory):
    """
    读取批次 items/*.json 并按题号稳定排序。

    输入：批次目录路径。
    输出：排序后的 item 记录列表。

    拒绝空目录、损坏记录和重复题号，避免导出静默丢数据。
    """
    directory = Path(directory)
    items = directory / "items"
    if not items.is_dir():
        raise ValueError(f"评测批次缺少 items 目录：{directory}")
    rows = []
    seen = set()
    for path in sorted(items.glob("*.json")):
        row = read_json(path)
        if not isinstance(row, dict) or row.get("question_id") is None:
            raise ValueError(f"逐题记录格式无效：{path.name}")
        qid = str(row["question_id"])
        if qid in seen:
            raise ValueError(f"逐题记录存在重复 question_id：{qid}")
        seen.add(qid)
        rows.append(row)
    if not rows:
        raise ValueError(f"items 中没有可导出的逐题记录：{directory}")
    return sorted(rows, key=_question_sort_key)


def _existing_total(directory, rows):
    """手动重导出优先沿用旧 summary 的数据集题数；没有旧汇总时退化为已保存题数。"""
    path = Path(directory) / "summary.json"
    if path.is_file():
        try:
            total = read_json(path).get("dataset_questions")
            if isinstance(total, int) and total >= len(rows):
                return total
        except (ValueError, OSError, json.JSONDecodeError):
            pass
    return len(rows)


def _judge_values(row):
    """把 item 中的结构化 judge 字段映射为 CSV 的四个中文列。"""
    judge = row.get("judge") if isinstance(row.get("judge"), dict) else {}
    scores = judge.get("scores") if isinstance(judge.get("scores"), dict) else {}
    return {
        JUDGE_COLUMNS[0]: scores.get("correctness", ""),
        JUDGE_COLUMNS[1]: scores.get("completeness", ""),
        JUDGE_COLUMNS[2]: scores.get("relevance", ""),
        JUDGE_COLUMNS[3]: judge.get("reason", ""),
    }


def _pct(value):
    return "" if value is None else f"{value * 100:.1f}%"


def _seconds(value):
    return "" if value is None else f"{value:.2f}s"


def _score_average(rows, key):
    values = []
    for row in rows:
        judge = row.get("judge") if isinstance(row.get("judge"), dict) else {}
        scores = judge.get("scores") if isinstance(judge.get("scores"), dict) else {}
        value = scores.get(key)
        if type(value) in (int, float):
            values.append(value)
    return average(values)


def judge_summary_markdown(rows):
    """
    生成按题型汇总的 Markdown 表格：检索指标 + 判分均分。

    输入：已判分或未判分的一组 item 记录。
    输出：Markdown 表格文本，首行"整体"，随后按题型分组。

    未判分时三个模型分数单元格留空；已判分时显示该题型均分。
    """
    groups = [("整体", rows)]
    groups.extend((name, [row for row in rows if row.get("question_type", "未分类") == name])
                  for name in sorted({row.get("question_type", "未分类") for row in rows}))
    lines = ["# RAG 评测与模型判分汇总", "",
             "检索指标按已有逐题记录统计；模型判分范围为 0–3，未判题不参与均值。", "",
             "| 题型 | 题数 | Hit@K | Recall@K | MRR@K | 检索耗时 | 生成耗时 | 总耗时 | 正确性 | 完整性 | 相关性 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, group in groups:
        summary = summarize(group)
        scores = [_score_average(group, key) for key in ("correctness", "completeness", "relevance")]
        formatted = ["" if value is None else f"{value:.2f}" for value in scores]
        lines.append(f"| {name} | {len(group)} | {_pct(summary['document_hit_at_k'])} | "
                     f"{_pct(summary['document_recall_at_k'])} | {_pct(summary['document_mrr_at_k'])} | "
                     f"{_seconds(summary['mean_retrieval_seconds'])} | "
                     f"{_seconds(summary['mean_generation_seconds'])} | "
                     f"{_seconds(summary['mean_total_seconds'])} | " + " | ".join(formatted) + " |")
    return "\n".join(lines) + "\n"


def export_results(directory, rows=None, total=None):
    """
    从内存记录或 items 生成全部汇总产物，全程纯本地。

    输入：
        directory: 批次目录路径。
        rows: 可选，内存中的 item 记录列表；为 None 时从 items/*.json 重新读取。
        total: 可选，数据集总题数；为 None 时沿用旧 summary.json 或退化为已保存题数。

    输出：summary 指标字典。

    产物：summary.json、results.jsonl、results.csv、report.md、judge_summary.md。
    CSV 写入前对以 = + - @ 开头的文本加单引号前缀，防止 Excel 误判为公式。
    """
    directory = Path(directory)
    rows = list(rows) if rows is not None else load_items(directory)
    total = _existing_total(directory, rows) if total is None else total
    if not isinstance(total, int) or total < len(rows):
        raise ValueError("数据集题数不能小于已保存的逐题记录数")

    summary = summarize(rows)
    summary.update(dataset_questions=total, remaining=total-len(rows),
                   metric_scope="文档级；K 是检索片段数，不代表正确段落命中；答案评分见 judge 字段",
                   latency_scope="不含初始化时间；包含本题接口等待，P50/P95 使用最近秩",
                   usage_scope="仅统计回答接口实际返回的用量；裁判用量保存在各 item 的 judge 字段")
    for field in ("question_type", "difficulty"):
        values = sorted({row.get(field, "未分类") for row in rows})
        summary["by_" + field] = {value: summarize([row for row in rows if row.get(field, "未分类") == value])
                                  for value in values}
    atomic_json(directory / "summary.json", summary)
    with (directory / "results.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    columns = ["question_id", "question_type", "difficulty", "question", "ground_truth", "answer",
               "eval_criteria", "status", "retrieved_sources", "missing_sources", "hit_at_k", "recall_at_k",
               "reciprocal_rank_at_k", "retrieval_seconds", "generation_seconds", "total_seconds",
               "prompt_tokens", "completion_tokens", "total_tokens", "error", "人工评分", "人工备注",
               *JUDGE_COLUMNS]
    with (directory / "results.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            item = {**row, **(row.get("metrics") or {}), **(row.get("usage") or {}), **_judge_values(row)}
            item["retrieved_sources"] = "；".join(
                f"{hit.get('metadata', {}).get('source')} 第{hit.get('metadata', {}).get('page')}页"
                for hit in row.get("hits", []))
            item["missing_sources"] = "；".join(row.get("missing_sources", []))
            output = {key: item.get(key, "") for key in columns}
            for key, value in output.items():
                if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
                    output[key] = "'" + value
            writer.writerow(output)

    lines = ["# RAG 批量评测结果", "",
             "模型判分范围为 0–3，仅供辅助分析；未判题显示为“尚未判分”。", ""]
    for row in rows:
        judge = row.get("judge") if isinstance(row.get("judge"), dict) else None
        if judge:
            scores = judge.get("scores") or {}
            judge_lines = [f"**模型判分：** 正确性 {scores.get('correctness', '—')}/3；"
                           f"完整性 {scores.get('completeness', '—')}/3；"
                           f"相关性 {scores.get('relevance', '—')}/3", "",
                           f"**判分理由：** {judge.get('reason', '（无）')}", ""]
        else:
            judge_lines = ["**模型判分：** 尚未判分", ""]
        lines += [f"## 题目 {row['question_id']} · {row.get('question_type', '未分类')} · {row.get('difficulty', '未分类')}", "",
                  f"**问题：** {row.get('question', '')}", "", f"**参考答案：** {row.get('ground_truth', '')}", "",
                  f"**RAG 回答：** {row.get('answer') or '（无回答）'}", "",
                  f"**评分标准：** {row.get('eval_criteria', '')}", "", *judge_lines,
                  f"状态：{row.get('status')}；耗时：{(row.get('total_seconds') or 0):.2f} 秒；"
                  f"缺失目标文档：{row.get('missing_sources', [])}", ""]
        for hit in row.get("hits", []):
            meta = hit.get("metadata", {})
            lines += [f"### 检索 [{meta.get('rank')}] {meta.get('source')} 第 {meta.get('page')} 页，"
                      f"相似度 {meta.get('score')}", "", hit.get("text", ""), ""]
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (directory / "judge_summary.md").write_text(judge_summary_markdown(rows), encoding="utf-8")
    return summary
