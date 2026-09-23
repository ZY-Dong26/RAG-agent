"""runner.py —— 逐题调用现有检索器和生成器，保存可复查的评测结果。
参考答案、评分标准与文档映射只用于报告，绝不传入回答提示词。
每题原子保存一次；恢复时校验配置快照，避免混合不同索引或模型的结果。
"""
import csv
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

from rag_agent.common.files import atomic_json, read_json, exclusive_lock
from rag_agent.common.progress import report, stage


def digest(path):
    """分块计算文件指纹，记录数据集、索引和实现版本。"""
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_dataset(path, source_map):
    """严格检查题号、问题及目标文档；不完整映射直接报错，避免误计为零分。"""
    rows = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("评测集必须是非空 JSON 数组")
    if not isinstance(source_map, dict) or any(not isinstance(v, str) or not v.strip() for v in source_map.values()):
        raise ValueError("文档映射必须是编号到非空文件名的字典")
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("每道题必须是 JSON 对象")
        qid = row.get("question_id")
        if type(qid) not in (int, str) or not str(qid).strip() or str(qid) in seen:
            raise ValueError("question_id 必须是唯一的非空字符串或整数")
        seen.add(str(qid))
        for field in ("question", "ground_truth", "question_type", "difficulty", "eval_criteria"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"题目 {qid} 缺少有效字段 {field}")
        refs = row.get("reference_context_ids")
        if not isinstance(refs, list) or any(not isinstance(ref, str) or ref not in source_map for ref in refs):
            raise ValueError(f"题目 {qid} 的 reference_context_ids 缺少文档映射")
    return rows


def retrieval_metrics(hits, targets, k):
    """在前 K 个片段中计算文档级指标；同一文档重复出现不重复计入召回。
    MRR 的位置按片段排名计数。没有目标标注时返回 None，不伪造零分。
    """
    targets = set(targets)
    if not targets:
        return {"hit_at_k": None, "recall_at_k": None, "reciprocal_rank_at_k": None}
    sources = [hit["metadata"].get("source") for hit in hits[:k]]
    rank = next((i for i, source in enumerate(sources, 1) if source in targets), None)
    return {"hit_at_k": int(rank is not None),
            "recall_at_k": len(targets.intersection(sources)) / len(targets),
            "reciprocal_rank_at_k": 1 / rank if rank else 0.0}


def average(values):
    """忽略缺失数据；没有可统计样本时返回 None，而不是误报为零。"""
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def summarize(rows):
    """汇总完成记录；检索成功但生成失败的题仍可参与检索指标。"""
    metric_rows = [row for row in rows if row.get("metrics") is not None]
    durations = sorted(row["total_seconds"] for row in rows)
    def percentile(q):
        """采用最近秩定义，保证小样本时结果可解释。"""
        return durations[max(0, math.ceil(len(durations) * q) - 1)] if durations else None
    return {
        "attempted": len(rows), "answered": sum(row["status"] == "answered" for row in rows),
        "success_rate": average([int(row["status"] == "answered") for row in rows]),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "retrieval_evaluated": sum(row["metrics"]["hit_at_k"] is not None for row in metric_rows),
        "document_hit_at_k": average([row["metrics"]["hit_at_k"] for row in metric_rows]),
        "document_recall_at_k": average([row["metrics"]["recall_at_k"] for row in metric_rows]),
        "document_mrr_at_k": average([row["metrics"]["reciprocal_rank_at_k"] for row in metric_rows]),
        "mean_retrieval_seconds": average([row.get("retrieval_seconds") for row in rows]),
        "mean_generation_seconds": average([row.get("generation_seconds") for row in rows]),
        "mean_total_seconds": average(durations), "p50_seconds": percentile(.5), "p95_seconds": percentile(.95),
        "token_usage_reported_questions": sum(row.get("usage") is not None for row in rows),
        "reported_tokens": {key: sum((row.get("usage") or {}).get(key) or 0 for row in rows)
                            for key in ("prompt_tokens", "completion_tokens", "total_tokens")},
        "questions_missing_target_sources": sum(bool(row.get("missing_sources")) for row in rows),
    }


def export_results(directory, rows, total):
    """导出机器记录、Excel 可读 CSV 和人工逐题阅读的 Markdown。"""
    directory = Path(directory)
    summary = summarize(rows)
    summary.update(dataset_questions=total, remaining=total-len(rows),
                   metric_scope="文档级；K 是检索片段数，不代表正确段落命中；没有自动答案评分",
                   latency_scope="不含初始化时间；包含本题接口等待，P50/P95 使用最近秩",
                   usage_scope="仅统计接口实际返回的用量；缺失用量及中断请求可能未计入")
    for field in ("question_type", "difficulty"):
        summary["by_" + field] = {value: summarize([row for row in rows if row[field] == value])
                                  for value in sorted({row[field] for row in rows})}
    atomic_json(directory / "summary.json", summary)
    with (directory / "results.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    columns = ["question_id", "question_type", "difficulty", "question", "ground_truth", "answer",
               "eval_criteria", "status", "retrieved_sources", "missing_sources", "hit_at_k", "recall_at_k",
               "reciprocal_rank_at_k", "retrieval_seconds", "generation_seconds", "total_seconds",
               "prompt_tokens", "completion_tokens", "total_tokens", "error", "人工评分", "人工备注"]
    with (directory / "results.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            item = {**row, **(row.get("metrics") or {}), **(row.get("usage") or {})}
            item["retrieved_sources"] = "；".join(f"{h['metadata'].get('source')} 第{h['metadata'].get('page')}页" for h in row["hits"])
            item["missing_sources"] = "；".join(row["missing_sources"])
            # 避免资料文本在 Excel 中被解释为公式；原始 JSON 保留原文。
            output = {key: item.get(key, "") for key in columns}
            for key, value in output.items():
                if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
                    output[key] = "'" + value
            writer.writerow(output)
    lines = ["# RAG 批量评测结果", "", "此报告不含自动答案评分；请结合参考答案、评分标准和检索证据人工评阅。", ""]
    for row in rows:
        lines += [f"## 题目 {row['question_id']} · {row['question_type']} · {row['difficulty']}", "",
                  f"**问题：** {row['question']}", "", f"**参考答案：** {row['ground_truth']}", "",
                  f"**RAG 回答：** {row.get('answer') or '（无回答）'}", "", f"**评分标准：** {row['eval_criteria']}", "",
                  f"状态：{row['status']}；耗时：{row['total_seconds']:.2f} 秒；缺失目标文档：{row['missing_sources']}", ""]
        for hit in row["hits"]:
            meta = hit["metadata"]
            lines += [f"### 检索 [{meta.get('rank')}] {meta.get('source')} 第 {meta.get('page')} 页，相似度 {meta.get('score')}", "", hit["text"], ""]
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def run_evaluation(rows, source_map, retriever, generator, directory, signature, k=5, limit=None, retry_failed=False):
    """串行执行并逐题保存；只有 question 和检索 hits 会交给生成器。
    limit 限制本次实际执行题数；恢复自动跳过成功题，失败题需显式 retry_failed。
    中断时导出已完成题目；当前尚未保存的 API 请求可能已计费，恢复时会重做该题。
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(directory / ".evaluation.lock"):
        manifest = directory / "run.json"
        if manifest.exists():
            if read_json(manifest) != signature:
                raise ValueError("评测集、索引、代码或参数已变化，请使用新的输出目录")
        else:
            if (directory / "items").exists():
                raise ValueError("存在逐题结果但没有运行清单，拒绝混合结果")
            atomic_json(manifest, signature)
        items = directory / "items"
        items.mkdir(exist_ok=True)
        results = {}
        for row in rows:
            key = hashlib.sha256(str(row["question_id"]).encode()).hexdigest()
            path = items / (key + ".json")
            if path.exists():
                results[str(row["question_id"])] = read_json(path)
        available = {chunk["metadata"].get("source") for chunk in retriever.store.chunks}
        count = 0
        try:
            for number, row in enumerate(rows, 1):
                qid = str(row["question_id"])
                old = results.get(qid)
                if old and not (retry_failed and old["status"] == "error"):
                    continue
                if limit is not None and count >= limit:
                    break
                count += 1
                targets = sorted({source_map[ref] for ref in row["reference_context_ids"]})
                result = {**row, "targets": targets, "missing_sources": sorted(set(targets)-available),
                          "hits": [], "answer": None, "usage": None, "metrics": None,
                          "retrieval_seconds": None, "generation_seconds": None, "error": None}
                report(f"[评测 {number}/{len(rows)}] 题目 {qid}：{row['question_type']}")
                started = time.perf_counter()
                phase = "retrieval"
                try:
                    with stage("检索评测资料"):
                        step = time.perf_counter()
                        result["hits"] = retriever.retrieve(row["question"], top_k=k)
                        result["retrieval_seconds"] = time.perf_counter()-step
                    result["metrics"] = retrieval_metrics(result["hits"], targets, k)
                    if result["hits"]:
                        phase = "generation"
                        # 保存实际发送的提示词，明确哪些证据因字符预算被截掉。
                        result["messages"] = generator.build_prompt(row["question"], result["hits"])
                        with stage("生成评测回答"):
                            step = time.perf_counter()
                            result["answer"] = generator.generate(row["question"], result["hits"])
                            result["generation_seconds"] = time.perf_counter()-step
                        result["usage"] = getattr(generator, "last_usage", None)
                        result["status"] = "answered" if result["answer"] else "empty_answer"
                    else:
                        result["status"] = "no_hits"
                except Exception as error:
                    # 不保存异常原文：HTTP 异常可能包含鉴权地址或密钥。
                    result["status"] = "error"
                    result["error"] = f"{phase}: {type(error).__name__}；请检查配置、连接及服务状态"
                    result[phase + "_seconds"] = time.perf_counter()-step
                    report(f"[失败] 题目 {qid}，阶段 {phase}；继续下一题")
                result["total_seconds"] = time.perf_counter()-started
                key = hashlib.sha256(qid.encode()).hexdigest()
                atomic_json(items / (key + ".json"), result)
                results[qid] = result
        finally:
            ordered = [results[str(row["question_id"])] for row in rows if str(row["question_id"]) in results]
            summary = export_results(directory, ordered, len(rows))
        report(f"[评测结束] 已记录 {summary['attempted']}/{len(rows)} 题，回答成功 {summary['answered']} 题；结果：{directory}")
        return summary
