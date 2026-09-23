"""runner.py —— 逐题调用现有检索器和生成器，保存可复查的评测结果。
参考答案、评分标准与文档映射只用于报告，绝不传入回答提示词。
每题原子保存一次；恢复时校验配置快照，避免混合不同索引或模型的结果。
"""
import hashlib
import json
import time
from pathlib import Path

from devtools.evaluation.exporter import export_results
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
        report(f"[评测结束] 已记录 {summary['attempted']}/{len(rows)} 题，回答成功 {summary['answered']} 题")
        # 终端直接给出核心指标摘要，避免每次结束后还要翻 summary.json。
        # 指标可能为 None（题目未标注目标文档），用占位符显示而不是伪造零分。
        def pct(value):
            """把 0~1 小数格式化为百分号字符串；None 显示为占位符。"""
            return "—" if value is None else f"{value * 100:.1f}%"

        def sec(value):
            """秒数保留一位小数；None 显示为占位符。"""
            return "—" if value is None else f"{value:.1f}s"

        tokens = summary.get("reported_tokens") or {}
        report(f"[检索] hit@k {pct(summary.get('document_hit_at_k'))} | "
               f"recall@k {pct(summary.get('document_recall_at_k'))} | "
               f"MRR {pct(summary.get('document_mrr_at_k'))}")
        report(f"[延迟] 平均 {sec(summary.get('mean_total_seconds'))}，"
               f"P50 {sec(summary.get('p50_seconds'))}，P95 {sec(summary.get('p95_seconds'))}")
        report(f"[消耗] prompt {tokens.get('prompt_tokens', 0)} + "
               f"completion {tokens.get('completion_tokens', 0)} = "
               f"{tokens.get('total_tokens', 0)} tokens")
        report(f"结果目录：{directory}")
        return summary
