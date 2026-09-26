"""evaluate.py —— 批量回答评测集，不调用评分模型；支持少量试跑和断点恢复。
PyCharm 直接运行默认读取 data/evaluation/testdata.json；默认执行全部尚未完成题目，会调用回答 LLM。
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from rag_agent.common.progress import configure_progress, report, stage


def code_snapshot(root, digest):
    """按逻辑模块名记录代码哈希，使纯目录迁移不改变历史批次身份。

    evaluation 两个文件使用迁移前的逻辑键，文件内容仍严格参与校验。
    不修改历史 run.json，也不放宽源码变化检查，保证旧批次可安全续跑。
    """
    root = Path(root)
    code = {str(path.relative_to(root)): digest(path)
            for path in sorted((root / "src/rag_agent").rglob("*.py"))}
    for path in sorted((root / "src/devtools/evaluation").rglob("*.py")):
        logical = Path("src/rag_agent/evaluation") / path.relative_to(root / "src/devtools/evaluation")
        code[str(logical)] = digest(path)
    return code


def main():
    """先校验数据和参数，再加载一次模型与索引，串行执行评测。"""
    configure_progress()
    parser = argparse.ArgumentParser(description="批量 RAG 评测（无评分模型）")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/evaluation/testdata.json")
    parser.add_argument("--source-map", type=Path, default=ROOT / "data/evaluation/testdata_sources.json")
    parser.add_argument("--output", type=Path, help="结果目录；指定已有目录时校验后恢复")
    parser.add_argument("--limit", type=int, help="本次最多执行多少题，适合少量试跑")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--retry-failed", action="store_true", help="重新执行已保存的错误题，可能再次产生费用")
    parser.add_argument("--validate-only", action="store_true", help="只检查数据与映射，不加载模型或调用 API")
    args = parser.parse_args()
    if (args.limit is not None and args.limit < 1) or (args.top_k is not None and args.top_k < 1):
        parser.error("limit 和 top-k 必须大于零")
    with stage("加载评测依赖并检查数据集"):
        from rag_agent import config
        from devtools.evaluation.runner import digest, load_dataset, run_evaluation
        mapping = json.loads(args.source_map.read_text(encoding="utf-8-sig"))
        rows = load_dataset(args.dataset, mapping)
    report(f"[评测集] {len(rows)} 题；不使用评分模型")
    if args.validate_only:
        report("[校验通过] 题目字段、题号及文档映射完整")
        return
    from rag_agent.qa.generator import Generator
    from rag_agent.qa.evidence_policy import EvidencePolicy
    from rag_agent.qa.reranker import Reranker
    from rag_agent.qa.retriever import Retriever
    with stage("加载当前混合索引与本地模型；整个批次只加载一次"):
        retriever = Retriever()
        reranker = Reranker() if config.RERANK_ENABLED else None
        evidence_policy = EvidencePolicy(config.RERANK_REJECT_THRESHOLD)
        generator = Generator()
    if reranker is not None:
        with stage("预加载并预热 BGE 重排模型；不计入逐题耗时"):
            reranker.prepare()
    k = args.top_k or config.TOP_K
    # 快照不记录 API Key 或带鉴权的 URL；源代码哈希包含提示词和评测实现。
    code = code_snapshot(ROOT, digest)
    model_dir = Path(config.EMBEDDING_MODEL)
    model_files = [(str(p.relative_to(model_dir)), p.stat().st_size, p.stat().st_mtime_ns)
                   for p in sorted(model_dir.rglob("*")) if p.is_file()]
    reranker_dir = Path(config.RERANKER_MODEL)
    reranker_files = [(str(p.relative_to(reranker_dir)), p.stat().st_size, p.stat().st_mtime_ns)
                      for p in sorted(reranker_dir.rglob("*")) if p.is_file()]
    bm25_manifest = retriever.store.index_path.parent / "bm25" / "manifest.json"
    signature = {"schema": 2, "dataset_sha256": digest(args.dataset), "mapping": mapping,
                 "index_sha256": digest(retriever.store.index_path), "metadata_sha256": digest(retriever.store.meta_path),
                 "index_version": retriever.store.index_path.parent.name,
                 "model": generator.model, "temperature": generator.temperature, "max_tokens": generator.max_tokens,
                 "enable_thinking": generator.enable_thinking if generator.thinking_supported else None,
                 "endpoint_fingerprint": hashlib.sha256(config.LLM_BASE_URL.encode()).hexdigest(),
                 "top_k": k, "max_context_chars": config.MAX_CONTEXT_CHARS,
                 "dense_top_k": config.DENSE_TOP_K, "bm25_top_k": config.BM25_TOP_K,
                 "fusion_top_k": config.FUSION_TOP_K, "rrf_k": config.RRF_K,
                 "reranker_enabled": config.RERANK_ENABLED,
                 "reranker_device": reranker.device if reranker is not None else None,
                 "reranker_batch_size": config.RERANK_BATCH_SIZE,
                 "reranker_max_length": config.RERANK_MAX_LENGTH,
                 "reranker_model": config.RERANKER_MODEL if config.RERANK_ENABLED else None,
                 "reranker_files_fingerprint": hashlib.sha256(
                     json.dumps(reranker_files).encode()).hexdigest() if config.RERANK_ENABLED else None,
                 "rerank_reject_threshold": config.RERANK_REJECT_THRESHOLD,
                 "bm25_manifest_sha256": digest(bm25_manifest),
                 "embedding_model": str(model_dir), "embedding_device": retriever.emb.device,
                 "embedding_max_length": config.EMBEDDING_MAX_LENGTH,
                 "embedding_files_fingerprint": hashlib.sha256(json.dumps(model_files).encode()).hexdigest(), "code": code}
    directory = args.output or ROOT / "data/outputs/evaluation" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_evaluation(rows, mapping, retriever, generator, directory, signature, k, args.limit,
                   args.retry_failed, reranker=reranker, evidence_policy=evidence_policy)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[中断] 已完成题目已保存；使用相同 --output 恢复。当前未保存请求可能已计费。", flush=True)
        sys.exit(130)
    except (ValueError, OSError, RuntimeError) as error:
        print(f"[评测未启动或未完成] {type(error).__name__}；检查数据、索引和输出目录配置。", flush=True)
        sys.exit(1)
