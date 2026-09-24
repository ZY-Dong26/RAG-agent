"""
builder.py —— 按文档增量更新知识库的业务层

职责：
    1. 比较 data/raw 中的 PDF 与当前活动索引，判断复用、新增、更新和缺失。
    2. 单份解析失败时保留其他成功结果；解析更新失败时尝试使用该文档的旧版向量。
    3. 只给新增或变化文档计算 Embedding，未变化文档复用 document_artifacts。
    4. 把所有有效文档重新组装成同代 FAISS 与 BM25 索引，验证后原子发布。

这里的“增量”指昂贵的解析、切块和向量计算按文档执行；活动 FAISS 不做原地追加。
完整候选索引很容易验证和回滚，也能正确处理文档替换、删除和更新失败。
"""
import time
import uuid
from pathlib import Path

import numpy as np
from rag_agent.common.progress import report as notify, stage

from rag_agent import config
from rag_agent.indexing import chunker
from rag_agent.indexing.artifact_store import (
    artifact_key, document_identity, file_hash, fingerprint, load_artifact, save_artifact,
)
from rag_agent.ingestion.mineru_adapter import ADAPTER_VERSION
from rag_agent.ingestion.mineru_client import MinerUCloud, parse_files
from rag_agent.ingestion.postprocessor import POSTPROCESSOR_VERSION, postprocess_document
from rag_agent.ingestion.mineru_settings import load_settings
from rag_agent.common.files import atomic_json, exclusive_lock, read_json


SPLITTER_VERSION = 3  # v3 表示切块元数据加入稳定的文档 ID 和内容型 chunk ID。
MANIFEST_VERSION = 3  # v3 表示每代索引同时包含已校验的 FAISS 与 BM25。


def input_paths(paths=None):
    """返回稳定排序后的 PDF 列表；建库当前只扫描 data/raw 的直接子文件。"""
    result = list(paths) if paths is not None else sorted(config.RAW_PDF_DIR.glob("*.pdf"))
    result = [Path(path) for path in result]
    if not result:
        raise RuntimeError("data/raw 下没有 PDF，保留现有索引")
    return result


def collect_parse_results(paths, resubmit=False, settings=None):
    """
    批量调用 MinerU，并把原先的扁平 documents 按来源文件重新分组。

    parse_files 会逐份捕获失败，因此一份 PDF 失败不会丢掉其他成功结果。这里不决定是否发布，
    默认部分成功和 --strict 的差异由 build_index 在拥有旧索引状态后统一判断。
    """
    settings = settings or load_settings(config.BASE_DIR)
    parser = MinerUCloud(settings, config.PROCESSED_DIR / "mineru")
    try:
        documents, report = parse_files(paths, parser, resubmit=resubmit)
    finally:
        parser.close()
    grouped = {Path(path).name: [] for path in paths}
    for document in documents:
        source = document.get("metadata", {}).get("source")
        if source in grouped:
            grouped[source].append(document)
    return grouped, report


def parse_documents(paths=None, resubmit=False):
    """
    只执行解析入口使用的严格流程，不切块、不向量化。

    单独运行 parse_documents.py 的目标是检查指定文档是否全部解析成功，所以仍采用严格结果；
    build_index 的默认部分成功逻辑不会改变这个调试入口的退出语义。
    """
    paths = input_paths(paths)
    grouped, report = collect_parse_results(paths, resubmit=resubmit)
    atomic_json(config.PROCESSED_DIR / "parse_report.json", report)
    failed = [row["source"] for row in report if row.get("status") != "done"]
    if failed:
        raise RuntimeError("以下文档解析失败：" + "、".join(failed)
                           + "；详情见 data/processed/parse_report.json")
    documents, audits = [], []
    for path in paths:
        refined, audit = postprocess_document(grouped.get(path.name, []), path.name)
        documents.extend(refined)
        audits.append({"source": path.name, **audit})
    atomic_json(config.PROCESSED_DIR / "documents.json", documents)
    atomic_json(config.PROCESSED_DIR / "postprocess_report.json", {
        "postprocessor_version": POSTPROCESSOR_VERSION, "documents": audits,
    })
    return documents


def embedding_signature():
    """
    计算影响向量结果的本地模型签名。

    不逐字节哈希大模型权重，使用相对路径、大小和修改时间识别本地模型变化；这与原建库逻辑
    保持一致。最大输入长度也参与签名，避免不同截断设置生成的向量被混用。
    """
    model_dir = Path(config.EMBEDDING_MODEL)
    files = sorted((str(path.relative_to(model_dir)), path.stat().st_size, path.stat().st_mtime_ns)
                   for path in model_dir.rglob("*") if path.is_file()) if model_dir.is_dir() else []
    return fingerprint({"model": str(model_dir), "files": files,
                        "max_length": config.EMBEDDING_MAX_LENGTH, "normalized": True})


def pipeline_signature(settings, embedding_id):
    """组合解析、适配、切块和 Embedding 配置；任一变化都会使旧文档向量失效。"""
    return fingerprint({
        "mineru_parameters": settings.parameters(),
        "adapter_version": ADAPTER_VERSION,
        "postprocessor_version": POSTPROCESSOR_VERSION,
        "splitter_version": SPLITTER_VERSION,
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "embedding_signature": embedding_id,
    })


def _active_manifest():
    """读取当前活动索引清单；旧格式索引没有逐文档记录时返回空清单并走一次迁移重建。"""
    from rag_agent.indexing import faiss_store
    try:
        store = faiss_store.VectorStore().load()
    except (FileNotFoundError, ValueError, RuntimeError):
        return {}
    path = store.index_path.parent / "build_manifest.json"
    return read_json(path) if path.is_file() else {}


def _active_hybrid_index_valid(expected_chunks):
    """
    验证当前代 FAISS、metadata 与 BM25 仍能共同回读，且 chunk_id 顺序等于本次候选。

    构建签名相同并不代表磁盘文件一定完好；若 BM25 缺失或损坏，应在本次仅用已有
    chunks/向量重新发布，而不是因为清单签名相同就继续保留不可用的活动版本。
    """
    from rag_agent.indexing import faiss_store
    from rag_agent.indexing.bm25_store import BM25Store
    try:
        store = faiss_store.VectorStore().load()
        BM25Store.load(store.index_path.parent / "bm25", store.chunks)
    except (FileNotFoundError, ValueError, RuntimeError, OSError):
        return False
    active_ids = [chunk.get("metadata", {}).get("chunk_id") for chunk in store.chunks]
    expected_ids = [chunk.get("metadata", {}).get("chunk_id") for chunk in expected_chunks]
    return active_ids == expected_ids


def _stable_chunks(documents, document_id, source_sha256, source_path):
    """
    给一份文档切块并补充稳定元数据。

    TextSplitter 仍负责正文/表格的实际边界；这里把其局部编号转换为内容型哈希。因为每份文档
    单独调用切分器，其他 PDF 的增删不会改变本文件 chunk 标识。
    """
    enriched = []
    for document in documents:
        enriched.append({"text": document["text"], "metadata": {
            **document["metadata"], "document_id": document_id,
            "document_sha256": source_sha256, "source_path": source_path,
        }})
    chunks = chunker.TextSplitter(config.CHUNK_SIZE, config.CHUNK_OVERLAP).split_documents(enriched)
    for local_index, chunk in enumerate(chunks):
        meta = chunk["metadata"]
        meta["segment_index"] = local_index
        meta["chunk_id"] = fingerprint({"document_id": document_id,
                                         "block_id": meta.get("block_id"),
                                         "segment_index": local_index, "text": chunk["text"]})
    return enriched, chunks


def _valid_old_record(record, source_sha256, pipeline_id):
    """只有源文件、整条处理流水线和产物校验都一致时，旧向量才可以直接复用。"""
    if (record.get("active_source_sha256") != source_sha256
            or record.get("pipeline_signature") != pipeline_id):
        return False
    try:
        load_artifact(config.VECTOR_DB_DIR, record.get("artifact_key"))
        return True
    except (FileNotFoundError, ValueError, OSError):
        return False


def _record_from_artifact(identity, source, relative, source_sha256, pipeline_id,
                          artifact, stats=None, status="active", **extra):
    """把已验收的单文档产物转换为活动索引清单中的一条记录。"""
    manifest = artifact["manifest"]
    return {
        "document_id": identity,
        "source": source,
        "source_path": relative,
        "active_source_sha256": source_sha256,
        "observed_source_sha256": source_sha256,
        "pipeline_signature": pipeline_id,
        "artifact_key": manifest["artifact_key"],
        "status": status,
        "pages": (stats or {}).get("parsed_pages", manifest.get("pages")),
        "chunks": manifest["chunks"],
        **extra,
    }


def build_index(force=False, strict=False, prune_missing=False):
    """
    按文档增量构建并发布知识库。

    :param force: 所有当前可解析 PDF 都重新切块和计算向量，但仍复用 MinerU 解析缓存。
    :param strict: 任一当前 PDF 没有成功处理到最新版本时拒绝发布。
    :param prune_missing: 从候选索引移除已不在 data/raw 的旧文档；默认保留并标记来源缺失。
    :return: 当前发布或复用的完整 chunk 列表。
    """
    with exclusive_lock(config.VECTOR_DB_DIR / ".build.lock"):
        paths = input_paths()
        notify(f"[扫描] 发现 {len(paths)} 份 PDF，正在核对已有产物")
        settings = load_settings(config.BASE_DIR)
        embedding_id = embedding_signature()
        pipeline_id = pipeline_signature(settings, embedding_id)
        old_manifest = _active_manifest()
        old_records = {row["document_id"]: row for row in old_manifest.get("document_records", [])
                       if isinstance(row, dict) and row.get("document_id")}

        # 第一遍只做便宜的本地比较，找出确实需要解析的文档。未变化且产物完整的文档不会联网。
        current = {}
        parse_paths = []
        report = []
        for path in paths:
            identity, relative = document_identity(path, config.RAW_PDF_DIR)
            source_sha256 = file_hash(path)
            old = old_records.get(identity)
            current[identity] = {"path": path, "relative": relative, "sha256": source_sha256, "old": old}
            if not force and old and _valid_old_record(old, source_sha256, pipeline_id):
                current[identity]["action"] = "reuse"
                report.append({"source": path.name, "status": "done", "stage": "vector_reused"})
            else:
                current[identity]["action"] = "process"
                parse_paths.append(path)

        notify(f"[计划] 复用向量 {len(current) - len(parse_paths)} 份，需处理 {len(parse_paths)} 份")
        grouped, parse_report = ({}, [])
        if parse_paths:
            grouped, parse_report = collect_parse_results(parse_paths, settings=settings)
            report.extend(parse_report)
        atomic_json(config.PROCESSED_DIR / "parse_report.json", report)
        parse_rows = {row["source"]: row for row in parse_report}

        # records 保存可进入候选索引的文档；unavailable 记录失败或使用旧版的文档。
        # pending 是解析成功、尚需准备向量产物的队列；failed_current 用于严格模式判断。
        records = []
        unavailable = []
        pending = []
        failed_current = []

        for identity, item in current.items():
            path, old = item["path"], item["old"]
            if item["action"] == "reuse":
                records.append({**old, "status": "active", "observed_source_sha256": item["sha256"],
                                "error": None})
                continue

            row = parse_rows.get(path.name, {"source": path.name, "status": "failed",
                                             "stage": "local_failed", "error": "解析结果缺失"})
            documents = grouped.get(path.name, [])
            if row.get("status") == "done" and documents:
                pending.append((identity, item, documents, row))
                continue

            failed_current.append(path.name)
            failure = {"document_id": identity, "source": path.name, "source_path": item["relative"],
                       "observed_source_sha256": item["sha256"], "status": "failed_new",
                       "stage": row.get("stage", "local_failed"), "error": row.get("error", "解析失败")}
            if old:
                # 新版失败时继续引用旧产物，同时明确记录“当前原文件与活动内容不一致”。
                try:
                    load_artifact(config.VECTOR_DB_DIR, old.get("artifact_key"))
                    records.append({**old, "status": "stale",
                                    "observed_source_sha256": item["sha256"],
                                    "error": failure["error"], "failed_stage": failure["stage"]})
                    failure["status"] = "stale"
                    failure["active_source_sha256"] = old.get("active_source_sha256")
                except (FileNotFoundError, ValueError, OSError):
                    failure["error"] += "；旧文档向量产物也不可用"
            unavailable.append(failure)

        # data/raw 中暂时找不到的旧文档默认保留，防止文件移动或同步延迟导致知识被误删。
        for identity, old in old_records.items():
            if identity in current or prune_missing:
                continue
            try:
                load_artifact(config.VECTOR_DB_DIR, old.get("artifact_key"))
                records.append({**old, "status": "missing_source", "observed_source_sha256": None,
                                "error": "原始 PDF 当前不在 data/raw；仍保留上次成功版本"})
            except (FileNotFoundError, ValueError, OSError):
                unavailable.append({"document_id": identity, "source": old.get("source"),
                                    "status": "missing_source", "error": "原始 PDF 和旧向量产物均不可用"})

        # 若有解析失败，严格模式在任何新向量计算和候选发布之前停止；成功解析缓存仍会保留供下次复用。
        if strict and failed_current:
            atomic_json(config.PROCESSED_DIR / "build_report.json", {
                "status": "strict_failed", "strict": True, "failed": unavailable,
                "message": "严格模式要求 data/raw 中每份 PDF 都处理到最新版本，活动索引未改变",
            })
            raise RuntimeError("严格模式下存在解析失败文档，未更新索引：" + "、".join(failed_current))

        # 第二遍只处理成功解析且不能复用活动产物的文档。Embedding 模型在第一次需要时才加载。
        # 这里没有逐文档捕获向量计算/产物写入异常；这些异常会终止本次构建，旧索引不变。
        embedder = None
        for identity, item, documents, stats in pending:
            refined_documents, postprocess_report = postprocess_document(documents, item["path"].name)
            notify(f"[切块] {item['path'].name}")
            normalized_docs, chunks = _stable_chunks(
                refined_documents, identity, item["sha256"], item["relative"])
            if not chunks:
                raise RuntimeError(f"{item['path'].name} 没有有效文本块，保留现有索引")
            key = artifact_key(identity, item["sha256"], pipeline_id,
                               nonce=uuid.uuid4().hex if force else None)
            try:
                artifact = load_artifact(config.VECTOR_DB_DIR, key) if not force else None
            except (FileNotFoundError, ValueError, OSError):
                artifact = None
            if artifact is None:
                if embedder is None:
                    with stage("加载本地 Embedding 模型"):
                        from rag_agent.indexing import embedder
                        embedder = embedder.Embedding()
                with stage(f"向量化 {item['path'].name}：{len(chunks)} 个文本块"):
                    vectors = embedder.embed_texts([chunk["text"] for chunk in chunks])
                artifact = save_artifact(config.VECTOR_DB_DIR, key, normalized_docs, chunks, vectors, {
                    "document_id": identity, "source": item["path"].name,
                    "source_path": item["relative"], "source_sha256": item["sha256"],
                    "pipeline_signature": pipeline_id, "embedding_signature": embedding_id,
                    "postprocessor_version": POSTPROCESSOR_VERSION,
                    "postprocess_fail_open": postprocess_report["fail_open"],
                    "pages": stats.get("parsed_pages"), "created_at": time.time(),
                }, postprocess_report=postprocess_report)
            records.append(_record_from_artifact(identity, item["path"].name, item["relative"],
                                                  item["sha256"], pipeline_id, artifact, stats))

        if not records:
            atomic_json(config.PROCESSED_DIR / "build_report.json", {
                "status": "failed", "strict": strict, "failed": unavailable,
                "message": "没有任何可发布的有效文档，活动索引未改变",
            })
            raise RuntimeError("没有任何成功或可复用的文档，保留现有索引")

        # 固定文档顺序，使相同输入总能产生相同 index_signature，并检查跨文档 chunk 不重复。
        records.sort(key=lambda row: (str(row.get("source_path", "")).casefold(), row["document_id"]))
        all_documents, all_chunks, matrices = [], [], []
        seen_chunks = set()
        for record in records:
            artifact = load_artifact(config.VECTOR_DB_DIR, record["artifact_key"])
            all_documents.extend(artifact["documents"])
            for chunk in artifact["chunks"]:
                chunk_id = chunk.get("metadata", {}).get("chunk_id")
                if not chunk_id or chunk_id in seen_chunks:
                    raise ValueError("候选索引包含缺失或重复的 chunk_id，拒绝发布")
                seen_chunks.add(chunk_id)
                all_chunks.append(chunk)
            matrices.append(artifact["vectors"])
        # 此处只检查维度一致，不能据此证明不同模型生成的向量语义兼容；更换模型需谨慎。
        dimensions = {matrix.shape[1] for matrix in matrices}
        if len(dimensions) != 1:
            raise ValueError("文档向量维度不一致，可能混用了不同 Embedding 模型")
        vectors = np.vstack(matrices)

        unavailable.sort(key=lambda row: str(row.get("source", "")).casefold())
        publication_state = [{key: row.get(key) for key in
                              ("document_id", "artifact_key", "status", "observed_source_sha256")}
                             for row in records]
        # 延迟导入 bm25s 相关模块，避免只运行解析或查看 --help 时加载稀疏检索依赖。
        from rag_agent.indexing.bm25_store import BM25Store
        bm25_signature = BM25Store.configuration_signature()
        index_signature = fingerprint({
            "records": publication_state,
            "unavailable": unavailable,
            "bm25_configuration": bm25_signature,
        })
        # BM25 配置只参与整库发布签名，不进入逐文档 pipeline_signature。因此 tokenizer 或
        # bm25s 参数变化时会用已有 chunks 重建本地索引，不会让解析缓存和文档向量失效。
        already_current = (old_manifest.get("signature") == index_signature
                           and _active_hybrid_index_valid(all_chunks))

        notify(f"[索引] 汇总 {len(records)} 份有效文档，{len(all_chunks)} 个向量")
        if not already_current:
            notify("[发布] 构建 FAISS 与中文 BM25 候选索引，回读验证后切换版本")
            from rag_agent.indexing import faiss_store
            candidate = faiss_store.VectorStore()
            # VectorStore.add 同时支持普通列表；转回列表避免 NumPy 数组参与布尔判断产生歧义。
            candidate.add(all_chunks, vectors.tolist())
            with stage(f"构建中文 BM25：{len(all_chunks)} 个文本块"):
                bm25_candidate = BM25Store().build(all_chunks)
            candidate.save({
                "manifest_version": MANIFEST_VERSION,
                "signature": index_signature,
                "pipeline_signature": pipeline_id,
                "embedding_signature": embedding_id,
                "bm25_configuration_signature": bm25_signature,
                "documents": [row["source"] for row in records],
                "document_records": records,
                "unavailable_documents": unavailable,
                "chunks": len(all_chunks),
                "embedding_model": str(Path(config.EMBEDDING_MODEL)),
                "partial": bool(unavailable),
                "created_at": time.time(),
            }, bm25_store=bm25_candidate)
        else:
            print("文档状态、向量产物与当前索引一致，跳过候选索引发布")

        # 这些文件供学习和人工查看；活动索引的权威数据仍是版本目录及其 build_manifest。
        try:
            atomic_json(config.PROCESSED_DIR / "documents.json", all_documents)
            atomic_json(config.CHUNKS_FILE, all_chunks)
        except OSError:
            print("索引已发布，但 documents.json/chunks.json 导出失败；活动版本数据仍然完整")

        reused = sum(1 for item in current.values() if item["action"] == "reuse")
        stale = sum(1 for row in records if row.get("status") == "stale")
        missing = sum(1 for row in records if row.get("status") == "missing_source")
        status = "partial" if unavailable else "complete"
        build_report = {
            "status": status, "strict": strict, "force": force, "prune_missing": prune_missing,
            "published": not already_current, "active_documents": len(records),
            "active_chunks": len(all_chunks), "reused_documents": reused,
            "updated_documents": len(pending), "stale_documents": stale,
            "missing_source_documents": missing, "failed": unavailable,
        }
        atomic_json(config.PROCESSED_DIR / "build_report.json", build_report)
        print(f"建库{('部分' if unavailable else '全部')}成功：{len(records)} 份可检索文档，"
              f"{len(all_chunks)} 个 chunk；失败/过期 {len(unavailable)} 份")
        return all_chunks
