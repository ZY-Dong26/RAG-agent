"""
artifact_store.py —— 按文档保存可复用的切块和向量产物

为什么需要这一层？
    FAISS 的活动索引是一份“整库快照”。如果只保存整库文件，那么新增一份 PDF 时，
    程序无法可靠判断哪些旧向量属于哪份文档，也就无法安全复用。

本模块把每份文档的 documents、chunks 和 vectors 单独保存为不可变产物：
    vector_db/document_artifacts/<artifact_key>/
        documents.json     MinerU 适配后的统一文档记录
        chunks.json        这份文档切分得到的 chunk
        vectors.npy        与 chunks 一一对应的 float32 向量
        manifest.json      内容哈希、处理版本、数量和文件校验值

建库时可以直接读取未变化文档的产物，只对新增或变化文档运行 Embedding；最后仍然把
所有有效产物组装成一份完整候选索引，验证通过后再切换 current.json。
"""
import hashlib
import json
import os
import re
import shutil
import uuid
from pathlib import Path

import numpy as np

from rag_agent.common.files import atomic_json, read_json


_KEY_PATTERN = re.compile(r"[0-9a-f]{64}")


def fingerprint(value):
    """把可 JSON 序列化的数据转换成稳定 SHA-256，供文档 ID、配置签名和产物键使用。"""
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_hash(path):
    """分块计算文件 SHA-256，避免读取大型 PDF 或向量文件时一次占用过多内存。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def document_identity(path, raw_dir):
    """
    根据 PDF 在 data/raw 下的相对路径生成稳定文档 ID。

    文档 ID 不使用文件内容哈希：同一路径的 PDF 内容更新后仍被视为“同一份文档的新版本”。
    Windows 路径不区分大小写，所以使用 casefold 后的 POSIX 相对路径生成 ID。
    """
    path = Path(path).resolve()
    raw_dir = Path(raw_dir).resolve()
    try:
        relative = path.relative_to(raw_dir).as_posix()
    except ValueError:
        raise ValueError("待建库 PDF 必须位于 data/raw 目录内") from None
    return fingerprint({"source_path": relative.casefold()}), relative


def artifact_key(document_id, source_sha256, pipeline_signature, nonce=None):
    """
    生成单文档向量产物键。

    普通建库使用确定性键，同样的 PDF 与处理配置可以命中已有产物；--force 传入 nonce，
    强制创建一个新产物。活动清单会记录新键，后续普通运行仍可继续复用它。
    """
    value = {"document_id": document_id, "source_sha256": source_sha256,
             "pipeline_signature": pipeline_signature}
    if nonce:
        value["force_nonce"] = nonce
    return fingerprint(value)


def _artifact_dir(root, key):
    """只接受本模块生成的 64 位十六进制键，防止清单中的异常值逃出产物目录。"""
    if not isinstance(key, str) or not _KEY_PATTERN.fullmatch(key):
        raise ValueError("文档向量产物键无效")
    return Path(root) / "document_artifacts" / key


def save_artifact(root, key, documents, chunks, vectors, manifest):
    """
    原子发布一份单文档向量产物，并立即从磁盘回读验收。

    文件先写到同级临时目录，最后一次目录重命名才使产物可见。即使进程在写入中途退出，
    后续建库也不会把半份数据当作可复用产物。
    """
    root = Path(root)
    final = _artifact_dir(root, key)
    if final.exists():
        # 确定性产物已经存在时直接验收并复用，避免覆盖另一轮成功生成的数据。
        return load_artifact(root, key)

    parent = final.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = parent / ("." + key + "." + uuid.uuid4().hex + ".tmp")
    temporary.mkdir(parents=False, exist_ok=False)
    try:
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(chunks) or not np.isfinite(matrix).all():
            raise ValueError("单文档向量必须是与 chunk 数量一致的有效二维数组")
        if not chunks or not documents:
            raise ValueError("拒绝保存没有文档记录或没有 chunk 的向量产物")

        atomic_json(temporary / "documents.json", documents)
        atomic_json(temporary / "chunks.json", chunks)
        # 读写两端均禁止 pickle；这里只保存 float32 数组，不序列化任意 Python 对象。
        np.save(temporary / "vectors.npy", matrix, allow_pickle=False)
        complete_manifest = {
            **manifest,
            "artifact_key": key,
            "documents": len(documents),
            "chunks": len(chunks),
            "dimension": int(matrix.shape[1]),
            "artifacts": {
                "documents.json": file_hash(temporary / "documents.json"),
                "chunks.json": file_hash(temporary / "chunks.json"),
                "vectors.npy": file_hash(temporary / "vectors.npy"),
            },
        }
        # manifest 最后写入；它相当于“这份产物已经完整”的提交标记。
        atomic_json(temporary / "manifest.json", complete_manifest)
        os.replace(temporary, final)
    except Exception:
        # 临时目录从未被任何活动索引引用，可以安全清理；正式产物和活动索引不会受影响。
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return load_artifact(root, key)


def load_artifact(root, key):
    """
    读取并验证一份单文档产物。

    除了检查数量和维度，还复核三个数据文件的 SHA-256。这样磁盘文件被截断或手工改动时，
    程序会停止复用，而不是把向量与错误文本组合进候选索引。
    """
    directory = _artifact_dir(root, key)
    manifest_path = directory / "manifest.json"
    required = [directory / "documents.json", directory / "chunks.json", directory / "vectors.npy"]
    if not manifest_path.is_file() or not all(path.is_file() for path in required):
        raise FileNotFoundError(f"文档向量产物不完整：{key}")
    manifest = read_json(manifest_path)
    if manifest.get("artifact_key") != key:
        raise ValueError("文档向量产物键与清单不一致")
    for path in required:
        expected = manifest.get("artifacts", {}).get(path.name)
        if not expected or file_hash(path) != expected:
            raise ValueError(f"文档向量产物校验失败：{path.name}")

    documents = read_json(directory / "documents.json")
    chunks = read_json(directory / "chunks.json")
    vectors = np.load(directory / "vectors.npy", allow_pickle=False)
    if (not isinstance(documents, list) or not isinstance(chunks, list) or vectors.ndim != 2
            or len(documents) != manifest.get("documents")
            or len(chunks) != manifest.get("chunks") or vectors.shape[0] != len(chunks)
            or vectors.shape[1] != manifest.get("dimension") or not np.isfinite(vectors).all()):
        raise ValueError("文档向量产物内容、数量或维度不一致")
    return {"documents": documents, "chunks": chunks, "vectors": vectors, "manifest": manifest}
