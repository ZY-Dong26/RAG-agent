"""
bm25_store.py —— 中文 BM25 索引层

职责：
    1. 使用同一套规则切分中文、英文、数字、型号和版本号。
    2. 基于 bm25s 构建和查询稀疏检索索引。
    3. 保存索引文件、稳定 chunk_id 顺序及文件校验值，并在加载时完整验证。

本模块不计算向量、不负责 RRF 融合，也不复制权威 chunk 正文。BM25 的第 i 条语料始终
对应同代 chunks_meta.json 的第 i 条记录；持久化时另外保存 chunk_id 顺序用于防止混读。
"""
import hashlib
import json
import logging
import re
from importlib.metadata import version
from pathlib import Path

import bm25s
import jieba

from rag_agent.common.files import atomic_json, read_json


TOKENIZER_VERSION = "jieba-search-v1"
BM25_K1 = 1.5
BM25_B = 0.75
BM25_METHOD = "lucene"
MANIFEST_FILE = "manifest.json"

# 英文分支保留 GPT-4o、Qwen3-Embedding-0.6B、v2.1 等字母数字组合；数字分支保留
# 年份、小数和百分比；连续中文交给 jieba 搜索模式产生适合召回的细粒度词。
_TOKEN_PATTERN = re.compile(
    r"[A-Za-z]+(?:[._+\-]?[A-Za-z0-9]+)*|\d+(?:\.\d+)*(?:%|％)?|[\u3400-\u9fff]+"
)
jieba.setLogLevel(logging.WARNING)


def tokenize(text):
    """
    将一段中英文混合文字转换为 BM25 token 列表。

    英文字母统一小写；标点不会单独形成 token；中文使用 lcut_for_search，使建库和查询
    共享完全相同的规则。输入为空或只有标点时返回空列表。
    """
    tokens = []
    for match in _TOKEN_PATTERN.finditer(str(text or "")):
        value = match.group(0)
        if re.fullmatch(r"[\u3400-\u9fff]+", value):
            tokens.extend(part.strip() for part in jieba.lcut_for_search(value) if part.strip())
        else:
            tokens.append(value.lower())
    return tokens


def _sha256(path):
    """分块计算单个索引文件的 SHA-256，避免一次把文件全部读入内存。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _chunk_ids(chunks):
    """提取并校验稳定 chunk_id；BM25 不接受缺失或重复标识的语料。"""
    ids = [chunk.get("metadata", {}).get("chunk_id") for chunk in chunks]
    if any(not isinstance(item, str) or not item for item in ids) or len(ids) != len(set(ids)):
        raise ValueError("BM25 语料包含缺失或重复的 chunk_id")
    return ids


class BM25Store:
    """保存与一代 FAISS 元数据顺序严格一致的 BM25 索引。"""

    def __init__(self, chunks=None, retriever=None):
        self.chunks = list(chunks or [])
        self.retriever = retriever

    @staticmethod
    def configuration():
        """返回影响本地 BM25 索引结果的版本和参数，供发布清单与重建判断使用。"""
        return {
            "tokenizer_version": TOKENIZER_VERSION,
            "bm25s_version": version("bm25s"),
            "jieba_version": version("jieba"),
            "k1": BM25_K1,
            "b": BM25_B,
            "method": BM25_METHOD,
        }

    @classmethod
    def configuration_signature(cls):
        """为 tokenizer、依赖版本及 BM25 参数生成稳定签名。"""
        value = json.dumps(cls.configuration(), ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(value).hexdigest()

    def info(self):
        """返回可写入 build_manifest.json 的 BM25 摘要，不包含正文。"""
        return {
            **self.configuration(),
            "configuration_signature": self.configuration_signature(),
            "corpus_count": len(self.chunks),
        }

    def build(self, chunks):
        """
        用权威 chunk 顺序构建内存索引。

        :param chunks: 与 FAISS 完全相同且含稳定 chunk_id 的字典列表。
        :return: self，便于 ``BM25Store().build(chunks).save(path)`` 链式调用。
        """
        chunks = list(chunks)
        if not chunks:
            raise ValueError("拒绝构建空 BM25 索引")
        _chunk_ids(chunks)
        corpus_tokens = [tokenize(chunk.get("text", "")) for chunk in chunks]
        self.retriever = bm25s.BM25(k1=BM25_K1, b=BM25_B, method=BM25_METHOD)
        self.retriever.index(corpus_tokens, show_progress=False)
        self.chunks = chunks
        return self

    def search(self, query, top_k):
        """查询 BM25，并返回和向量检索一致的 text/metadata 结构及 BM25 调试分数。"""
        if self.retriever is None or not self.chunks:
            return []
        query_tokens = tokenize(query)
        if not query_tokens or top_k <= 0:
            return []
        k = min(int(top_k), len(self.chunks))
        result = self.retriever.retrieve([query_tokens], k=k, show_progress=False)
        hits = []
        for rank, (index, score) in enumerate(zip(result.documents[0], result.scores[0]), 1):
            # bm25s 在 k 大于实际匹配数时可能补回零分文档；这些不是关键词命中，
            # 不应凭列表位置获得 RRF 加分。
            if float(score) <= 0:
                continue
            chunk = self.chunks[int(index)]
            hits.append({
                "text": chunk["text"],
                "metadata": {
                    **chunk["metadata"],
                    "bm25_rank": len(hits) + 1,
                    "bm25_score": float(score),
                },
            })
        return hits

    def save(self, directory):
        """
        保存 bm25s 文件和校验清单。调用者应传入尚未发布的新版本目录。

        清单最后写入；任一索引文件未完成时都不会出现可通过校验的 BM25 目录。
        """
        if self.retriever is None or not self.chunks:
            raise ValueError("BM25 索引尚未构建")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        self.retriever.save(directory)
        files = sorted(path for path in directory.iterdir() if path.is_file())
        manifest = {
            **self.info(),
            "chunk_ids": _chunk_ids(self.chunks),
            "artifacts": {path.name: _sha256(path) for path in files},
        }
        atomic_json(directory / MANIFEST_FILE, manifest)
        return manifest

    @classmethod
    def load(cls, directory, chunks):
        """
        回读并验证 BM25 索引、依赖配置、文件哈希、语料数量和 chunk_id 顺序。

        旧代索引没有 bm25/ 时给出明确重建提示，绝不静默降级为单路向量检索。
        """
        directory = Path(directory)
        manifest_path = directory / MANIFEST_FILE
        if not manifest_path.is_file():
            raise RuntimeError("当前活动索引不含 BM25；请运行 scripts/build_index.py 重建本地索引")
        manifest = read_json(manifest_path)
        expected_ids = _chunk_ids(chunks)
        if manifest.get("configuration_signature") != cls.configuration_signature():
            raise RuntimeError("BM25 tokenizer 或依赖配置已变化；请重新运行 scripts/build_index.py")
        if manifest.get("corpus_count") != len(chunks) or manifest.get("chunk_ids") != expected_ids:
            raise ValueError("BM25 与 FAISS 的 chunk 数量或 ID 顺序不一致")
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, dict) or not artifacts:
            raise ValueError("BM25 索引清单缺少文件校验信息")
        for name, expected_hash in artifacts.items():
            path = directory / name
            if not path.is_file() or _sha256(path) != expected_hash:
                raise ValueError(f"BM25 索引文件校验失败：{name}")
        try:
            retriever = bm25s.BM25.load(directory, load_corpus=False)
        except Exception as error:
            raise ValueError("BM25 索引无法加载或已损坏") from error
        return cls(chunks=chunks, retriever=retriever)
