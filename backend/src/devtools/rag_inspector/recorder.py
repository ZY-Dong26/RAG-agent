"""
recorder.py —— 透明记录一轮 RAG 调试问答的完整轨迹

职责：
    1. 保存问题、命中 chunk、实际发送的 messages、模型回答和各阶段耗时。
    2. 用 RecordingClient 代理 OpenAI 客户端，只拦截 chat.completions.create，记录请求参数和回答。
    3. 额外读取活动索引的 chunk 元数据，为每个命中块找同一文档中相邻 segment。

设计原因：
    - 调试时需要知道"到底发给模型什么了"，而不是猜测。RecordingClient 在不改动 Generator 的前提下截获请求。
    - 不记录 API Key 或客户端连接信息；只保存可序列化的请求参数和回答文本。
    - 相邻 chunk 单独加载，不污染检索器本身；radius=1 表示命中块前后各取一个 segment。
"""
from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any


def _jsonable(value: Any) -> Any:
    """
    把请求参数递归转成可写入 JSON 的普通对象。

    输入：Generator 传给 SDK 的任意 kwargs。
    输出：只含 str/int/float/bool/dict/list 的结构；其他类型一律转字符串。

    设计原因：SDK 内部对象（如 httpx.Client）不可序列化，这里只取字面量字段，
    避免把客户端连接信息或 API Key 写进调试报告。
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


@dataclass
class TraceRecorder:
    """
    一轮调试问答的完整数据记录器。

    字段按问答流程填充：record_retrieval → record_request → record_answer/record_error。
    不包含 API Key 或客户端连接信息。
    """

    question: str
    hits: list[dict] = field(default_factory=list)
    retrieval_seconds: float | None = None
    request: dict = field(default_factory=dict)
    answer: str | None = None
    generation_seconds: float | None = None
    error: str | None = None
    neighbors: dict[str, list[dict]] = field(default_factory=dict)

    def record_retrieval(self, hits: list[dict], elapsed: float) -> None:
        self.hits = deepcopy(hits)
        self.retrieval_seconds = elapsed

    def record_request(self, request: dict) -> None:
        self.request = _jsonable(deepcopy(request))

    def record_answer(self, answer: str, elapsed: float) -> None:
        self.answer = answer
        self.generation_seconds = elapsed

    def record_error(self, error: Exception, elapsed: float | None = None) -> None:
        self.error = f"{type(error).__name__}: {error}"
        self.generation_seconds = elapsed

    def as_dict(self) -> dict:
        return {
            "question": self.question,
            "timing": {
                "retrieval_seconds": self.retrieval_seconds,
                "generation_seconds": self.generation_seconds,
            },
            "request": self.request,
            "hits": self.hits,
            "neighbors": self.neighbors,
            "answer": self.answer,
            "error": self.error,
        }


class _RecordingCompletions:
    """代理 chat.completions.create：先记录请求 kwargs，再转发给真实客户端。"""
    def __init__(self, delegate, recorder: TraceRecorder):
        self._delegate = delegate
        self._recorder = recorder

    def create(self, *args, **kwargs):
        # Generator 传给 SDK 的参数在这里被原样截获；其中没有 API Key。
        self._recorder.record_request(kwargs)
        started = perf_counter()
        try:
            response = self._delegate.create(*args, **kwargs)
        except Exception as error:
            self._recorder.record_error(error, perf_counter() - started)
            raise
        answer = response.choices[0].message.content.strip()
        self._recorder.record_answer(answer, perf_counter() - started)
        return response


class _RecordingChat:
    """代理 client.chat，只替换 completions 属性；其他属性透传给真实客户端。"""
    def __init__(self, delegate, recorder: TraceRecorder):
        self._delegate = delegate
        self.completions = _RecordingCompletions(delegate.completions, recorder)

    def __getattr__(self, name):
        return getattr(self._delegate, name)


class RecordingClient:
    """
    OpenAI 客户端的透明代理，只拦截 chat.completions.create。

    用法：generator.client = RecordingClient(原始客户端, recorder)，用完后恢复原值。
    其他属性（如 base_url、api_key）通过 __getattr__ 透传，不暴露给记录器。
    """

    def __init__(self, delegate, recorder: TraceRecorder):
        self._delegate = delegate
        self.chat = _RecordingChat(delegate.chat, recorder)

    def __getattr__(self, name):
        return getattr(self._delegate, name)


def _active_chunks_path(project_root: Path) -> Path | None:
    vector_root = project_root / "data" / "vector_db"
    pointer = vector_root / "current.json"
    if pointer.exists():
        generation = json.loads(pointer.read_text(encoding="utf-8")).get("generation")
        candidate = vector_root / "generations" / str(generation) / "chunks_meta.json"
    else:
        candidate = vector_root / "chunks_meta.json"
    return candidate if candidate.exists() else None


def load_neighbor_chunks(project_root: Path, hits: list[dict], radius: int = 1) -> dict[str, list[dict]]:
    """
    读取活动索引元数据，为每个命中块找同一文档中相邻的 segment。

    输入：
        project_root: 项目根目录。
        hits: 检索返回的命中块列表。
        radius: 向前后各取几个 segment，默认 1（命中块前后各一个）。

    输出：{命中排名: [相邻chunk列表]}，用于调试"命中块上下文是否足够"。

    设计原因：FAISS 只返回相似度最高的 chunk，但答案可能需要它前后的上下文。
    这里从 chunks_meta.json 按 document_id + segment_index 定位相邻块，不影响检索本身。
    """
    chunks_path = _active_chunks_path(Path(project_root))
    if chunks_path is None or radius < 1:
        return {}
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    by_document: dict[tuple, dict[int, dict]] = {}
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        segment = metadata.get("segment_index")
        if not isinstance(segment, int):
            continue
        key = (metadata.get("document_id"), metadata.get("source"))
        by_document.setdefault(key, {})[segment] = chunk

    result: dict[str, list[dict]] = {}
    for fallback_rank, hit in enumerate(hits, 1):
        metadata = hit.get("metadata", {})
        rank = str(metadata.get("rank", fallback_rank))
        segment = metadata.get("segment_index")
        key = (metadata.get("document_id"), metadata.get("source"))
        if not isinstance(segment, int) or key not in by_document:
            continue
        nearby = []
        for offset in range(-radius, radius + 1):
            if offset == 0:
                continue
            chunk = by_document[key].get(segment + offset)
            if chunk is not None:
                nearby.append(deepcopy(chunk))
        if nearby:
            result[rank] = nearby
    return result
