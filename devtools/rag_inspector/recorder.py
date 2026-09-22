"""透明记录实际 LLM 请求，并补充检索结果与相邻文本块。"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any


def _jsonable(value: Any) -> Any:
    """把请求参数转换为可写入 JSON 的普通对象，不读取客户端内部配置。"""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


@dataclass
class TraceRecorder:
    """保存一轮调试问答的数据；不包含 API Key 或客户端连接信息。"""

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
    def __init__(self, delegate, recorder: TraceRecorder):
        self._delegate = delegate
        self.completions = _RecordingCompletions(delegate.completions, recorder)

    def __getattr__(self, name):
        return getattr(self._delegate, name)


class RecordingClient:
    """OpenAI 客户端的透明代理，只拦截 chat.completions.create。"""

    def __init__(self, delegate, recorder: TraceRecorder):
        self._delegate = delegate
        self.chat = _RecordingChat(delegate.chat, recorder)

    def __getattr__(self, name):
        return getattr(self._delegate, name)


def _active_chunks_path(project_root: Path) -> Path | None:
    vector_root = project_root / "vector_db"
    pointer = vector_root / "current.json"
    if pointer.exists():
        generation = json.loads(pointer.read_text(encoding="utf-8")).get("generation")
        candidate = vector_root / "generations" / str(generation) / "chunks_meta.json"
    else:
        candidate = vector_root / "chunks_meta.json"
    return candidate if candidate.exists() else None


def load_neighbor_chunks(project_root: Path, hits: list[dict], radius: int = 1) -> dict[str, list[dict]]:
    """读取活动索引元数据，为每个命中块找同一文档中相邻的 segment。"""
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
