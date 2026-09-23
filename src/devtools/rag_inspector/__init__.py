"""RAG 单次问答诊断记录与可视化报告。"""

from .recorder import RecordingClient, TraceRecorder, load_neighbor_chunks
from .report import write_trace_report

__all__ = ["RecordingClient", "TraceRecorder", "load_neighbor_chunks", "write_trace_report"]
