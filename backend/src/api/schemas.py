"""
schemas.py —— 浏览器与问答服务之间的稳定数据格式

只返回页面需要的字段，不把 FAISS 元数据、磁盘绝对路径或客户端配置原样暴露给浏览器。
后续增加流式接口时仍沿用同一份问题和最终结果格式。
"""
from pydantic import BaseModel, Field, field_validator


class AskRequest(BaseModel):
    """单轮提问；当前问答不读取浏览器显示的历史消息。"""

    question: str = Field(min_length=1, max_length=2000)

    @field_validator("question")
    @classmethod
    def nonempty_question(cls, value: str) -> str:
        """去掉首尾空白；只有空白的请求在进入模型前拒绝。"""
        value = value.strip()
        if not value:
            raise ValueError("问题不能为空")
        return value


class Citation(BaseModel):
    """最终证据的公开字段；rank 对应回答里的 [编号]。"""

    rank: int | None
    source: str
    page: int | str | None
    text: str
    rerank_score: float | None
    fusion_score: float | None


class Timing(BaseModel):
    """秒数；None 表示该阶段没有执行，例如拒答时的生成。"""

    recall_seconds: float | None
    rerank_seconds: float | None
    retrieval_seconds: float | None
    generation_seconds: float | None
    total_seconds: float | None


class AskResponse(BaseModel):
    """一次问答的答案、门控状态、引用和耗时。"""

    answer: str | None
    answerable: bool
    reason: str | None
    evidence_status: str
    threshold_calibrated: bool
    citations: list[Citation]
    timing: Timing


class StatusResponse(BaseModel):
    """已加载服务的只读状态，不包含密钥或模型绝对路径。"""

    ready: bool
    vector_count: int
    index_version: str | None
    reranker_enabled: bool
    reranker_device: str | None
