"""
schemas.py —— 浏览器与问答服务之间的稳定数据格式

只返回页面需要的字段，不把 FAISS 元数据、磁盘绝对路径或客户端配置原样暴露给浏览器。
后续增加流式接口时仍沿用同一份问题和最终结果格式。
"""
from pydantic import BaseModel, Field, field_validator


class AskRequest(BaseModel):
    """单轮提问；会话 ID 只用于存储归属，不把历史消息交给 RAG 模型。"""

    question: str = Field(min_length=1, max_length=2000)
    conversation_id: str | None = None

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
    """一次问答的最终答案状态、检索门控状态、引用和耗时。"""

    conversation_id: str
    answer: str | None
    answerable: bool  # 最终是否回答；LLM 在证据通过后仍可能拒答
    reason: str | None  # generation_refused 表示生成模型拒答
    evidence_status: str  # 仅表示检索证据门控，不代表最终回答
    threshold_calibrated: bool
    citations: list[Citation]
    timing: Timing


class ConversationSummary(BaseModel):
    """侧栏所需的会话信息；列表接口不传输消息正文。"""

    id: str
    title: str
    created_at: str
    updated_at: str


class StoredMessage(BaseModel):
    """历史消息；用户消息没有回答快照，助手消息保存当轮最终结果。"""

    id: int
    role: str
    text: str
    created_at: str
    answerable: bool | None = None
    reason: str | None = None
    evidence_status: str | None = None
    threshold_calibrated: bool | None = None
    citations: list[Citation] | None = None
    timing: Timing | None = None


class ConversationDetail(ConversationSummary):
    """打开旧会话时一次返回按保存顺序排列的消息。"""

    messages: list[StoredMessage]


class StatusResponse(BaseModel):
    """已加载服务的只读状态，不包含密钥或模型绝对路径。"""

    ready: bool
    vector_count: int
    index_version: str | None
    reranker_enabled: bool
    reranker_device: str | None
