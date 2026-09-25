"""
service.py —— 命令行与未来前端共用的问答应用服务

前端不应直接操作 FAISS、BM25、重排器或 LLM 客户端，只调用 RAGService.ask()。
这样以后增加 FastAPI、桌面界面或网页时，检索和生成规则仍然只有一份实现。
"""
from dataclasses import dataclass
from rag_agent.common.progress import report, stage
from rag_agent import config

from rag_agent.qa.evidence_policy import EvidencePolicy, REFUSAL_ANSWER
from rag_agent.qa.generator import Generator
from rag_agent.qa.reranker import Reranker
from rag_agent.qa.retriever import Retriever


@dataclass(frozen=True)
class AnswerResult:
    """一次问答的统一返回值，包含证据门控状态，供 CLI 和未来前端明确区分拒答原因。"""
    answer: str | None
    hits: list
    answerable: bool
    reason: str | None
    evidence_status: str
    threshold_calibrated: bool


class RAGService:
    """编排“混合召回 → 重排 → 证据判断 → 生成”，不负责终端输入输出。"""

    def __init__(self, retriever=None, reranker=None, evidence_policy=None, generator=None):
        # 允许外部传入替身对象，便于测试；正常运行时各依赖只创建一次。
        """保存注入的检索器和生成器；未传入时创建默认依赖，供后续各轮问答复用。"""
        self.retriever = retriever or Retriever()
        self.reranker = reranker if reranker is not None else (
            Reranker() if config.RERANK_ENABLED else None
        )
        self.evidence_policy = evidence_policy or EvidencePolicy(config.RERANK_REJECT_THRESHOLD)
        self.generator = generator or Generator()

    def ask(self, question):
        """完成一次单轮问答；证据不充分时直接返回统一拒答文案，不调用 LLM。"""
        with stage("Dense 与 BM25 召回及 RRF 融合"):
            candidates = self.retriever.retrieve(question)

        if self.reranker is not None:
            with stage(f"BGE 重排 {len(candidates)} 条融合候选"):
                hits = self.reranker.rerank(question, candidates)
        else:
            hits = candidates[:config.RERANK_TOP_K]
            for rank, hit in enumerate(hits, 1):
                hit["metadata"]["rank"] = rank
            report(f"[重排] 已关闭，按 RRF 排名保留 {len(hits)} 条")

        decision = self.evidence_policy.evaluate(hits)
        report(f"[证据门控] {decision.status}"
               + ("（阈值尚未校准）" if not decision.threshold_calibrated and hits else ""))
        if not decision.answerable:
            return AnswerResult(
                answer=REFUSAL_ANSWER,
                hits=hits,
                answerable=False,
                reason=decision.reason,
                evidence_status=decision.status,
                threshold_calibrated=decision.threshold_calibrated,
            )
        with stage("等待云端 LLM 生成回答"):
            answer = self.generator.generate(question, hits)
        return AnswerResult(
            answer=answer,
            hits=hits,
            answerable=True,
            reason=decision.reason,
            evidence_status=decision.status,
            threshold_calibrated=decision.threshold_calibrated,
        )
