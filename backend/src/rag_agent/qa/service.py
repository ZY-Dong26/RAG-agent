"""
service.py —— 命令行与 FastAPI 共用的问答应用服务

浏览器通过 FastAPI 提问；API 路由和命令行入口都调用 RAGService.ask()。
两种入口共用一套检索和生成规则，不在路由中复制业务流程。
"""
from dataclasses import dataclass
from time import perf_counter

from rag_agent.common.progress import report, stage
from rag_agent import config
from rag_agent.qa.evidence_policy import EvidencePolicy, REFUSAL_ANSWER
from rag_agent.qa.generator import Generator
from rag_agent.qa.reranker import Reranker
from rag_agent.qa.retriever import Retriever


@dataclass(frozen=True)
class AnswerResult:
    """一次问答的答案、证据门控状态和各阶段耗时；计时不包含启动时的模型预加载。"""
    answer: str | None
    hits: list
    answerable: bool
    reason: str | None
    evidence_status: str
    threshold_calibrated: bool
    timing: dict[str, float | None]


class RAGService:
    """编排“混合召回 → 重排 → 证据判断 → 生成”，不负责终端输入输出。"""

    def __init__(self, retriever=None, reranker=None, evidence_policy=None, generator=None):
        """保存可注入的依赖；正常运行时各模型实例由整个会话复用。"""
        self.retriever = retriever or Retriever()
        self.reranker = reranker if reranker is not None else (
            Reranker() if config.RERANK_ENABLED else None
        )
        self.evidence_policy = evidence_policy or EvidencePolicy(config.RERANK_REJECT_THRESHOLD)
        self.generator = generator or Generator()

    def prepare(self):
        """在入口宣布就绪前加载并预热重排模型；假重排器可省略 prepare。"""
        if self.reranker is not None and hasattr(self.reranker, "prepare"):
            with stage("加载并预热 BGE 重排模型"):
                self.reranker.prepare()
            report(f"[重排模型] 运行设备：{self.reranker.device}")

    def ask(self, question):
        """完成一次单轮问答，并把召回、重排和生成分别计时；拒答不调用 LLM。"""
        started = perf_counter()
        with stage("Dense 与 BM25 召回及 RRF 融合"):
            phase_started = perf_counter()
            candidates = self.retriever.retrieve(question)
            recall_seconds = perf_counter() - phase_started

        if self.reranker is not None:
            with stage(f"BGE 重排 {len(candidates)} 条融合候选"):
                phase_started = perf_counter()
                hits = self.reranker.rerank(question, candidates)
                rerank_seconds = perf_counter() - phase_started
        else:
            hits = candidates[:config.RERANK_TOP_K]
            for rank, hit in enumerate(hits, 1):
                hit["metadata"]["rank"] = rank
            rerank_seconds = None
            report(f"[重排] 已关闭，按 RRF 排名保留 {len(hits)} 条")

        timing = {
            "recall_seconds": recall_seconds,
            "rerank_seconds": rerank_seconds,
            "retrieval_seconds": perf_counter() - started,
            "generation_seconds": None,
            "total_seconds": None,
        }
        decision = self.evidence_policy.evaluate(hits)
        report(f"[证据门控] {decision.status}"
               + ("（阈值尚未校准）" if not decision.threshold_calibrated and hits else ""))
        if not decision.answerable:
            timing["total_seconds"] = perf_counter() - started
            return AnswerResult(
                answer=REFUSAL_ANSWER, hits=hits, answerable=False,
                reason=decision.reason, evidence_status=decision.status,
                threshold_calibrated=decision.threshold_calibrated, timing=timing,
            )

        with stage("等待云端 LLM 生成回答"):
            phase_started = perf_counter()
            answer = self.generator.generate(question, hits)
            timing["generation_seconds"] = perf_counter() - phase_started
        timing["total_seconds"] = perf_counter() - started
        return AnswerResult(
            answer=answer, hits=hits, answerable=True,
            reason=decision.reason, evidence_status=decision.status,
            threshold_calibrated=decision.threshold_calibrated, timing=timing,
        )
