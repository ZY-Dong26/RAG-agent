"""
service.py —— 命令行与 FastAPI 共用的问答应用服务

浏览器通过 FastAPI 提问；API 路由和命令行入口都调用 RAGService.ask()。
两种入口共用一套检索和生成规则，不在路由中复制业务流程。
"""
from dataclasses import dataclass
import re
from time import perf_counter

from rag_agent.common.progress import report, stage
from rag_agent import config
from rag_agent.qa.evidence_policy import EvidencePolicy, REFUSAL_ANSWER
from rag_agent.qa.generator import Generator
from rag_agent.qa.reranker import Reranker
from rag_agent.qa.retriever import Retriever


# 生成模型可能在约定的拒答句后追加引用编号；仅匹配整段拒答，避免把正文中的引用误判为拒答。
GENERATION_REFUSAL = re.compile(r"^资料中没有找到相关内容[。.!！]?(?:\s*\[\d+\])*\s*$")


@dataclass(frozen=True)
class AnswerResult:
    """一次问答的最终可回答状态、证据门控状态和耗时；计时不含模型预加载。"""
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
        """完成单轮问答并分别计时；证据门控拒答时不调用 LLM。"""
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
        # 证据门控只决定是否调用 LLM；LLM 仍可能判断片段不足以回答。
        # 保留 evidence_status=answerable 以说明检索阶段已通过，answerable 表示最终回答状态。
        generation_refused = bool(GENERATION_REFUSAL.fullmatch(answer))
        return AnswerResult(
            answer=answer, hits=hits, answerable=not generation_refused,
            reason="generation_refused" if generation_refused else decision.reason,
            evidence_status=decision.status,
            threshold_calibrated=decision.threshold_calibrated, timing=timing,
        )
