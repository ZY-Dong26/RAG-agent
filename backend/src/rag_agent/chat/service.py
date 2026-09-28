"""
service.py —— 命令行与 FastAPI 共用的单轮问答编排

浏览器通过 FastAPI 提问，流式路由调用 ask_stream()；命令行仍调用 ask()。
两种入口共用检索、证据判断和 qa.generator 生成；生成结束后在 chat.answer 整理
最终状态，完整回答的会话快照由 chat.snapshot 保存，具体 SQL 留在 storage.chat_history。
"""
from dataclasses import dataclass
from time import perf_counter

from rag_agent.common.progress import report, stage
from rag_agent import config
from rag_agent.retrieval.evidence_policy import EvidencePolicy, REFUSAL_ANSWER
from rag_agent.qa.generator import Generator, select_prompt_hits
from rag_agent.chat.answer import finalize_answer, is_generation_refusal
from rag_agent.retrieval.reranker import Reranker
from rag_agent.retrieval.retriever import Retriever


@dataclass(frozen=True)
class AnswerResult:
    """一次问答的最终状态；hits 是有效引用，retrieval_hits 供本地诊断复查。"""
    answer: str | None
    hits: list
    answerable: bool
    reason: str | None
    evidence_status: str
    threshold_calibrated: bool
    timing: dict[str, float | None]
    retrieval_hits: list


class RAGService:
    """编排“混合召回 → 重排 → 证据判断 → 生成 → 答案状态”，不负责终端或 HTTP。"""

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

    def _retrieve(self, question, started):
        """两种问答入口共用召回、重排和证据门控，返回最终证据、耗时及判断。"""
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
        return hits, timing, decision

    def ask(self, question):
        """完成单轮问答并分别计时；证据门控拒答时不调用 LLM。"""
        started = perf_counter()
        hits, timing, decision = self._retrieve(question, started)
        if not decision.answerable:
            timing["total_seconds"] = perf_counter() - started
            return AnswerResult(
                answer=REFUSAL_ANSWER, hits=[], answerable=False,
                reason=decision.reason, evidence_status=decision.status,
                threshold_calibrated=decision.threshold_calibrated, timing=timing,
                retrieval_hits=hits,
            )

        prompt_hits = select_prompt_hits(hits)
        if not prompt_hits:
            # 有检索结果但第一块就超出提示词预算，不能让模型在空资料上生成答案。
            report("[提示词] 没有证据能放入字符预算，本轮不调用回答模型")
            timing["total_seconds"] = perf_counter() - started
            return AnswerResult(
                answer=REFUSAL_ANSWER, hits=[], answerable=False,
                reason="context_budget_empty", evidence_status=decision.status,
                threshold_calibrated=decision.threshold_calibrated, timing=timing,
                retrieval_hits=hits,
            )

        with stage("等待云端 LLM 生成回答"):
            phase_started = perf_counter()
            answer = self.generator.generate(question, prompt_hits)
            timing["generation_seconds"] = perf_counter() - phase_started
        timing["total_seconds"] = perf_counter() - started
        # 证据门控只决定是否调用 LLM；LLM 仍可能判断片段不足以回答。
        # 保留 evidence_status=answerable 以说明检索阶段已通过，answerable 表示最终回答状态。
        generation_refused = is_generation_refusal(answer)
        answer, cited_hits, invalid_count = finalize_answer(answer, prompt_hits)
        if not answer:
            raise ValueError("回答为空")
        if invalid_count:
            report(f"[引用] 已移除 {invalid_count} 个不在本轮提示词中的编号")
        return AnswerResult(
            answer=answer, hits=cited_hits, answerable=not generation_refused,
            reason="generation_refused" if generation_refused else decision.reason,
            evidence_status=decision.status,
            threshold_calibrated=decision.threshold_calibrated, timing=timing,
            retrieval_hits=hits,
        )

    def ask_stream(self, question):
        """依次产出检索状态、生成增量和最终结果；不在服务层写聊天数据库。"""
        started = perf_counter()
        yield "status", {"stage": "retrieving", "message": "正在检索并重排资料…"}
        hits, timing, decision = self._retrieve(question, started)
        prompt_hits = select_prompt_hits(hits) if decision.answerable else []
        # SSE 提前公开的来源只能来自实际提示词；最终 done 再收窄到答案引用的来源。
        yield "retrieval", {"hits": prompt_hits, "timing": timing.copy()}
        if not decision.answerable:
            timing["total_seconds"] = perf_counter() - started
            yield "done", AnswerResult(
                answer=REFUSAL_ANSWER, hits=[], answerable=False,
                reason=decision.reason, evidence_status=decision.status,
                threshold_calibrated=decision.threshold_calibrated, timing=timing,
                retrieval_hits=hits,
            )
            return

        if not prompt_hits:
            report("[提示词] 没有证据能放入字符预算，本轮不调用回答模型")
            timing["total_seconds"] = perf_counter() - started
            yield "done", AnswerResult(
                answer=REFUSAL_ANSWER, hits=[], answerable=False,
                reason="context_budget_empty", evidence_status=decision.status,
                threshold_calibrated=decision.threshold_calibrated, timing=timing,
                retrieval_hits=hits,
            )
            return

        yield "status", {"stage": "generating", "message": "正在生成回答…"}
        parts = []
        with stage("等待云端 LLM 流式生成回答"):
            phase_started = perf_counter()
            for delta in self.generator.stream(question, prompt_hits):
                parts.append(delta)
                yield "delta", {"text": delta}
            timing["generation_seconds"] = perf_counter() - phase_started
        answer = "".join(parts).strip()
        if not answer:
            raise ValueError("回答为空")
        timing["total_seconds"] = perf_counter() - started
        generation_refused = is_generation_refusal(answer)
        answer, cited_hits, invalid_count = finalize_answer(answer, prompt_hits)
        if not answer:
            raise ValueError("回答为空")
        if invalid_count:
            report(f"[引用] 已移除 {invalid_count} 个不在本轮提示词中的编号")
        yield "done", AnswerResult(
            answer=answer, hits=cited_hits, answerable=not generation_refused,
            reason="generation_refused" if generation_refused else decision.reason,
            evidence_status=decision.status,
            threshold_calibrated=decision.threshold_calibrated, timing=timing,
            retrieval_hits=hits,
        )
