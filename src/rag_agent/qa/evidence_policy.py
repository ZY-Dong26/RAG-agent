"""
evidence_policy.py —— 重排后的证据充分性门控

本模块只根据最终候选和已校准的 reranker 阈值做决定。Dense、BM25 与 RRF 分数均不作为
拒答依据；阈值为 None 时明确标记尚未校准，但不会擅自硬拒答。
"""
from dataclasses import dataclass


REFUSAL_ANSWER = "资料中没有找到足够相关的内容。"


@dataclass(frozen=True)
class EvidenceDecision:
    """证据门控结果；status 固定为 no_candidates、below_threshold 或 answerable。"""
    answerable: bool
    status: str
    reason: str | None
    threshold_calibrated: bool


class EvidencePolicy:
    """使用最终 Top-1 的 reranker_score 判断是否允许进入生成阶段。"""

    def __init__(self, threshold=None):
        self.threshold = threshold

    def evaluate(self, hits):
        """返回门控决定；不修改候选，也不调用任何模型。"""
        if not hits:
            return EvidenceDecision(False, "no_candidates", "没有检索候选", self.threshold is not None)
        if self.threshold is None:
            return EvidenceDecision(True, "answerable", "threshold_not_calibrated", False)
        score = hits[0].get("metadata", {}).get("rerank_score")
        if score is None:
            raise ValueError("已配置拒答阈值，但最终候选缺少 rerank_score")
        if float(score) < self.threshold:
            return EvidenceDecision(False, "below_threshold", "最佳重排分数低于阈值", True)
        return EvidenceDecision(True, "answerable", None, True)
