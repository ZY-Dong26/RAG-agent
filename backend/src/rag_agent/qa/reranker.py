"""
reranker.py —— 本地 BGE Cross-Encoder 重排器

职责：对 RRF 候选的 ``(问题, 文本块)`` 配对批量打分，并按原始 logit 降序返回最终证据。
模型采用延迟加载，一个 Reranker 实例在多轮问答中只加载一次；空候选不会触发模型加载。
"""
import math
from pathlib import Path

from rag_agent import config


def _sigmoid(value):
    """数值稳定的 sigmoid，仅用于展示可读分数；该值不声明为真实概率。"""
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


class Reranker:
    """使用本地 BGE 模型或注入的假评分器重排候选。"""

    def __init__(self, scorer=None, model_name=None, device=None, max_length=None,
                 batch_size=None, top_k=None):
        """
        :param scorer: 测试注入函数 ``scorer(query, texts) -> logits``；传入后不加载真实模型。
        :param model_name: 本地模型目录，默认读取 config.RERANKER_MODEL。
        :param device: cuda/cpu；None 时在首次加载时自动选择。
        """
        self.scorer = scorer
        self.model_name = model_name or config.RERANKER_MODEL
        self.device = device
        self.max_length = max_length or config.RERANK_MAX_LENGTH
        self.batch_size = batch_size or config.RERANK_BATCH_SIZE
        self.top_k = top_k or config.RERANK_TOP_K
        self.tokenizer = None
        self.model = None

    def _load_model(self):
        """只从本地目录加载一次模型；CUDA 半精度失败时安全回退到 float32。"""
        if self.scorer is not None or self.model is not None:
            return
        if not Path(self.model_name).is_dir():
            raise FileNotFoundError(
                f"找不到本地重排模型目录：{self.model_name}\n"
                "请准备 BAAI/bge-reranker-v2-m3 的完整本地目录，程序不会自动下载模型。"
            )
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
        model_args = {"local_files_only": True}
        if self.device == "cuda":
            model_args["torch_dtype"] = torch.float16
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, **model_args
            ).to(self.device)
        except (RuntimeError, ValueError, TypeError):
            # 个别 GPU、驱动或模型结构不支持半精度分类头；仅在 CUDA 半精度路径回退。
            if self.device != "cuda":
                raise
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, local_files_only=True
            ).to(self.device)
        self.model.eval()

    def _model_logits(self, query, texts):
        """按 batch 推理并返回与输入文本一一对应的 Python float logit。"""
        import torch

        logits = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            pairs = [[query, text] for text in batch]
            encoded = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {name: value.to(self.device) for name, value in encoded.items()}
            with torch.no_grad():
                output = self.model(**encoded).logits
            output = output.reshape(len(batch), -1)
            # bge-reranker-v2-m3 为单 logit；兼容多分类头时取最后一列作为相关类分数。
            column = output[:, 0] if output.shape[1] == 1 else output[:, -1]
            logits.extend(float(value) for value in column.detach().float().cpu().tolist())
        return logits

    def rerank(self, query, candidates, top_k=None):
        """
        对融合候选批量评分，保留 raw logit 与 sigmoid 展示分数，返回最终 Top-K。

        sigmoid 只做单调映射，方便阅读；拒答阈值若启用，应针对当前模型和评测集校准。
        """
        candidates = list(candidates)
        if not candidates:
            return []
        self._load_model()
        texts = [item["text"] for item in candidates]
        raw_logits = (self.scorer(query, texts) if self.scorer is not None
                      else self._model_logits(query, texts))
        if len(raw_logits) != len(candidates):
            raise ValueError("重排评分数量与候选数量不一致")

        scored = []
        for item, raw_logit in zip(candidates, raw_logits):
            logit = float(raw_logit)
            if not math.isfinite(logit):
                raise ValueError("重排模型返回 NaN 或 Inf")
            scored.append({
                "text": item["text"],
                "metadata": {
                    **item["metadata"],
                    "rerank_logit": logit,
                    "rerank_score": _sigmoid(logit),
                },
            })
        limit = self.top_k if top_k is None else top_k
        scored.sort(key=lambda item: (
            -item["metadata"]["rerank_logit"],
            item["metadata"].get("fusion_rank", 10**9),
            item["metadata"].get("chunk_id", ""),
        ))
        result = scored[:max(0, limit)]
        for rank, item in enumerate(result, 1):
            item["metadata"]["rank"] = rank
            item["metadata"]["rerank_rank"] = rank
        return result
