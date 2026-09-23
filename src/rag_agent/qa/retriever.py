"""
retriever.py —— Dense + BM25 混合召回与 RRF 融合

职责：
    1. 用 Qwen3 Embedding + FAISS 召回语义相近的候选。
    2. 用中文 BM25 召回关键词、型号和数字匹配的候选。
    3. 按稳定 chunk_id 去重，以加权 RRF 融合两路排名。

本模块不加载重排模型、不判断证据是否充分，也不调用生成模型。Dense 余弦分数与 BM25
分数的量纲不同，不能直接相加；RRF 只使用各自排名，使两路结果可以稳定组合。
"""
from rag_agent import config
from rag_agent.common.progress import report


def reciprocal_rank_fusion(dense_hits, bm25_hits, top_k=None, rrf_k=None,
                           dense_weight=None, bm25_weight=None):
    """
    按 chunk_id 合并两路候选，并计算 ``weight / (rrf_k + rank)``。

    每条结果保留 dense_rank/dense_score、bm25_rank/bm25_score 与 fusion_score；只在一路
    出现的候选同样参与排序。相同融合分数按最佳单路排名和 chunk_id 稳定排序。
    """
    top_k = config.FUSION_TOP_K if top_k is None else top_k
    rrf_k = config.RRF_K if rrf_k is None else rrf_k
    dense_weight = config.RRF_DENSE_WEIGHT if dense_weight is None else dense_weight
    bm25_weight = config.RRF_BM25_WEIGHT if bm25_weight is None else bm25_weight
    if top_k <= 0 or rrf_k < 0:
        return []

    merged = {}

    def add(hits, route, weight):
        for fallback_rank, hit in enumerate(hits, 1):
            metadata = hit.get("metadata", {})
            chunk_id = metadata.get("chunk_id")
            if not chunk_id:
                raise ValueError("混合召回结果缺少稳定 chunk_id")
            rank = int(metadata.get(f"{route}_rank", fallback_rank))
            item = merged.setdefault(chunk_id, {
                "text": hit["text"],
                "metadata": {key: value for key, value in metadata.items()
                             if key not in {"score", "rank"}},
                "_best_rank": rank,
                "_score": 0.0,
            })
            item["_best_rank"] = min(item["_best_rank"], rank)
            item["_score"] += weight / (rrf_k + rank)
            item["metadata"][f"{route}_rank"] = rank
            route_score = metadata.get(f"{route}_score", metadata.get("score"))
            if route_score is not None:
                item["metadata"][f"{route}_score"] = float(route_score)

    add(dense_hits, "dense", dense_weight)
    add(bm25_hits, "bm25", bm25_weight)
    ordered = sorted(merged.values(), key=lambda item: (
        -item["_score"], item["_best_rank"], item["metadata"]["chunk_id"],
    ))[:top_k]
    for rank, item in enumerate(ordered, 1):
        item["metadata"]["fusion_score"] = item.pop("_score")
        item["metadata"]["fusion_rank"] = rank
        item["metadata"]["rank"] = rank
        item.pop("_best_rank")
    return ordered


class Retriever:
    """执行两路召回并返回去重后的 RRF 候选，供后续重排器使用。"""

    def __init__(self, vector_store=None, bm25_store=None, embedding=None,
                 dense_top_k=None, bm25_top_k=None, fusion_top_k=None):
        self.dense_top_k = dense_top_k or config.DENSE_TOP_K
        self.bm25_top_k = bm25_top_k or config.BM25_TOP_K
        self.fusion_top_k = fusion_top_k or config.FUSION_TOP_K

        if vector_store is not None:
            self.store = vector_store
        else:
            from rag_agent.indexing.faiss_store import VectorStore
            self.store = VectorStore().load()

        if bm25_store is not None:
            self.bm25 = bm25_store
        else:
            from rag_agent.indexing.bm25_store import BM25Store
            # VectorStore.load 已把 index_path 固定到当前代；BM25 必须从同一目录回读，
            # 因而 current.json 在初始化过程中即使变化也不会造成跨代混读。
            self.bm25 = BM25Store.load(self.store.index_path.parent / "bm25", self.store.chunks)

        if embedding is not None:
            self.emb = embedding
        else:
            from rag_agent.indexing.embedder import Embedding
            self.emb = Embedding()

    def retrieve(self, query, top_k=None):
        """
        执行 Dense Top-N、BM25 Top-N 与 RRF，返回最多 fusion_top_k 条候选。

        :param top_k: 可覆盖本次融合候选数，主要供离线评测使用；正常问答采用集中配置。
        """
        query_vector = self.emb.embed_query(query)
        dense_hits = self.store.search(query_vector, top_k=self.dense_top_k)
        for rank, hit in enumerate(dense_hits, 1):
            score = hit["metadata"].pop("score", None)
            hit["metadata"]["dense_rank"] = rank
            if score is not None:
                hit["metadata"]["dense_score"] = float(score)
        report(f"[召回] Dense {len(dense_hits)} 条，BM25 正在查询")

        bm25_hits = self.bm25.search(query, top_k=self.bm25_top_k)
        report(f"[召回] BM25 {len(bm25_hits)} 条")
        fused = reciprocal_rank_fusion(
            dense_hits,
            bm25_hits,
            top_k=self.fusion_top_k if top_k is None else top_k,
        )
        report(f"[融合] RRF 去重后保留 {len(fused)} 条候选")
        return fused

    @staticmethod
    def format_context(hits):
        """把候选整理为带来源和页码的可读文本，供调试工具使用。"""
        blocks = []
        for hit in hits:
            meta = hit["metadata"]
            blocks.append(
                f"[{meta.get('rank', '?')}] 来源:{meta.get('source', '未知来源')} "
                f"第{meta.get('page', '?')}页\n{hit['text'].strip()}"
            )
        return "\n\n".join(blocks)
