# 05_retriever.py —— 检索层：问题 → 向量 → 召回 top-k chunk
# 职责：只做"召回"，不写答案。是 03(向量化) 和 04(向量库) 之间的胶水
# 关键设计：
#   1. 组合而非继承：Retriever持有 embedding + vector_store 两个实例，
#      embedding负责"问题→查询向量"，vector_store负责"向量→命中chunk"
#   2. 依赖注入：构造函数可传现成实例（测试时传假对象、问答时共享加载），
#      不传则自动从磁盘加载（模型 + FAISS索引）
# 重要约定：包内模块导入统一使用 from src import xxx

from src import config


class Retriever:
    """检索器：把用户问题转成向量，从向量库召回最相似的top_k个chunk"""

    def __init__(self, vector_store=None, embedding=None, top_k=None):
        """
        :param vector_store: 04的VectorStore实例；为None时自动load()磁盘索引
        :param embedding: 03的Embedding实例；为None时自动加载本地模型（首次较慢）
        :param top_k: 召回chunk数量，默认config.TOP_K
        """
        self.top_k = top_k or config.TOP_K

        # 向量库：优先用外部注入的实例（避免重复load文件）
        if vector_store is not None:
            self.store = vector_store
        else:
            from src import vector_store
            self.store = vector_store.VectorStore().load()

        # 查询向量化：优先用外部注入的实例（避免重复加载大模型）
        if embedding is not None:
            self.emb = embedding
        else:
            from src import embedding
            self.emb = embedding.Embedding()

    def retrieve(self, query, top_k=None):
        """
        检索主流程：query → 查询向量 → FAISS召回 → 命中chunk列表
        :param query: 用户问题字符串
        :param top_k: 本次召回数量，默认用self.top_k
        :return: hits，list[dict]，每项：
                 {"text": 原文, "metadata": {"source","page","chunk_id","score","rank"}}
        """
        k = top_k or self.top_k

        # 1. 问题向量化（03：自动套官方指令模板）
        query_vector = self.emb.embed_query(query)

        # 2. 相似度召回（04：返回带score的chunk，按相似度降序）
        hits = self.store.search(query_vector, top_k=k)

        # 3. 补上排名序号，供展示和引用编号使用
        for i, hit in enumerate(hits, 1):
            hit["metadata"]["rank"] = i

        return hits

    @staticmethod
    def format_context(hits):
        """
        把命中chunk拼成给LLM的上下文文本（带编号和来源）
        这是06_generator的输入格式：资料块之间用空行分隔，方便LLM阅读
        """
        blocks = []
        for hit in hits:
            meta = hit["metadata"]
            rank = meta.get("rank", "?")
            source = meta.get("source", "未知来源")
            page = meta.get("page", "?")
            blocks.append(f"[{rank}] 来源:{source} 第{page}页\n{hit['text'].strip()}")
        return "\n\n".join(blocks)


if __name__ == "__main__":
    # 自测：python -m src.05_retriever
    # 依赖：先跑过 python -m src.04_vector_store 建库
    retriever = Retriever()
    hits = retriever.retrieve("什么是检索增强生成？")

    print(f"召回 {len(hits)} 个chunk：\n")
    for h in hits:
        m = h["metadata"]
        print(f"[{m['rank']}] {m['source']} 第{m['page']}页 相似度={m['score']:.4f}")
        print(f"    文本预览: {h['text'][:50]}...")
