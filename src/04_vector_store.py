# 04_vector_store.py —— 向量存储层：用FAISS把向量和chunk关联起来
# 职责：只负责【向量入库、相似度检索、索引持久化】，不关心向量怎么来
# 检索约定：03输出的向量已L2归一化 → 用IndexFlatIP(内积)，内积即余弦相似度
# 持久化：vector_db/index.faiss（FAISS二进制索引）+ vector_db/chunks_meta.json（chunk元信息）
#         索引第i行 ↔ chunks_meta.json第i个chunk，一一对应
# 重要约定：包内模块导入统一使用 from src import xxx
# 注意：文件名带数字前缀，不能用点号直接导入（如 import src.04_vector_store 会语法报错）

import json
from pathlib import Path

import faiss
import numpy as np

from src import config


class VectorStore:
    """
    基于FAISS的向量库
    说明：FAISS索引只存向量，不存文本。所以并行保存一份chunk列表(JSON)，
         检索时用FAISS返回的索引行号，去chunk列表里取回对应的文本与元信息。
    """

    def __init__(self, persist_dir=None):
        """
        初始化：指定索引与元数据的存放目录
        :param persist_dir: 持久化目录，默认使用config.VECTOR_DB_DIR
        """
        # 目录不存在也没关系，save()时会自动创建
        self.persist_dir = Path(persist_dir or config.VECTOR_DB_DIR)
        # 两个持久化文件：FAISS二进制索引 + chunk元信息JSON
        self.index_path = self.persist_dir / "index.faiss"
        self.meta_path = self.persist_dir / "chunks_meta.json"

        # 运行时状态：index是FAISS索引对象；chunks是与索引行号一一对应的chunk列表
        self.index = None
        self.chunks = []

    def add(self, chunks, vectors):
        """
        批量入库：chunk列表 + 对应向量列表 → FAISS索引
        参数：
            chunks：chunk字典列表，[{"text": "...", "metadata": {...}}, ...]
            vectors：向量列表，[[0.1, ...], [0.2, ...], ...]，
                     每个向量维度必须一致，且与03输出一致（1024维）
        """
        # 空数据直接返回，避免下面数组维度判断报错
        if not chunks or not vectors:
            print("没有chunk或向量可入库，跳过")
            return

        # list → np.float32二维数组：FAISS只接受这种格式
        # shape = [文本数量, 向量维度]
        matrix = np.asarray(vectors, dtype=np.float32)
        if len(matrix.shape) != 2:
            raise ValueError(f"vectors必须是二维数组[文本数, 维度]，实际shape={matrix.shape}")

        # 创建内积索引：03输出已归一化，内积等价余弦相似度，值越大越相似
        # IndexFlatIP是暴力检索（精确但慢），小库足够；数据量大后可换HNSW等近似索引
        self.index = faiss.IndexFlatIP(matrix.shape[1])
        # 全部向量一次性入库，行号按加入顺序0,1,2...分配
        self.index.add(matrix)

        # 保存chunk列表，与索引行号一一对应
        self.chunks = chunks
        print(f"入库完成: {len(chunks)} 个chunk，向量维度 {matrix.shape[1]}")

    def search(self, query_vector, top_k=None):
        """
        相似度检索：查询向量 → 最相似的top_k个chunk
        参数：
            query_vector：查询向量（一维list，与入库向量同维度）
            top_k：返回数量，默认config.TOP_K
        返回：
            hits：列表，每项是chunk字典，metadata里追加了相似度score
                  [{"text": "...", "metadata": {"source","page","chunk_id","score"}}, ...]
        """
        # 空库防护：还没入库或库为空时直接返回空列表
        if self.index is None or self.index.ntotal == 0:
            print("向量库为空，请先调用add()入库或load()加载索引")
            return []

        top_k = top_k or config.TOP_K
        # 查询向量转成float32二维数组：FAISS要求 [查询数, 维度]
        q = np.asarray([query_vector], dtype=np.float32)
        # search返回两个数组：
        #   distances：相似度分数（归一化后=余弦相似度，越大越相似）
        #   indices：命中的索引行号（-1表示结果不足）
        distances, indices = self.index.search(q, top_k)

        hits = []
        for score, idx in zip(distances[0], indices[0]):
            if idx == -1:  # 过滤空结果（chunk数少于top_k时会出现）
                continue
            chunk = self.chunks[int(idx)]
            # 拷贝metadata并追加score，下游(retriever/generator)直接用
            hits.append({
                "text": chunk["text"],
                "metadata": {**chunk["metadata"], "score": float(score)},
            })
        return hits

    def save(self):
        """持久化：FAISS索引写二进制文件，chunk元信息写JSON"""
        if self.index is None:
            print("没有索引可保存，请先调用add()入库")
            return
        # 目录不存在则创建
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        # FAISS官方序列化：索引存二进制
        faiss.write_index(self.index, str(self.index_path))
        # chunk列表存JSON：检索时要靠它把行号映射回文本
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)
        print(f"索引已保存: {self.index_path}")
        print(f"chunk元信息已保存: {self.meta_path}")

    def load(self):
        """从磁盘恢复索引与chunk元信息（问答阶段每次启动调用）"""
        if not (self.index_path.exists() and self.meta_path.exists()):
            raise FileNotFoundError(
                f"找不到索引文件：{self.index_path}\n"
                "请先运行 python -m src.04_vector_store 建库。"
            )
        self.index = faiss.read_index(str(self.index_path))
        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        print(f"索引已加载: {self.index_path}，共 {self.index.ntotal} 个向量")
        return self


if __name__ == "__main__":
    # 自测/建库入口：python -m src.04_vector_store
    # 流程：读chunks.json → 批量向量化 → 入库 → 保存索引 → 检索自测

    # 1. 读取02_text_splitter产出的chunk
    if not config.CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"找不到 {config.CHUNKS_FILE}\n"
            "请先运行 python -m src.02_text_splitter 生成chunk。"
        )
    with open(config.CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"读取到 {len(chunks)} 个chunk")

    # 2. 批量向量化（延迟导入：只在建库时加载03的模型，平时不背这个大依赖）
    from src import embedding
    emb = embedding.Embedding()
    print(f"正在向量化 {len(chunks)} 个chunk（device={emb.device}，请稍等）...")
    vectors = emb.embed_texts([c["text"] for c in chunks])
    print(f"向量化完成: {len(vectors)} 条，维度 {len(vectors[0])}")

    # 3. 入库 + 持久化
    store = VectorStore()
    store.add(chunks, vectors)
    store.save()

    # 4. 检索自测：拿一个问题验证能否召回相关chunk
    test_query = "什么是检索增强生成？"
    query_vector = emb.embed_query(test_query)
    hits = store.search(query_vector, top_k=3)

    print(f"\n=====检索自测: {test_query}=====")
    for i, hit in enumerate(hits, 1):
        meta = hit["metadata"]
        print(f"[{i}] 来源:{meta['source']} 页码:{meta['page']} 相似度:{meta['score']:.4f}")
        print(f"    文本预览: {hit['text'][:60]}...")
