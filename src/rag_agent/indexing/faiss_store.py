# faiss_store.py —— 向量存储层：用FAISS把向量和chunk关联起来
# 职责：只负责【向量入库、相似度检索、索引持久化】，不关心向量怎么来
# 检索约定：embedder 输出的向量已L2归一化 → 用IndexFlatIP(内积)，内积即余弦相似度
# 持久化：vector_db/generations/<版本>/ 下保存 FAISS、BM25 与 chunks_meta.json
#          current.json 指定活动版本；没有版本指针时兼容根目录旧文件
#         索引第i行 ↔ chunks_meta.json第i个chunk，一一对应

import json
import uuid
import re
from pathlib import Path

import faiss
import numpy as np

from rag_agent import config
from rag_agent.common.files import atomic_json, read_json


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
        # 初始路径兼容旧索引；load/save 会按活动版本更新实际文件路径
        self.index_path = self.persist_dir / "index.faiss"
        self.meta_path = self.persist_dir / "chunks_meta.json"

        # 运行时状态：index是FAISS索引对象；chunks是与索引行号一一对应的chunk列表
        self.index = None
        self.chunks = []

    def add(self, chunks, vectors):
        """
        完整构建：chunk 列表 + 对应向量列表 → 新 FAISS 索引。
        注意：add 每次都会重建内存索引，不是在已有索引后追加；增量复用由 builder 负责。
        参数：
            chunks：chunk字典列表，[{"text": "...", "metadata": {...}}, ...]
            vectors：向量列表，[[0.1, ...], [0.2, ...], ...]，
                     每个向量维度必须一致，且与 embedder 的输出一致（当前模型为 1024 维）
        """
        # 空数据直接返回，避免下面数组维度判断报错
        if not chunks or not vectors:
            print("没有chunk或向量可入库，跳过")
            return

        # list → np.float32二维数组：本项目统一使用连续的 float32 二维数据
        # shape = [文本数量, 向量维度]
        matrix = np.asarray(vectors, dtype=np.float32)
        if len(matrix.shape) != 2:
            raise ValueError(f"vectors必须是二维数组[文本数, 维度]，实际shape={matrix.shape}")

        if matrix.shape[0] != len(chunks) or not np.isfinite(matrix).all():
            raise ValueError("向量数量必须与文本块一致，向量不能含 NaN/Inf")

        # 创建内积索引：embedder 输出已归一化，内积等价余弦相似度，值越大越相似
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

    def save(self, manifest=None, bm25_store=None):
        """
        发布一份完整索引：写新版本 → 回读验证 → 原子切换 current.json。
        :param manifest: 本次建库签名及配置摘要，用于下次判断能否复用
        :param bm25_store: 已按相同 chunk 顺序构建的 BM25Store；省略时在本地自动构建
        为什么不用先覆盖 index.faiss 再覆盖 chunks_meta.json？
        如果两次写入之间出错，会出现“新向量配旧文本”的错误引用。
        这里先把相关文件放进独立版本目录，全部通过验证后才切换一个指针文件。
        旧版本和首次升级前的根目录索引均保留，发布前失败不会影响它们。
        """
        if self.index is None or self.index.ntotal == 0:
            raise ValueError("拒绝发布空索引")
        if self.index.ntotal != len(self.chunks):
            raise ValueError("向量与元数据数量不一致")
        # 1. 每次发布分配独立版本目录，写入过程中不覆盖正在使用的索引。
        generation = uuid.uuid4().hex
        directory = self.persist_dir / "generations" / generation
        directory.mkdir(parents=True, exist_ok=False)
        index_path = directory / "index.faiss"
        faiss.write_index(self.index, str(index_path))
        atomic_json(directory / "chunks_meta.json", self.chunks)

        # BM25 与 FAISS 必须作为一代索引共同发布。BM25 写入或校验失败时，新目录可以保留
        # 用于诊断，但 current.json 尚未切换，因此线上仍读取上一代完整索引。
        from rag_agent.indexing.bm25_store import BM25Store
        bm25_store = bm25_store or BM25Store().build(self.chunks)
        if [c.get("metadata", {}).get("chunk_id") for c in bm25_store.chunks] != [
                c.get("metadata", {}).get("chunk_id") for c in self.chunks]:
            raise ValueError("BM25 与 FAISS 的 chunk 数量或 ID 顺序不一致")
        bm25_store.save(directory / "bm25")
        manifest = {**(manifest or {}), "bm25": bm25_store.info()}
        atomic_json(directory / "build_manifest.json", manifest)

        # 2. 从磁盘重新读取刚写入的两种索引，而不是只验证内存对象，检查保存结果是否完整。
        candidate = VectorStore(directory).load()
        if candidate.index.d != self.index.d or candidate.chunks != self.chunks:
            raise ValueError("新索引回读验证失败，保留旧索引")
        # 检查回读向量中的 NaN/Inf；这些非正常数值会让相似度计算失去意义。
        if not np.isfinite(candidate.index.reconstruct_n(0, candidate.index.ntotal)).all():
            raise ValueError("新索引包含无效向量，保留旧索引")
        BM25Store.load(directory / "bm25", candidate.chunks)
        # 3. 唯一发布点：此前任何步骤失败都不会改变当前版本指针。
        atomic_json(self.persist_dir / "current.json", {"generation": generation})
        self.index_path = index_path
        self.meta_path = directory / "chunks_meta.json"
        print(f"新索引已验证并发布: {generation}")

    def load(self):
        """
        读取当前活动版本；尚未升级时兼容根目录的旧索引。
        只读取一次 current.json 并固定版本目录，确保索引与元数据来自同一版本。
        :return: self，支持 VectorStore().load() 这样的链式调用
        """
        # current.json 像一本书的“当前版本书签”，里面只存版本标识，不存向量本身。
        pointer = self.persist_dir / "current.json"
        directory = self.persist_dir
        if pointer.exists():
            # 把版本名限定为随机生成的十六进制标识，避免无效路径指向版本目录之外。
            generation = read_json(pointer).get("generation", "")
            if not isinstance(generation, str) or not re.fullmatch(r"[0-9a-f]{32}", generation):
                raise ValueError("索引版本指针无效")
            directory = self.persist_dir / "generations" / generation
        self.index_path = directory / "index.faiss"
        self.meta_path = directory / "chunks_meta.json"
        if not (self.index_path.exists() and self.meta_path.exists()):
            raise FileNotFoundError("找不到完整索引，请先运行 python scripts/build_index.py")
        self.index = faiss.read_index(str(self.index_path))
        self.chunks = read_json(self.meta_path)
        if self.index.ntotal != len(self.chunks) or self.index.ntotal == 0:
            raise ValueError("索引数量与元数据不一致或为空")
        print(f"索引已加载，共 {self.index.ntotal} 个向量")
        return self


if __name__ == "__main__":
    # 旧的整库演示入口会实际计算向量并发布索引，不保存逐文档增量清单。
    # 日常运行请使用 scripts/build_index.py，避免绕过 builder 的增量管理。
    # 自测/建库入口：python -m rag_agent.indexing.faiss_store
    # 流程：读chunks.json → 批量向量化 → 入库 → 保存索引 → 检索自测

    # 1. 读取 chunker 导出的 chunk
    if not config.CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"找不到 {config.CHUNKS_FILE}\n"
            "请先运行 python -m rag_agent.indexing.chunker 生成chunk。"
        )
    with open(config.CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"读取到 {len(chunks)} 个chunk")

    # 2. 批量向量化（延迟导入：只在需要计算向量时加载 embedder 的模型，平时不背这个大依赖）
    from rag_agent.indexing import embedder
    emb = embedder.Embedding()
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
