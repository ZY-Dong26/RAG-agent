"""
build_index.py —— 建库启动入口（离线，支持增量跳过）

职责：把 src/ 下的业务模块按顺序串起来，一条命令建库：
     PDF → chunk → 向量 → FAISS 索引
     01_document_loader → 02_text_splitter → 03_embedding → 04_vector_store

增量设计（解决"每次都要重新切分建库"的问题）：
     阶段1（切分）：chunks.json 已存在 且 没有比它更新的PDF → 跳过
     阶段2（向量化）：index.faiss 已存在 且 不旧于 chunks.json → 跳过
     加 --force 强制全量重建（比如改了切分参数、或怀疑索引不一致时）

运行（在项目根目录）：
     python scripts/build_index.py           # 增量：能跳过的都跳过
     python scripts/build_index.py --force   # 强制全量重建

注意：这里用文件时间戳做判断，足够覆盖日常场景；
     生产级做法是记录每个PDF的哈希，只重处理真正变化的文件。
"""
import argparse
import json
import sys
from pathlib import Path

# 关键点：scripts/ 不是包，直接运行时 sys.path[0] 是 scripts/ 目录，
# 必须手动把项目根目录加进导入路径，才能 from src import xxx
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, document_loader, text_splitter


def _pdfs_newer_than_chunks():
    """是否需要重新切分：chunks.json 不存在，或存在比它更新的PDF"""
    if not config.CHUNKS_FILE.exists():
        return True
    chunks_mtime = config.CHUNKS_FILE.stat().st_mtime
    return any(
        p.stat().st_mtime > chunks_mtime
        for p in config.RAW_PDF_DIR.glob("*.pdf")
    )


def _index_up_to_date():
    """是否需要重新向量化：索引两个文件都存在，且不旧于 chunks.json"""
    idx = config.VECTOR_DB_DIR / "index.faiss"
    meta = config.VECTOR_DB_DIR / "chunks_meta.json"
    if not (idx.exists() and meta.exists()):
        return False
    return idx.stat().st_mtime >= config.CHUNKS_FILE.stat().st_mtime


def build_index(force=False):
    """
    增量建库：只重跑"变了的那一段"，没变的直接复用中间产物
    :param force: True 时忽略中间产物，强制全量重建
    :return: chunks 列表（切分产物，供调试用）
    """
    chunks = None

    # ===== 阶段1：PDF → chunk（可跳过）=====
    if force or _pdfs_newer_than_chunks():
        docs = document_loader.load_all_pdfs(config.RAW_PDF_DIR)
        splitter = text_splitter.TextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
        )
        chunks = splitter.split_documents(docs)
        text_splitter.TextSplitter.save_chunks(chunks, config.CHUNKS_FILE)
        print(f"阶段1完成: {len(chunks)} 个chunk → {config.CHUNKS_FILE}")
    else:
        print(f"阶段1跳过: {config.CHUNKS_FILE} 已存在且没有更新的PDF")

    # ===== 阶段2：chunk → 向量 → 索引（可跳过）=====
    if force or not _index_up_to_date():
        # 阶段1被跳过时，chunks 从落盘文件读取，不重新解析PDF
        if chunks is None:
            with open(config.CHUNKS_FILE, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            print(f"读取已有chunk: {len(chunks)} 条")

        # 延迟导入：只有真要向量化时才加载本地大模型（torch）
        from src import embedding, vector_store

        emb = embedding.Embedding()
        print(f"正在向量化 {len(chunks)} 个chunk（device={emb.device}，请稍等）...")
        vectors = emb.embed_texts([c["text"] for c in chunks])

        store = vector_store.VectorStore()
        store.add(chunks, vectors)
        store.save()
        print("\n=====建库完成=====")
    else:
        print(f"阶段2跳过: {config.VECTOR_DB_DIR} 的索引已是最新（--force 可强制重建）")

    return chunks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG 离线建库入口（增量）")
    parser.add_argument(
        "--force", action="store_true",
        help="强制全量重建，忽略已有中间产物"
    )
    args = parser.parse_args()
    build_index(force=args.force)
