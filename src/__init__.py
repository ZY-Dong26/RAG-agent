"""
src/__init__.py —— 包初始化：数字前缀文件名的"懒加载映射"

为什么需要这段代码？
    src 下的文件按流水线顺序编号（00_config.py、01_document_loader.py …），
    目录里一眼就能看出执行顺序，方便阅读。
    但 Python 语法规定：模块名必须以字母或下划线开头，
    数字开头的文件名（如 00_config.py）无法用 import 语句直接导入。

    这里用 PEP 562 的 __getattr__ 做懒加载映射：
        磁盘文件  src/00_config.py           ↔  导入名  src.config
        磁盘文件  src/01_document_loader.py  ↔  导入名  src.document_loader

    注意：数字前缀文件不能用点号导入（from src.config import ... 会报错），
    统一写法是 from src import xxx，然后 xxx.名字 使用。
    例如：from src import config; config.RAW_PDF_DIR

说明：07_rag_pipeline.py 已移除，启动入口移到 scripts/：
      scripts/build_index.py 负责建库，scripts/chat.py 负责命令行问答。
"""
import importlib

# 业务模块名 → 数字前缀文件名（按流水线顺序）
_ALIASES = {
    "config": "src.00_config",                    # 全局配置最先加载
    "document_loader": "src.01_document_loader",  # 1.加载原始文档
    "text_splitter": "src.02_text_splitter",      # 2.文档切分chunk
    "embedding": "src.03_embedding",              # 3.文本向量化
    "vector_store": "src.04_vector_store",        # 4.向量入库
    "retriever": "src.05_retriever",              # 5.检索模块（问答阶段召回）
    "generator": "src.06_generator",              # 6.LLM生成回答
}


def __getattr__(name):
    """懒加载：只有真正用到时才导入对应文件（from src import xxx 会触发）"""
    if name not in _ALIASES:
        raise AttributeError(f"module 'src' has no attribute '{name}'")
    return importlib.import_module(_ALIASES[name])
