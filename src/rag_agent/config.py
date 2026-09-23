"""
config.py —— 全局配置：所有路径、参数、模型名集中在这里管理

原则：
    1. 业务模块（ingestion/indexing 等）不直接写路径和参数，
       统一从本模块取值；以后改目录结构、改参数，只动这一个文件。
    2. 密钥读取自根目录 .env 或系统环境变量；系统环境变量优先，不硬编码。
"""
import os
from pathlib import Path

# LLM 与 MinerU 共用根目录 .env，以 LLM_* / MINERU_* 区分配置，不注入环境变量。
from dotenv import dotenv_values
_llm_env = dotenv_values(Path(__file__).resolve().parents[2] / ".env")


def _llm_setting(name, default):
    """
    只取调用者指定的 LLM 或 Embedding 配置项；系统环境变量优先，不修改进程环境。
    """
    return os.environ.get(name, _llm_env.get(name) or default)


def _model_path(value, default):
    """把模型配置解析为绝对路径；相对路径统一以项目根目录为基准。"""
    path = Path(value or default).expanduser()
    return str(path if path.is_absolute() else BASE_DIR / path)


# 项目根目录：本文件位于 src/rag_agent/，向上三级回到项目根目录
BASE_DIR = Path(__file__).resolve().parents[2]

# ========== 数据目录 ==========
RAW_PDF_DIR = BASE_DIR / "data" / "raw"            # 原始PDF输入（放这里）
PROCESSED_DIR = BASE_DIR / "data" / "processed"    # 中间产物目录（chunks.json等）
CHUNKS_FILE = PROCESSED_DIR / "chunks.json"        # 切分结果文件

VECTOR_DB_DIR = BASE_DIR / "data" / "vector_db"             # 向量库持久化目录（FAISS索引+chunk元信息）

# ========== 文本切分参数 ==========
CHUNK_SIZE = 800        # 每个文本块最大字符数
CHUNK_OVERLAP = 150     # 相邻文本块重叠字符数（保留上下文）

# ========== 检索参数 ==========
# 两路召回的原始分数不在同一量纲，先各自取候选，再用排名倒数 RRF 融合。
DENSE_TOP_K = 30
BM25_TOP_K = 30
FUSION_TOP_K = 20
RRF_K = 60
RRF_DENSE_WEIGHT = 1.0
RRF_BM25_WEIGHT = 1.0

# BGE 重排只处理融合后的少量候选。拒答阈值必须通过项目自己的评测集校准；
# None 表示尚未校准，因此默认不会仅凭重排分数硬拒答。
RERANK_TOP_K = 5
RERANK_MAX_LENGTH = 1024
RERANK_BATCH_SIZE = 8
RERANK_REJECT_THRESHOLD = None
RERANK_ENABLED = _llm_setting("RERANK_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}

# 保留旧名称供评测脚本和外部调用使用；它现在表示最终交给生成器的证据数量。
TOP_K = RERANK_TOP_K


# ========== 本地Embedding模型配置 ==========
# 本地模型路径：相对路径以项目根目录为基准，也支持环境变量指定绝对路径
EMBEDDING_MODEL = _model_path(_llm_setting("EMBEDDING_MODEL", ""), "model/Qwen3-Embedding-0.6B")
EMBEDDING_MAX_LENGTH = 8192   # Qwen3-Embedding支持最长32k，取8192平衡速度

# 本地重排模型只从磁盘加载，不允许 transformers 自动联网下载。
RERANKER_MODEL = _model_path(_llm_setting("RERANKER_MODEL", ""), "model/bge-reranker-v2-m3")


# ========== 云端LLM配置（OpenAI兼容协议） ==========
# 约定：所有主流云端大模型都提供OpenAI兼容接口，换厂商只改下面三个变量：
#   DeepSeek    : LLM_BASE_URL=https://api.deepseek.com/v1   LLM_MODEL=deepseek-chat
#   通义千问    : LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1  LLM_MODEL=qwen-plus
#   月之暗面Kimi: LLM_BASE_URL=https://api.moonshot.cn/v1    LLM_MODEL=moonshot-v1-8k
#   OpenAI      : LLM_BASE_URL=https://api.openai.com/v1     LLM_MODEL=gpt-4o-mini
LLM_BASE_URL = _llm_setting("LLM_BASE_URL", "https://api.deepseek.com/v1")
LLM_API_KEY = _llm_setting("LLM_API_KEY", "")          # 在项目根目录 .env 里填：LLM_API_KEY=sk-xxx
LLM_MODEL = _llm_setting("LLM_MODEL", "deepseek-chat")
LLM_TEMPERATURE = 0.3       # 低温度：事实问答更稳，减少编造
LLM_MAX_TOKENS = 1024       # 单次回答最大输出token
MAX_CONTEXT_CHARS = 4000    # 拼给LLM的资料字符预算（不等同于 Token 上限，不含系统提示词和问题）
