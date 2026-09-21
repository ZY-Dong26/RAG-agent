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


# 项目根目录：本文件位于 src/rag_agent/，向上三级回到项目根目录
BASE_DIR = Path(__file__).resolve().parents[2]

# ========== 数据目录 ==========
RAW_PDF_DIR = BASE_DIR / "data" / "raw"            # 原始PDF输入（放这里）
PROCESSED_DIR = BASE_DIR / "data" / "processed"    # 中间产物目录（chunks.json等）
CHUNKS_FILE = PROCESSED_DIR / "chunks.json"        # 切分结果文件

VECTOR_DB_DIR = BASE_DIR / "vector_db"             # 向量库持久化目录（FAISS索引+chunk元信息）

# ========== 文本切分参数 ==========
CHUNK_SIZE = 800        # 每个文本块最大字符数
CHUNK_OVERLAP = 150     # 相邻文本块重叠字符数（保留上下文）

# ========== 检索参数 ==========
TOP_K = 5               # 检索时返回的候选chunk数量


# ========== 本地Embedding模型配置 ==========
# 本地模型路径：相对路径以项目根目录为基准，也支持环境变量指定绝对路径
_embedding_path = Path(_llm_setting("EMBEDDING_MODEL", "") or "model/Qwen3-Embedding-0.6B").expanduser()
EMBEDDING_MODEL = str(_embedding_path if _embedding_path.is_absolute() else BASE_DIR / _embedding_path)
EMBEDDING_MAX_LENGTH = 8192   # Qwen3-Embedding支持最长32k，取8192平衡速度


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
