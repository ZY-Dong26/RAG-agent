"""
config.py —— 全局配置：所有路径、参数、模型名集中在这里管理

原则：
    1. 业务模块（document_loader/text_splitter等）不直接写路径和参数，
       统一从本模块取值；以后改目录结构、改参数，只动这一个文件。
    2. 敏感信息（API Key）一律从环境变量读取，不硬编码在代码里。
"""
import os
from pathlib import Path

# 尝试加载项目根目录的 .env 文件（保存API Key等敏感信息）。
# 没有安装 python-dotenv 时跳过，不影响其他功能。
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

# 项目根目录：本文件位于 src/ 下，父目录的父目录即项目根
BASE_DIR = Path(__file__).resolve().parent.parent

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
_embedding_path = Path(os.getenv("EMBEDDING_MODEL") or "model/Qwen3-Embedding-0.6B").expanduser()
EMBEDDING_MODEL = str(_embedding_path if _embedding_path.is_absolute() else BASE_DIR / _embedding_path)
EMBEDDING_MAX_LENGTH = 8192   # Qwen3-Embedding支持最长32k，取8192平衡速度


# ========== 云端LLM配置（OpenAI兼容协议） ==========
# 约定：所有主流云端大模型都提供OpenAI兼容接口，换厂商只改下面三个变量：
#   DeepSeek    : LLM_BASE_URL=https://api.deepseek.com/v1   LLM_MODEL=deepseek-chat
#   通义千问    : LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1  LLM_MODEL=qwen-plus
#   月之暗面Kimi: LLM_BASE_URL=https://api.moonshot.cn/v1    LLM_MODEL=moonshot-v1-8k
#   OpenAI      : LLM_BASE_URL=https://api.openai.com/v1     LLM_MODEL=gpt-4o-mini
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")          # 在项目根目录 .env 里填：LLM_API_KEY=sk-xxx
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
LLM_TEMPERATURE = 0.3       # 低温度：事实问答更稳，减少编造
LLM_MAX_TOKENS = 1024       # 单次回答最大输出token
MAX_CONTEXT_CHARS = 4000    # 拼给LLM的资料字符上限（防止超出模型窗口）
