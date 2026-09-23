"""
judge_config.py —— 裁判模型的独立配置声明

职责：
    1. 从项目根目录 .env 读取裁判模型的连接信息（base_url、api_key、model）。
    2. 声明判分策略参数（temperature、max_tokens），固定为代码常量。

设计原因：
    - 裁判模型只在 LLM 自动判分时使用，不属于问答运行时链路，因此不放在 rag_agent/config.py，
      避免核心包感知开发工具。
    - 连接信息（地址、密钥、模型名）走 .env，便于切换不同裁判模型；系统环境变量优先于项目 .env。
    - temperature 和 max_tokens 是判分策略而非环境差异，固定写在代码里；想试不同温度直接改这两行。
"""
import os
from pathlib import Path

from dotenv import dotenv_values

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_ENV = dotenv_values(_PROJECT_ROOT / ".env")

# 服务连接信息允许系统环境变量覆盖项目 `.env`，便于临时切换裁判模型。
JUDGE_LLM_API_KEY = os.environ.get("JUDGE_LLM_API_KEY") or _ENV.get("JUDGE_LLM_API_KEY", "")
JUDGE_LLM_BASE_URL = os.environ.get("JUDGE_LLM_BASE_URL") or _ENV.get("JUDGE_LLM_BASE_URL", "")
JUDGE_LLM_MODEL = os.environ.get("JUDGE_LLM_MODEL") or _ENV.get("JUDGE_LLM_MODEL", "")

# 判分策略参数：temperature 固定 0 保证分数稳定可复现；max_tokens 覆盖三维度分数加理由的常规长度。
JUDGE_LLM_TEMPERATURE = 0
JUDGE_LLM_MAX_TOKENS = 800
