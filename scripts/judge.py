"""
scripts/judge.py —— 用独立裁判模型对已有评测批次自动判分

职责：
    1. 接收一个已有评测批次目录，逐题调用裁判模型打分。
    2. 不加载检索器、不调用回答模型，只消费 items/*.json 中已有的题目和回答。
    3. 支持 --limit 少量试跑、--force 全部重判；中断后直接重跑同一命令即可续跑。
    4. 判分结束后自动刷新 CSV、报告和 judge_summary.md，并在终端打印汇总表。

用法：
    python scripts/judge.py data/outputs/evaluation/<批次目录>
    python scripts/judge.py data/outputs/evaluation/<批次目录> --limit 3
    python scripts/judge.py data/outputs/evaluation/<批次目录> --force
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rag_agent.common.progress import configure_progress, stage


def main():
    configure_progress()
    parser = argparse.ArgumentParser(description="用独立 LLM 对已有 RAG 评测批次自动判分")
    parser.add_argument("batch", type=Path, help="已有评测批次目录")
    parser.add_argument("--limit", type=int, help="本次最多新判多少题，适合少量试跑")
    parser.add_argument("--force", action="store_true", help="重新判所有题，会再次产生裁判模型费用")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("limit 必须大于零")

    # 延迟导入：只在真正判分时才加载 openai 和裁判配置，减少 --help 时的依赖。
    from openai import OpenAI
    from devtools.evaluation.judge_config import (
        JUDGE_LLM_API_KEY,
        JUDGE_LLM_BASE_URL,
        JUDGE_LLM_MAX_TOKENS,
        JUDGE_LLM_MODEL,
        JUDGE_LLM_TEMPERATURE,
    )
    from devtools.evaluation.judger import judge_directory
    missing = [name for name, value in (("JUDGE_LLM_API_KEY", JUDGE_LLM_API_KEY),
                                        ("JUDGE_LLM_BASE_URL", JUDGE_LLM_BASE_URL),
                                        ("JUDGE_LLM_MODEL", JUDGE_LLM_MODEL)) if not value]
    if missing:
        raise RuntimeError("裁判模型配置缺失：" + "、".join(missing))
    with stage("创建独立裁判模型客户端"):
        client = OpenAI(api_key=JUDGE_LLM_API_KEY, base_url=JUDGE_LLM_BASE_URL)
    result = judge_directory(args.batch, client, JUDGE_LLM_MODEL, JUDGE_LLM_BASE_URL,
                             temperature=JUDGE_LLM_TEMPERATURE, max_tokens=JUDGE_LLM_MAX_TOKENS,
                             limit=args.limit, force=args.force)
    if result["failed"]:
        raise RuntimeError(f"{result['failed']} 道题判分失败；已成功题目已保存，可直接重跑续判")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[中断] 已完成判分均已保存；用同一批次目录重跑即可续跑。当前请求可能已计费。", flush=True)
        sys.exit(130)
    except (ValueError, OSError, RuntimeError) as error:
        print(f"[判分未启动或未完成] {type(error).__name__}；检查批次目录和裁判模型配置。", flush=True)
        sys.exit(1)
