"""
scripts/export_results.py —— 从已有评测批次纯本地重建全部导出产物

职责：
    1. 读取批次目录下 items/*.json，重新生成 summary.json、results.jsonl、results.csv、report.md、judge_summary.md。
    2. 不加载 Embedding、不调用回答模型或裁判模型，纯本地读 JSON 写文件。

使用场景：
    - 改了导出格式（CSV 列、Markdown 模板）后，对老批次零成本重建产物。
    - 只想看最新判分结果，不想重新跑 84 题评测。
    - 判分中断后手动刷新一次产物。

用法：
    python scripts/export_results.py data/outputs/evaluation/20260923-143550-697003
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from devtools.evaluation.exporter import export_results, load_items
from rag_agent.common.progress import configure_progress, report, stage
from rag_agent.common.files import exclusive_lock


def main():
    configure_progress()
    parser = argparse.ArgumentParser(description="从 items/*.json 重新导出评测产物；不调用任何模型")
    parser.add_argument("batch", type=Path, help="已有评测批次目录")
    args = parser.parse_args()
    # 与 evaluate.py / judge.py 共用批次锁，避免导出时另一进程正在写 items。
    with exclusive_lock(args.batch / ".evaluation.lock"):
        with stage("读取逐题记录并重新导出本地产物"):
            rows = load_items(args.batch)
            summary = export_results(args.batch, rows)
    report(f"[导出完成] {summary['attempted']} 题；目录：{args.batch}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as error:
        print(f"[导出失败] {error}", flush=True)
        sys.exit(1)
