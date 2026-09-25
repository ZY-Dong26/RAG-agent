"""
parse_documents.py —— 解析调试入口：只完成 RAG 流程的“文档解析”阶段
适用场景：先检查扫描件的 OCR、表格和页码，再决定是否继续建立向量库。
关键设计：
    1. --file 指定一份 PDF；不指定时处理 data/raw 下全部 PDF。
    2. 普通重跑会恢复任务或复用缓存；--resubmit 才显式创建新任务。
    3. 重新提交必须指定单个文件，避免意外重复提交整个文档库。
    4. 本入口不加载 Embedding，不更新 FAISS，也不调用回答生成模型。
"""
import argparse
import sys
from pathlib import Path
# 直接运行 scripts 下的文件时，Python 默认从 scripts 找模块；这里加入项目的 src 目录，以便导入 rag_agent 包。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rag_agent.common.progress import configure_progress, report, stage

# 先输出启动提示，再导入可能耗时的依赖，避免运行窗口长时间空白。
if __name__ == "__main__":
    configure_progress()

with stage("加载运行依赖"):
    from rag_agent import config
    from rag_agent.indexing.builder import parse_documents



def main():
    """
    解析命令行参数并执行单独解析；返回 0 表示成功，1 表示本次解析未全部成功。
    """
    parser = argparse.ArgumentParser(description="只解析 PDF 并缓存结果，不更新索引")
    parser.add_argument("--file", type=Path, help="仅处理指定 PDF；省略则处理 data/raw 下全部 PDF")
    parser.add_argument("--resubmit", action="store_true",
                        help="已人工核对任务后，显式创建新任务；必须同时指定 --file")
    # 把命令行文字转换为 args.file、args.resubmit 等属性；--file 会直接转为 Path 对象。
    args = parser.parse_args()
    # 普通恢复无需 --resubmit；显式重提可能产生新任务，所以限制为单份文件。
    if args.resubmit and args.file is None:
        parser.error("--resubmit 必须配合 --file，避免批量重复提交")
    if args.file and (not args.file.is_file() or args.file.suffix.lower() != ".pdf"):
        parser.error("--file 必须指向存在的 PDF")
    try:
        # 直接调用核心包的解析服务，避免启动脚本之间互相导入并产生路径依赖。
        report("[解析] " + (str(args.file.name) if args.file else "检查 data/raw 中全部 PDF"))
        with stage("文档解析与验收；结果见 parse_report.json"):
            parse_documents([args.file] if args.file else None, resubmit=args.resubmit)
    except (RuntimeError, ValueError, OSError) as error:
        print(f"解析失败: {error}", file=sys.stderr)
        # 非零退出码可让终端脚本或自动化任务知道本次失败，而不是误判“命令运行完就算成功”。
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
