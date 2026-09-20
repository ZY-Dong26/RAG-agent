"""
build_index.py —— 增量建库命令行入口

默认策略：
    允许部分成功。新文档失败时跳过；已入库文档更新失败时继续使用旧版。
    未变化文档复用按文档保存的向量，只给新增或变化文档计算 Embedding。

可选策略：
    --strict         要求 data/raw 中每份 PDF 都成功处理到最新版本，否则不发布。
    --force          强制重新切块和向量化，仍复用合格的 MinerU 解析缓存。
    --prune-missing  从候选索引移除已经不在 data/raw 的旧文档。

真正的业务流程位于 src/rag_agent/indexing/builder.py；入口保持精简，便于分别学习命令行层和业务层。
"""
import argparse
import sys
from pathlib import Path

# 直接运行 scripts 下的文件时，先把项目根目录加入模块搜索路径。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_agent.indexing import builder


def parse_documents(paths=None, resubmit=False):
    """兼容解析调试入口：只解析指定 PDF，不建立向量索引。"""
    return builder.parse_documents(paths, resubmit=resubmit)


def build_index(force=False, strict=False, prune_missing=False):
    """把命令行参数转交给按文档增量建库业务层。"""
    return builder.build_index(force=force, strict=strict, prune_missing=prune_missing)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MinerU 云端解析 + 按文档增量向量建库")
    parser.add_argument("--force", action="store_true",
                        help="强制重新切块和向量化所有当前文档，仍复用 MinerU 解析缓存")
    parser.add_argument("--strict", action="store_true",
                        help="任何当前 PDF 处理失败时都不发布候选索引")
    parser.add_argument("--prune-missing", action="store_true",
                        help="从候选索引移除已不在 data/raw 的旧文档")
    args = parser.parse_args()
    try:
        build_index(force=args.force, strict=args.strict, prune_missing=args.prune_missing)
    except (RuntimeError, ValueError, OSError) as error:
        print(f"建库失败: {error}", file=sys.stderr)
        sys.exit(1)
