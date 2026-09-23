"""开发期 RAG 问答入口：不改核心流程，额外生成每轮 JSON 与 HTML 诊断报告。"""

import argparse
import sys
import webbrowser
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from devtools.rag_inspector import RecordingClient, TraceRecorder, load_neighbor_chunks, write_trace_report
from rag_agent.common.progress import configure_progress, report, stage

if __name__ == "__main__":
    configure_progress()

with stage("加载运行依赖"):
    from rag_agent.qa.generator import Generator
    from rag_agent.qa.retriever import Retriever


EXIT_WORDS = {"exit", "quit", "q", "退出", "再见"}


def _ask(question, retriever, generator, open_report=False):
    recorder = TraceRecorder(question)
    started = perf_counter()
    with stage("问题向量化与资料检索"):
        hits = retriever.retrieve(question)
    recorder.record_retrieval(hits, perf_counter() - started)
    report(f"[检索] 找到 {len(hits)} 条候选资料")

    answer = None
    original_client = generator.client
    try:
        if hits:
            generator.client = RecordingClient(original_client, recorder)
            with stage("等待云端 LLM 生成回答"):
                answer = generator.generate(question, hits)
    except Exception as error:
        if recorder.error is None:
            recorder.record_error(error)
        raise
    finally:
        generator.client = original_client
        recorder.neighbors = load_neighbor_chunks(PROJECT_ROOT, hits, radius=1)
        _, html_path = write_trace_report(recorder, PROJECT_ROOT / "data/outputs/debug")
        print(f"\n诊断报告: {html_path.resolve()}")
        if open_report:
            webbrowser.open(html_path.resolve().as_uri())

    return answer, hits


def main():
    parser = argparse.ArgumentParser(description="带本地 HTML 诊断报告的 RAG 调试问答")
    parser.add_argument("--open", action="store_true", help="每轮问答后自动用默认浏览器打开报告")
    args = parser.parse_args()

    print("正在加载 embedding 模型、FAISS 索引和 LLM 客户端……")
    try:
        with stage("加载 FAISS 索引和本地 Embedding 模型"):
            retriever = Retriever()
        with stage("检查 LLM 配置并创建客户端"):
            generator = Generator()
    except (RuntimeError, ValueError, OSError) as error:
        print(f"[启动失败] {error}")
        return

    print("\nRAG 调试问答已就绪。每轮都会生成本地诊断报告，输入 exit 退出。\n")
    while True:
        try:
            question = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见")
            break
        if not question:
            continue
        if question.lower() in EXIT_WORDS:
            print("再见")
            break
        try:
            answer, hits = _ask(question, retriever, generator, args.open)
            if not hits:
                print("向量库中没有找到相关内容。\n")
                continue
            print(f"\n回答: {answer}\n\n引用来源:")
            for hit in hits:
                metadata = hit["metadata"]
                print(
                    f"  [{metadata.get('rank', '?')}] {metadata.get('source', '?')} "
                    f"第{metadata.get('page', '?')}页 (相似度{metadata['score']:.4f})"
                )
            print()
        except Exception as error:
            print(f"[错误] {error}\n")


if __name__ == "__main__":
    main()
