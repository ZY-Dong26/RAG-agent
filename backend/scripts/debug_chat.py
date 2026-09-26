"""
debug_chat.py —— 带本地 HTML 诊断报告的 RAG 调试问答入口

职责：
    1. 与 chat.py 相同的问答流程，但每轮额外生成 JSON 轨迹和 HTML 诊断报告。
    2. 用 RecordingClient 代理 LLM 客户端，记录实际发送的 messages 和模型回答。
    3. 补充命中块相邻 segment，帮助定位检索或生成哪一步出了问题。

设计原因：
    - 不改 Generator/Retriever 核心代码；只在调用期间替换 client，用完恢复原值。
    - 报告输出到 data/outputs/debug/<时间戳>_<问题摘要>_<随机>/，多轮调试不互相覆盖。
    - --open 参数可每轮自动用浏览器打开最新报告，省去手动找文件。
"""

import argparse
import sys
import webbrowser
from pathlib import Path

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
    from rag_agent.qa.service import RAGService


EXIT_WORDS = {"exit", "quit", "q", "退出", "再见"}


def _ask(question, service, generator, open_report=False):
    """
    执行一轮调试问答：完整服务链路 → 记录 → 写诊断报告。

    输入：问题字符串、已加载的检索器和生成器、是否自动打开浏览器。
    输出：(回答文本, 命中块列表)。

    流程：
        1. 正常检索，记录命中块和耗时。
        2. 临时把 generator.client 换成 RecordingClient，调 generate。
        3. finally 中恢复原 client，加载相邻 chunk，写 JSON + HTML 报告。
    """
    recorder = TraceRecorder(question)
    original_client = generator.client
    try:
        generator.client = RecordingClient(original_client, recorder)
        result = service.ask(question)
        recorder.record_retrieval(result.hits, result.timing["retrieval_seconds"],
                                  recall_seconds=result.timing["recall_seconds"],
                                  rerank_seconds=result.timing["rerank_seconds"],
                                  total_seconds=result.timing["total_seconds"])
        if recorder.answer is None:
            recorder.answer = result.answer
    except Exception as error:
        if recorder.error is None:
            recorder.record_error(error)
        raise
    finally:
        generator.client = original_client
        hits = result.hits if "result" in locals() else []
        recorder.neighbors = load_neighbor_chunks(PROJECT_ROOT, hits, radius=1)
        _, html_path = write_trace_report(recorder, PROJECT_ROOT / "data/outputs/debug")
        print(f"\n诊断报告: {html_path.resolve()}")
        if open_report:
            webbrowser.open(html_path.resolve().as_uri())

    return result


def main():
    parser = argparse.ArgumentParser(description="带本地 HTML 诊断报告的 RAG 调试问答")
    parser.add_argument("--open", action="store_true", help="每轮问答后自动用默认浏览器打开报告")
    args = parser.parse_args()

    print("正在加载 Embedding、FAISS、BM25 索引和 LLM 客户端……")
    try:
        with stage("加载 FAISS 索引和本地 Embedding 模型"):
            retriever = Retriever()
        with stage("检查 LLM 配置并创建客户端"):
            generator = Generator()
        service = RAGService(retriever=retriever, generator=generator)
        service.prepare()
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
            result = _ask(question, service, generator, args.open)
            if not result.answerable:
                print(f"拒答: {result.answer}（{result.evidence_status}）\n")
                continue
            print(f"\n回答: {result.answer}\n\n引用来源:")
            for hit in result.hits:
                metadata = hit["metadata"]
                score = metadata.get("rerank_score")
                print(
                    f"  [{metadata.get('rank', '?')}] {metadata.get('source', '?')} "
                    f"第{metadata.get('page', '?')}页"
                    + (f" (重排分数{score:.4f})" if score is not None else " (按 RRF 排名)")
                )
            print()
        except Exception as error:
            print(f"[错误] {error}\n")


if __name__ == "__main__":
    main()
