"""
chat.py —— 命令行问答入口（常驻 REPL）

职责：启动时加载混合索引和本地模型（整个会话只做一次），然后循环接收用户问题：
     Dense + BM25 → RRF → BGE 重排 → 证据门控 → 云端 LLM → 展示最终引用。
·
运行（在项目根目录）：
     python scripts/chat.py

退出：输入 exit / quit / q / 退出，或按 Ctrl+C
前置：先跑过 python scripts/build_index.py 建库；
     在项目根目录 .env 里配好 LLM_API_KEY（否则启动时会给提示）。
"""
import sys
from pathlib import Path

# scripts 只负责终端交互；把 src 加入路径后调用 rag_agent 的公共问答服务
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_agent.common.progress import configure_progress, report, stage

# 先输出启动提示，再导入可能耗时的依赖，避免运行窗口长时间空白。
if __name__ == "__main__":
    configure_progress()

with stage("加载运行依赖"):
    from rag_agent.qa.generator import Generator
    from rag_agent.qa.retriever import Retriever
    from rag_agent.qa.service import RAGService


EXIT_WORDS = {"exit", "quit", "q", "退出", "再见"}


def main():
    # 1. 一次性加载：模型 + 索引（最耗时的部分，整个会话只做一次）
    """启动终端问答：加载依赖一次，逐轮调用服务；退出词或 Ctrl+C 结束会话。"""
    print("正在加载 Embedding、FAISS 和 BM25 索引（首次约几秒，请稍等）...")
    try:
        with stage("加载 FAISS、BM25 索引和本地 Embedding 模型"):
            ret = Retriever()
    except (RuntimeError, ValueError, OSError) as error:
        print(f"[启动失败] {error}；请先运行 scripts/build_index.py", flush=True)
        return

    # 2. 创建云端LLM客户端（API Key 缺失时在这里给出明确指引）
    try:
        with stage("检查 LLM 配置并创建客户端"):
            gen = Generator()
    except RuntimeError as e:
        print(f"\n[错误] {e}")
        sys.exit(1)

    service = RAGService(retriever=ret, generator=gen)
    try:
        service.prepare()
    except (RuntimeError, ValueError, OSError):
        print("[启动失败] 本地重排模型加载或预热失败；请检查模型目录、设备和显存。", flush=True)
        return
    print("\nRAG 命令行问答已就绪。输入 exit 退出。\n")

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
            # 完整链路由服务层统一编排；脚本只负责终端输入输出。
            result = service.ask(question)
            hits = result.hits
            if not result.answerable:
                if result.evidence_status == "no_candidates":
                    print(f"\n拒答: {result.answer}（没有检索候选）\n")
                else:
                    print(f"\n拒答: {result.answer}（相关性不足）\n")
                continue

            reply = result.answer
            if not result.threshold_calibrated:
                print("\n提示: 重排拒答阈值尚未通过评测集校准，本轮未执行分数硬拒答。")

            # 打印回答 + 引用来源（可溯源）
            print(f"\n回答: {reply}\n")
            print("引用来源:")
            for h in hits:
                m = h["metadata"]
                rerank = m.get("rerank_score")
                score_text = f"重排分数{rerank:.4f}" if rerank is not None else "按 RRF 排名"
                print(f"  [{m.get('rank', '?')}] {m.get('source', '?')} "
                      f"第{m.get('page', '?')}页 ({score_text})")
            print()
        except Exception as e:
            print(f"[错误] {e}\n")


if __name__ == "__main__":
    main()
