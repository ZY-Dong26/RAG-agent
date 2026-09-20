"""
chat.py —— 命令行问答入口（常驻 REPL）

职责：启动时把 embedding模型 + FAISS索引 加载进内存（整个会话只做一次），
     然后循环接收用户问题：检索 → 云端LLM生成 → 打印回答与引用来源。
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

from rag_agent.qa.generator import Generator
from rag_agent.qa.retriever import Retriever
from rag_agent.qa.service import RAGService

EXIT_WORDS = {"exit", "quit", "q", "退出", "再见"}


def main():
    # 1. 一次性加载：模型 + 索引（最耗时的部分，整个会话只做一次）
    print("正在加载 embedding模型 + FAISS索引（首次约几秒，请稍等）...")
    ret = Retriever()

    # 2. 创建云端LLM客户端（API Key 缺失时在这里给出明确指引）
    try:
        gen = Generator()
    except RuntimeError as e:
        print(f"\n[错误] {e}")
        sys.exit(1)

    service = RAGService(retriever=ret, generator=gen)
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
            # 检索 → 生成（模型和索引已在内存，每问一次只算向量不重载）
            result = service.ask(question)
            hits = result.hits
            if not hits:
                print("向量库中没有找到相关内容。\n")
                continue

            reply = result.answer

            # 打印回答 + 引用来源（可溯源）
            print(f"\n回答: {reply}\n")
            print("引用来源:")
            for h in hits:
                m = h["metadata"]
                print(f"  [{m.get('rank', '?')}] {m.get('source', '?')} "
                      f"第{m.get('page', '?')}页 (相似度{m['score']:.4f})")
            print()
        except Exception as e:
            print(f"[错误] {e}\n")


if __name__ == "__main__":
    main()
