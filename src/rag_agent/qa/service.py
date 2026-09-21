"""
service.py —— 命令行与未来前端共用的问答应用服务

前端不应直接操作 FAISS、Embedding 或 LLM 客户端，只调用 RAGService.ask()。
这样以后增加 FastAPI、桌面界面或网页时，检索和生成规则仍然只有一份实现。
"""
from dataclasses import dataclass
from rag_agent.common.progress import report, stage

from rag_agent.qa.generator import Generator
from rag_agent.qa.retriever import Retriever


@dataclass(frozen=True)
class AnswerResult:
    """一次问答的统一返回值；answer 为 None 表示向量库没有召回资料。"""
    answer: str | None
    hits: list


class RAGService:
    """编排“检索 → 生成”，不负责终端输入输出，因而可以被多种界面复用。"""

    def __init__(self, retriever=None, generator=None):
        # 允许外部传入替身对象，便于测试；正常运行时各依赖只创建一次。
        """保存注入的检索器和生成器；未传入时创建默认依赖，供后续各轮问答复用。"""
        self.retriever = retriever or Retriever()
        self.generator = generator or Generator()

    def ask(self, question):
        """检索与问题相关的资料；有命中时调用 LLM，返回答案和完整引用记录。"""
        # 每次问答独立处理，不保存多轮聊天历史；这里也没有相似度阈值或重排序。
        with stage("问题向量化与资料检索"):
            hits = self.retriever.retrieve(question)
        report(f"[检索] 找到 {len(hits)} 条候选资料")
        if not hits:
            return AnswerResult(answer=None, hits=[])
        with stage("等待云端 LLM 生成回答"):
            answer = self.generator.generate(question, hits)
        return AnswerResult(answer=answer, hits=hits)
