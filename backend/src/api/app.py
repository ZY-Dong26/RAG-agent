"""
app.py —— FastAPI 应用入口

启动时建立本机聊天表、构造一次 RAGService，并在接收请求前加载索引和模型。
API 层处理 HTTP；问答编排位于 chat.service，SQLite 实现位于 storage.chat_history。
"""
import logging
from contextlib import asynccontextmanager
from threading import Lock

from fastapi import FastAPI

from rag_agent.storage.chat_history import ChatHistory, DEFAULT_HISTORY_PATH
from api.routes import chat, conversations, status

logger = logging.getLogger("api")


def _default_service_factory():
    """把耗时依赖延迟到服务启动阶段，导入模块和离线测试不会加载模型。"""
    from rag_agent.chat.service import RAGService
    return RAGService()


def create_app(service_factory=None, history_path=DEFAULT_HISTORY_PATH) -> FastAPI:
    """创建应用；测试可注入假服务和临时 SQLite 路径，不访问云端或正式数据。"""
    factory = service_factory or _default_service_factory

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """先建聊天表，再一次性准备模型；关闭时释放本地资源。"""
        try:
            history = ChatHistory(history_path)
            history.initialize()
            service = factory()
            service.prepare()
        except Exception as error:
            # 不输出异常内容：云端客户端异常可能携带签名地址或密钥。
            logger.error("RAG 服务初始化失败：%s；请检查索引、模型、密钥与设备", type(error).__name__)
            raise RuntimeError("RAG 服务初始化失败；请检查后端配置和日志") from None
        app.state.service = service
        app.state.history = history
        app.state.ready = True
        try:
            yield
        finally:
            app.state.ready = False
            app.state.service = None
            app.state.history = None

    app = FastAPI(title="RAG Agent API", lifespan=lifespan)
    app.state.service = None
    app.state.history = None
    app.state.ready = False
    app.state.ask_lock = Lock()
    app.include_router(status.router, prefix="/api/v1")
    app.include_router(chat.router, prefix="/api/v1")
    app.include_router(conversations.router, prefix="/api/v1")
    return app


app = create_app()
