"""
app.py —— FastAPI 应用入口

启动时只构造一次 RAGService，并在接收请求前加载索引、Embedding 和重排器。
路由仅转换 HTTP 数据；命令行入口与 API 均调用 qa.service，不复制检索或生成规则。
"""
import logging
from contextlib import asynccontextmanager
from threading import Lock

from fastapi import FastAPI

from api.routes import chat, status

logger = logging.getLogger("api")


def _default_service_factory():
    """把耗时依赖延迟到服务启动阶段，导入模块和离线测试不会加载模型。"""
    from rag_agent.qa.service import RAGService
    return RAGService()


def create_app(service_factory=None) -> FastAPI:
    """创建应用；测试可注入假服务，不加载真实模型或访问云端。"""
    factory = service_factory or _default_service_factory

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """一次性准备模型；关闭时移除引用，让进程释放本地资源。"""
        try:
            service = factory()
            service.prepare()
        except Exception as error:
            # 不输出异常内容：云端客户端异常可能携带签名地址或密钥。
            logger.error("RAG 服务初始化失败：%s；请检查索引、模型、密钥与设备", type(error).__name__)
            raise RuntimeError("RAG 服务初始化失败；请检查后端配置和日志") from None
        app.state.service = service
        app.state.ready = True
        try:
            yield
        finally:
            app.state.ready = False
            app.state.service = None

    app = FastAPI(title="RAG Agent API", lifespan=lifespan)
    app.state.service = None
    app.state.ready = False
    app.state.ask_lock = Lock()
    app.include_router(status.router, prefix="/api/v1")
    app.include_router(chat.router, prefix="/api/v1")
    return app


app = create_app()
