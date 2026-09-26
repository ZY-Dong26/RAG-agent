"""status.py —— 返回已加载索引和重排器的只读状态。"""
from pathlib import Path

from fastapi import APIRouter, Request

from api.schemas import StatusResponse

router = APIRouter()


@router.get("/status", response_model=StatusResponse)
def status(request: Request) -> StatusResponse:
    """前端用来显示服务是否可提问；不触发模型加载或云端调用。"""
    service = getattr(request.app.state, "service", None)
    if service is None or not request.app.state.ready:
        return StatusResponse(ready=False, vector_count=0, index_version=None,
                              reranker_enabled=False, reranker_device=None)
    store = service.retriever.store
    reranker = service.reranker
    return StatusResponse(
        ready=True,
        vector_count=len(store.chunks),
        index_version=Path(store.index_path).parent.name,
        reranker_enabled=reranker is not None,
        reranker_device=reranker.device if reranker is not None else None,
    )
