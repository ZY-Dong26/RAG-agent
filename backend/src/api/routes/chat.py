"""
chat.py —— 单轮问答 HTTP 路由

复用同一 RAGService 实例。当前 Generator 保存 last_usage，且 6 GB 显存不宜并发重排，
因此同时只允许一个问答请求运行；第二个请求收到明确的忙碌状态。
"""
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from api.schemas import AskRequest, AskResponse, Citation
from api.chat_history import ConversationNotFound

router = APIRouter()
logger = logging.getLogger("api")


def _citation(hit: dict) -> Citation:
    """只提取浏览器需要的最终证据字段，不返回任意索引元数据。"""
    metadata = hit.get("metadata", {})
    rank = metadata.get("rank")
    page = metadata.get("page")
    return Citation(
        rank=int(rank) if rank is not None else None,
        source=Path(str(metadata.get("source") or "未知来源")).name,
        page=page if isinstance(page, (int, str)) else None,
        text=str(hit.get("text", "")),
        rerank_score=metadata.get("rerank_score"),
        fusion_score=metadata.get("fusion_score"),
    )


@router.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest, request: Request) -> AskResponse:
    """只把当前问题交给 RAG；模型成功后再原子保存本轮问答。"""
    service = getattr(request.app.state, "service", None)
    if service is None or not request.app.state.ready:
        raise HTTPException(status_code=503, detail="问答服务尚未就绪")
    lock = request.app.state.ask_lock
    if not lock.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="正在处理另一条问题，请稍后重试")
    try:
        history = request.app.state.history
        if payload.conversation_id and not history.exists(payload.conversation_id):
            raise HTTPException(status_code=404, detail="会话不存在")
        result = service.ask(payload.question)
        if not result.answer:
            raise ValueError("回答为空")
        # 先验证公开响应，再写数据库；验证失败不会留下半组问答。
        response = AskResponse(
            conversation_id=payload.conversation_id or "",
            answer=result.answer,
            answerable=result.answerable,
            reason=result.reason,
            evidence_status=result.evidence_status,
            threshold_calibrated=result.threshold_calibrated,
            citations=[_citation(hit) for hit in result.hits],
            timing=result.timing,
        )
        snapshot = response.model_dump(exclude={"conversation_id", "answer"})
        response.conversation_id = history.save_exchange(
            payload.conversation_id, payload.question, response.answer, snapshot,
        )
        return response
    except ConversationNotFound:
        raise HTTPException(status_code=404, detail="会话不存在") from None
    except HTTPException:
        raise
    except Exception as error:
        # SDK 异常可能包含密钥或签名地址；只记录类型，不记录异常文本和问题内容。
        logger.error("问答请求失败：%s", type(error).__name__)
        raise HTTPException(status_code=500, detail="问答处理失败，请检查后端日志") from None
    finally:
        lock.release()
