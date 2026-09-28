"""
chat.py —— 单轮问答 HTTP 路由（普通 JSON 与 SSE 流式响应）

复用同一 RAGService 实例。当前 Generator 保存 last_usage，且 6 GB 显存不宜并发重排，
因此同时只允许一个问答请求运行；第二个请求收到明确的忙碌状态。
"""
import logging
import json
from pathlib import Path
from threading import Lock

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from api.schemas import AskRequest, AskResponse, Citation
from rag_agent.storage.chat_history import ConversationNotFound
from rag_agent.chat.snapshot import save_answer_snapshot

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


def _answer_response(result, conversation_id: str) -> AskResponse:
    """两种接口共用最终结果格式；入库前先校验公开字段。"""
    if not result.answer:
        raise ValueError("回答为空")
    return AskResponse(
        conversation_id=conversation_id,
        answer=result.answer,
        answerable=result.answerable,
        reason=result.reason,
        evidence_status=result.evidence_status,
        threshold_calibrated=result.threshold_calibrated,
        citations=[_citation(hit) for hit in result.hits],
        timing=result.timing,
    )


def _sse(name: str, data: dict) -> str:
    """每个事件独立编码为 JSON，正文中的换行不会破坏 SSE 分帧。"""
    return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


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
        # 先验证公开响应，再写数据库；验证失败不会留下半组问答。
        response = _answer_response(result, payload.conversation_id or "")
        response.conversation_id = save_answer_snapshot(
            history, payload.conversation_id, payload.question, response,
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


@router.post("/ask/stream")
def ask_stream(payload: AskRequest, request: Request) -> StreamingResponse:
    """以 SSE 推送状态和答案增量；只有完整生成后才保存本轮问答。"""
    service = getattr(request.app.state, "service", None)
    if service is None or not request.app.state.ready:
        raise HTTPException(status_code=503, detail="问答服务尚未就绪")
    lock = request.app.state.ask_lock
    if not lock.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="正在处理另一条问题，请稍后重试")
    history = request.app.state.history
    try:
        exists = not payload.conversation_id or history.exists(payload.conversation_id)
    except Exception as error:
        lock.release()
        logger.error("流式问答准备失败：%s", type(error).__name__)
        raise HTTPException(status_code=500, detail="问答处理失败，请检查后端日志") from None
    if not exists:
        lock.release()
        raise HTTPException(status_code=404, detail="会话不存在")

    # 响应建立后可能在开始迭代前被断开；后台回调和迭代器 finally 共用幂等释放。
    release_guard = Lock()
    released = False

    def release_once():
        nonlocal released
        with release_guard:
            if not released:
                released = True
                lock.release()

    def events():
        try:
            for name, data in service.ask_stream(payload.question):
                if name == "retrieval":
                    # 检索完成即可展示来源；只公开 _citation 允许的字段。
                    yield _sse(name, {
                        "citations": [_citation(hit).model_dump() for hit in data["hits"]],
                        "timing": data["timing"],
                    })
                elif name == "done":
                    response = _answer_response(data, payload.conversation_id or "")
                    response.conversation_id = save_answer_snapshot(
                        history, payload.conversation_id, payload.question, response,
                    )
                    yield _sse("done", response.model_dump())
                else:
                    yield _sse(name, data)
        except Exception as error:
            # 响应头已发送，不能再改为 HTTP 500；事件只带通用提示。
            logger.error("流式问答失败：%s", type(error).__name__)
            yield _sse("error", {"message": "问答处理失败，请检查后端日志"})
        finally:
            release_once()

    return StreamingResponse(
        events(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        background=BackgroundTask(release_once),
    )
