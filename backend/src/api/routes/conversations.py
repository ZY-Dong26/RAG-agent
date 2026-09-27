"""conversations.py —— 聊天历史的只读列表、详情和删除接口。"""
import logging
import sqlite3

from fastapi import APIRouter, HTTPException, Request, Response

from api.schemas import ConversationDetail, ConversationSummary

router = APIRouter()
logger = logging.getLogger("api")


@router.get("/conversations", response_model=list[ConversationSummary])
def list_conversations(request: Request):
    """只返回标题和时间；消息正文由详情接口按需读取。"""
    try:
        return request.app.state.history.list_conversations()
    except sqlite3.Error as error:
        logger.error("会话列表读取失败：%s", type(error).__name__)
        raise HTTPException(status_code=500, detail="读取聊天记录失败") from None


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
def get_conversation(conversation_id: str, request: Request):
    """按 ID 读取保存时的消息、引用、耗时和拒答状态。"""
    try:
        conversation = request.app.state.history.get_conversation(conversation_id)
    except sqlite3.Error as error:
        logger.error("会话读取失败：%s", type(error).__name__)
        raise HTTPException(status_code=500, detail="读取聊天记录失败") from None
    if conversation is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return conversation


@router.delete("/conversations/{conversation_id}", status_code=204)
def delete_conversation(conversation_id: str, request: Request):
    """删除一条会话；问答执行期间返回忙碌，避免边生成边删除。"""
    lock = request.app.state.ask_lock
    if not lock.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="正在处理问题，请稍后删除会话")
    try:
        if not request.app.state.history.delete_conversation(conversation_id):
            raise HTTPException(status_code=404, detail="会话不存在")
        return Response(status_code=204)
    except sqlite3.Error as error:
        logger.error("会话删除失败：%s", type(error).__name__)
        raise HTTPException(status_code=500, detail="删除聊天记录失败") from None
    finally:
        lock.release()
