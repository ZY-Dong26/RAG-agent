"""问答快照保存：完整答案形成后，提取公开结果并成组写入会话。

输入是已经由 API 校验的最终响应；本模块不认识 HTTP，也不执行 SQL。普通问答与
流式问答共用这一步，因此生成中断时不会留下半截回答。
"""


def save_answer_snapshot(history, conversation_id, question, response):
    """保存问题、答案和当轮公开快照，返回新建或已有的会话 ID。

    :param history: 负责原子写入的 SQLite 会话对象。
    :param conversation_id: 已有会话 ID；首轮传 None。
    :param question: 本轮用户问题。
    :param response: 已验证的公开响应模型，需提供 model_dump() 和 answer。
    """
    snapshot = response.model_dump(exclude={"conversation_id", "answer"})
    return history.save_exchange(conversation_id, question, response.answer, snapshot)
