// api.js —— 前端唯一的 HTTP 访问层。
// 页面只用相对路径请求 /api；开发时 Vite 把请求代理到本机 FastAPI，避免页面写死后端地址。
// 返回已解析的 JSON，失败则抛出适合直接展示给用户的错误文本；不要在这里记录问题或密钥。

/** 输入为 API 路径和 fetch 选项；输出为 JSON 响应，HTTP/网络错误转成可显示的 Error。 */
async function request(path, options = {}) {
  let response
  try {
    response = await fetch(path, options)
  } catch {
    throw new Error('无法连接后端服务，请先启动 FastAPI。')
  }

  const data = await response.json().catch(() => ({}))
  if (!response.ok) {
    // 后端未启动时，Vite 代理通常返回 502，响应体不一定是 JSON。
    if ([502, 503, 504].includes(response.status)) {
      throw new Error('前端已启动，但后端未连接。请先启动 FastAPI 服务，再点击重试。')
    }
    if (response.status === 429) {
      throw new Error('知识库正在处理另一条问题，请稍后再试。')
    }
    if (response.status === 422) {
      throw new Error('问题不能为空，且不能超过 2000 个字符。')
    }
    if (response.status === 404) {
      throw new Error('会话不存在，可能已被删除。请刷新会话列表。')
    }
    throw new Error(typeof data.detail === 'string' ? data.detail : '请求失败，请稍后重试。')
  }
  return data
}

/** 只读状态查询，不会触发问答或云端模型。 */
export function getStatus() {
  return request('/api/v1/status')
}

/** 单轮提问：conversation_id 只用于归档；不会把历史消息发送给模型。 */
export function askQuestion(question, conversationId = null) {
  return request('/api/v1/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, conversation_id: conversationId }),
  })
}

/** SSE 分帧器：跨网络块保留未结束的事件，也能处理正文里的中文和换行。 */
export function createSseParser(onEvent) {
  let buffer = ''
  return {
    push(text) {
      buffer = (buffer + text).replace(/\r\n/g, '\n')
      let boundary
      while ((boundary = buffer.indexOf('\n\n')) !== -1) {
        const frame = buffer.slice(0, boundary)
        buffer = buffer.slice(boundary + 2)
        let name = 'message'
        const lines = []
        for (const line of frame.split('\n')) {
          if (line.startsWith('event:')) name = line.slice(6).trim()
          if (line.startsWith('data:')) lines.push(line.slice(5).trimStart())
        }
        if (lines.length) onEvent(name, JSON.parse(lines.join('\n')))
      }
    },
  }
}

/** POST 流式提问；收到 done 才算完整成功，流中断会抛出错误。 */
export async function streamQuestion(question, conversationId, onEvent) {
  let response
  try {
    response = await fetch('/api/v1/ask/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, conversation_id: conversationId }),
    })
  } catch {
    throw new Error('无法连接后端服务，请先启动 FastAPI。')
  }
  if (!response.ok) {
    const data = await response.json().catch(() => ({}))
    if (response.status === 429) throw new Error('知识库正在处理另一条问题，请稍后再试。')
    if (response.status === 404) throw new Error('会话不存在，可能已被删除。请刷新会话列表。')
    if (response.status === 422) throw new Error('问题不能为空，且不能超过 2000 个字符。')
    if ([502, 503, 504].includes(response.status)) throw new Error('前端已启动，但后端未连接。请先启动 FastAPI 服务，再点击重试。')
    throw new Error(typeof data.detail === 'string' ? data.detail : '请求失败，请稍后重试。')
  }
  if (!response.body) throw new Error('浏览器无法读取流式响应。')

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let done = null
  const parser = createSseParser((name, data) => {
    if (name === 'error') throw new Error(data.message || '问答处理失败。')
    if (name === 'done') done = data
    onEvent(name, data)
  })
  try {
    while (true) {
      const { value, done: ended } = await reader.read()
      if (ended) break
      parser.push(decoder.decode(value, { stream: true }))
    }
    parser.push(decoder.decode())
    if (!done) throw new Error('回答传输中断，本轮没有完整保存。')
    return done
  } catch (error) {
    await reader.cancel().catch(() => {})
    throw error
  } finally {
    reader.releaseLock()
  }
}

/** 页面启动时获取摘要；点击会话后再按 ID 获取消息快照。 */
export function listConversations() {
  return request('/api/v1/conversations')
}

export function getConversation(id) {
  return request(`/api/v1/conversations/${encodeURIComponent(id)}`)
}

/** 只删除聊天记录，后端不会触碰知识库和索引。 */
export function deleteConversation(id) {
  return request(`/api/v1/conversations/${encodeURIComponent(id)}`, { method: 'DELETE' })
}
