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
    throw new Error(typeof data.detail === 'string' ? data.detail : '请求失败，请稍后重试。')
  }
  return data
}

/** 只读状态查询，不会触发问答或云端模型。 */
export function getStatus() {
  return request('/api/v1/status')
}

/** 单轮提问：只发送当前 question；引用和耗时由后端统一返回。 */
export function askQuestion(question) {
  return request('/api/v1/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
  })
}
