# 前端工程

这是学习 RAG 的单页问答界面，使用 Vue 3、Vite 和 JavaScript。页面负责输入问题、展示 Markdown/KaTeX 公式、最终引用和各阶段耗时；检索、重排、证据门控与生成都由 `backend/` 完成。浏览器不加载索引、模型文件或 API Key。

## 从哪里开始读

~~~text
frontend/
├── index.html                    # 提供 Vue 挂载节点 #app
├── vite.config.js                # 开发时把 /api 转发到 FastAPI
├── package.json                  # 前端依赖和运行命令
└── src/
    ├── main.js                   # 挂载 App 并加载全局/KaTeX 样式
    ├── App.vue                   # 本页会话、服务状态、输入框和整体布局
    ├── components/ChatMessage.vue # 单条消息、公式、引用与耗时展示
    ├── lib/api.js                # 唯一的 HTTP 请求入口和错误提示转换
    └── style.css                 # 主题变量、响应式布局和回答样式
~~~

建议按 `main.js → App.vue → lib/api.js → components/ChatMessage.vue → style.css` 阅读。`App.vue` 管理交互状态，`api.js` 负责请求，`ChatMessage.vue` 只展示传入的消息。

## 一次提问如何流转

~~~text
输入框 → App.vue 校验并显示等待消息 → api.js POST /api/v1/ask
      → Vite 代理 → FastAPI → RAGService.ask()
      → App.vue 写入答案、引用与耗时 → ChatMessage.vue 渲染
~~~

前端只发送 `{"question":"当前问题"}`，不发送上方的对话历史。页面虽然能展示多轮消息，后端目前仍逐题独立检索；左侧会话列表也只存在于当前页面内存，刷新后清空。只有深浅色偏好保存在浏览器 `localStorage`。

## 启动与验证

如果后端环境和索引已经准备好，可在项目根目录运行 [start.ps1](../start.ps1)，它会启动前后端并打开浏览器。首次安装依赖仍按下面两端分别完成。

先在另一个 PowerShell 终端启动后端（工作目录 `backend/`）：

```powershell
.\.venv\Scripts\python.exe scripts/serve_api.py
```

再在另一个 PowerShell 终端进入 `frontend/` 目录执行：

```powershell
npm ci          # 首次安装，或 package-lock.json 变化后执行
npm run dev
```

打开 Vite 显示的地址，默认是 `http://127.0.0.1:5173/`。本机已用 Node.js 24 验证。`npm run dev` 只启动前端，不会自动启动 Python。生产构建检查使用 `npm run build`；当前仓库没有配置前端生产部署。

开发时浏览器只请求同源的 `/api`，`vite.config.js` 将它转发到 `http://127.0.0.1:8000`，因此页面代码不写死后端地址，也不需要开发用跨域配置。部署到别处时，需要由部署环境提供同样的 `/api` 路由。

## 与后端的接口

| 请求 | 用途 | 页面如何使用 |
|---|---|---|
| `GET /api/v1/status` | 返回 `ready`、向量数量和重排设备 | 打开页面和每 30 秒查询一次；未就绪时禁用发送 |
| `POST /api/v1/ask` | 接收单个 `question`，返回答案、引用、证据状态和分项耗时 | 发送后先显示等待占位，完成时填入本轮结果 |

`ready=true` 只表示后端已加载本地索引和模型，不代表云端回答接口一定可用。`POST` 可能因空问题或长度超限返回 422，后端忙碌返回 429，处理失败返回 500。`api.js` 把这些状态转换为页面提示；Vite 无法连接后端时也会提示先启动 FastAPI。字段定义见 [API 数据格式](../backend/src/api/schemas.py)。

## 当前范围

- 回答支持 `$...$` 行内公式和单独成行的 `$$...$$` 块级公式；复制回答保留原始文本。
- 左侧的新对话、搜索和历史列表只在当前页面内存中工作；刷新后清空，后端尚无会话存储。
- 页面会显示多轮消息，但每次仅向后端发送当前问题，现有 RAG 服务不会读取上方对话历史。
- 示例提问只填入输入框，点击发送才会调用后端及已配置的云端回答模型。
- 右上角和侧栏显示真实后端状态；未连接时禁用发送。回答中的引用和耗时可展开查看。
- 当前回答为完成后一次性返回，尚未接入流式输出、模型选择、文件上传或 Agent。
- 深浅色偏好保存在浏览器本地；对话内容不会保存在浏览器持久存储中。

## 回答的渲染与安全

`ChatMessage.vue` 先用 Marked 把回答解析为 Markdown，再由 KaTeX 扩展渲染公式。`nonStandard` 选项允许 `$...$` 紧贴中文。生成的 HTML 经 DOMPurify 清理后才交给 `v-html`；“复制”按钮复制未渲染的原始回答文本，便于粘贴到笔记或编辑器。引用列表和召回、重排、生成耗时来自同一次 API 响应，展开面板不会再次检索。

## 排查顺序

页面显示“未连接”时，先看后端终端是否已显示启动完成，再访问 `http://127.0.0.1:8000/api/v1/status`。若页面显示就绪但提问失败，检查后端终端提示的失败阶段：状态接口不检查云端 LLM 连通性。后端启动、配置和离线测试分别见 [运行入口](../backend/scripts/README.md)、[配置示例](../backend/.env.example) 和 [测试说明](../backend/tests/README.md)。
