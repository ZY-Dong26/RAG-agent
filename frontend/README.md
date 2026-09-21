# 前端目录

这里预留给后续网页或桌面界面。前端不直接读取 `vector_db`，也不直接调用 MinerU、
Embedding 或 LLM 客户端；问答统一经过 `rag_agent.qa.service.RAGService`。

- 如果使用 Streamlit、Gradio 等 Python 前端，可以直接调用 `RAGService.ask()`。
- 如果使用 React、Vue 等浏览器前端，后续在独立 API 层包装同一个服务，再通过 HTTP 调用。
- PDF 上传、建库进度和失败文档状态以后也应由应用/API 层暴露，不把本地路径写进页面组件。

当前没有安装任何前端依赖，这个目录不会影响命令行解析、建库和聊天流程。
