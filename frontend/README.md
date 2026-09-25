# 前端工程

这是独立的 JavaScript/Vite 工程，目前只有基础页面和构建入口，不包含问答、上传或 API 调用。

在本目录执行：

~~~powershell
npm install
npm run dev
~~~

构建检查使用 `npm run build`。Vite 8 需要 Node.js 20.19+ 或 22.12+。

浏览器前端以后通过独立 API 层调用 `backend/src/rag_agent/qa/service.py` 提供的服务；不直接读取索引、模型或后端 `.env`。
