# 源码目录

这里保存具体实现，手动运行请使用 `backend/scripts/` 下的入口。

```text
src/
├── rag_agent/                # 核心 RAG 业务
│   ├── config.py             # 路径、切块、检索、重排与 LLM 配置
│   ├── ingestion/            # PDF 解析、缓存恢复、结果适配与本地规则后处理
│   ├── indexing/             # 切块、向量化、BM25、产物复用与同代索引发布
│   ├── qa/                   # 混合召回、RRF、BGE 重排、证据门控与回答生成
│   └── common/               # JSON 保存、文件锁与进度提示
├── api/                      # FastAPI 入口、状态/问答路由及公开响应格式
└── devtools/                 # 开发辅助实现
    ├── evaluation/           # 批量评测、检索指标、结果导出、模型判分与汇总
    └── rag_inspector/        # 问答过程记录与 JSON、HTML 诊断报告
```

依赖方向：API 与开发工具可以调用核心 RAG；核心问答和建库逻辑不依赖 API 或开发工具。浏览器调用 `api/routes/chat.py`，该路由再调用 `rag_agent/qa/service.py`；浏览器不直接导入 Python 包。`api/schemas.py` 定义公开请求与响应字段。

建议先按 `ingestion → indexing → qa` 学习数据与问答流程，再读 `api` 看 HTTP 适配，最后看 `common` 和 `devtools`。前端文件与状态流转见 [前端说明](../../frontend/README.md)。
