# 源码目录

这里保存具体实现，手动运行请使用 `backend/scripts/` 下的入口。

```text
src/
├── rag_agent/                # 核心 RAG 业务
│   ├── config.py             # 路径、切块、检索、重排与 LLM 配置
│   ├── ingestion/            # PDF 解析、缓存恢复、结果适配与数据清洗
│   ├── indexing/             # 切块、向量化、索引产物复用与发布
│   ├── retrieval/            # Dense/BM25 召回、融合、重排与证据判断
│   ├── qa/                   # 送入模型的证据与提示词、LLM 生成
│   ├── chat/                 # 问答编排、流式输出、答案后处理与快照保存
│   ├── storage/              # SQLite 会话记录实现
│   └── common/               # JSON 保存、文件锁与进度提示
├── api/                      # FastAPI 入口、状态/问答路由及公开响应格式
└── devtools/                 # 开发辅助实现
    ├── evaluation/           # 批量评测、检索指标、结果导出、模型判分与汇总
    └── rag_inspector/        # 问答过程记录与 JSON、HTML 诊断报告
```

依赖方向：API 与开发工具可以调用核心 RAG；核心问答和建库逻辑不依赖 API 或开发工具。浏览器调用 `api/routes/chat.py`，该路由调用 `rag_agent/chat/service.py` 编排问答，通过 `chat/snapshot.py` 保存完整回答；SQLite 语句位于 `rag_agent/storage/chat_history.py`。`api/schemas.py` 定义公开请求与响应字段。
`qa/generator.py` 统一选出字符预算内的证据并构造提示词；`chat/answer.py` 在完整生成后校验引用编号。普通与流式问答使用相同规则，诊断工具仍可读取完整重排结果。

建议按 `ingestion → indexing → retrieval → qa → chat → api` 阅读数据与问答流程。前端文件与状态流转见 [前端说明](../../frontend/README.md)。
