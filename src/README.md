# 源码目录

这里保存具体实现，手动运行请使用项目根目录 `scripts/` 下的入口。

```text
src/
├── rag_agent/                # 核心 RAG 业务
│   ├── config.py             # 路径、切块、检索、重排与 LLM 配置
│   ├── ingestion/            # PDF 解析、缓存恢复、结果适配与本地规则后处理
│   ├── indexing/             # 切块、向量化、BM25、产物复用与同代索引发布
│   ├── qa/                   # 混合召回、RRF、BGE 重排、证据门控与回答生成
│   └── common/               # JSON 保存、文件锁与进度提示
└── devtools/                 # 开发辅助实现
    ├── evaluation/           # 批量评测、检索指标、结果导出、模型判分与汇总
    └── rag_inspector/        # 问答过程记录与 JSON、HTML 诊断报告
```

依赖方向：开发工具可以调用核心 RAG，核心 RAG 不依赖开发工具。
建议阅读顺序：`ingestion → indexing → qa`，再看 `common` 和 `devtools`。
