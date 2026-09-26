# 数据目录说明

```text
data/
├── raw/                     # 原始 PDF，不随意删除
├── processed/               # 解析缓存、统一文本、切块和建库报告
│   └── mineru/              # 云端原始结果与任务恢复记录
├── vector_db/               # 正式索引、历史版本和单文档向量产物
├── evaluation/              # 人工编写的评测输入
│   ├── testdata.json         # 84 道题、参考答案与评分标准
│   └── testdata_sources.json # 文档编号到 PDF 文件名的映射
└── outputs/                 # 可按批次整理的工具输出
    ├── evaluation/          # 逐题结果、CSV、Markdown 与运行清单
    └── debug/               # 调试 JSON 和 HTML 报告
```

- 以下相对路径均以 `backend/` 为基准。核心源码位于 `src/rag_agent/`，日常入口位于 `scripts/`；所有运行入口位于 `scripts/`，辅助实现位于 `src/devtools/`。
- 评测输入不会参与知识库构建；不要将参考答案放到 `raw/`。
- `processed/mineru/` 用于避免重复上传；删除后可能需要再次使用云端解析。
- 删除 `vector_db/` 后需要重新建库；目录迁移时保留的既有索引仍在这里。
- `vector_db/document_artifacts/<artifact_key>/manifest.json` 的 `chunking_report` 是每份文档的切块审计统计；`chunks.json` 可核对章节前缀、页码和来源块 ID。
- `outputs/` 不是运行知识库的必要输入，但删除评测结果会丢失断点恢复与效果对比依据。
- 评测集、映射和本说明可纳入 Git；原始 PDF、解析缓存、索引与工具输出继续忽略。
- 旧评测批次迁移时核验了索引与元数据内容；`run.before-data-move.json` 保留原清单，
  `path-migration.json` 记录此次纯路径迁移。实际模型、索引或代码变更仍受恢复校验保护。

## backend 目录迁移

数据、模型和配置随 Python 工程移入 `backend/`。索引文件与原始资料未重新生成；默认模型的建库签名保留迁移前的逻辑标识，以继续复用现有向量。

已有评测批次的 `run.json` 仅更新了模型绝对路径，并在各批次保留 `run.before-backend-move.json`。最新批次的源码快照同步记录了这次建库签名路径兼容修改；更早批次原本已有其他源码差异，仍按原有规则拒绝混跑。每代索引清单的 `embedding_model` 展示路径同步更新，并保留 `build_manifest.before-backend-move.json`。
