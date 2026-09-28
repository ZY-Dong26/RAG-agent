# 数据目录说明

```text
data/
├── raw/                     # 原始资料：PDF，不随意删除
├── evaluation/              # 原始评测输入，不参与知识库构建
├── processed/               # 中间产物：解析缓存、清洗结果、切块和建库报告
│   └── mineru/              # 云端原始结果与任务恢复记录
├── vector_db/               # 索引产物：正式索引、历史版本和单文档向量
├── chat_history.sqlite3     # 运行时数据：本机聊天记录，Git 忽略
└── outputs/                 # 运行时数据：可按批次整理的工具输出
    ├── evaluation/          # 逐题结果、CSV、Markdown 与运行清单
    └── debug/               # 调试 JSON 和 HTML 报告
```

- 以下相对路径均以 `backend/` 为基准。核心源码位于 `src/rag_agent/`，日常入口位于 `scripts/`；所有运行入口位于 `scripts/`，辅助实现位于 `src/devtools/`。
- 评测输入不会参与知识库构建；不要将参考答案放到 `raw/`。
- `evaluation/test_dataset.json` 保留新测评集原稿；从中转换的 `evaluation/testdata.json` 保存 100 道题、参考答案、原文证据和评分标准，供默认批量评测入口读取。`evaluation/testdata_sources.json` 保存文档编号和 PDF 文件名的映射。拒答题的来源表示相关资料，不表示该资料支持题目中的虚构前提。
- 以上分类说明用途，不迁移现有目录。`raw/` 路径参与稳定文档 ID，`processed/` 和 `vector_db/` 含可复用缓存及索引；`chat_history.sqlite3` 保留现有会话。纯源码整理无需重建索引。
- `processed/mineru/` 用于避免重复上传；删除后可能需要再次使用云端解析。
- `chat_history.sqlite3` 仅保存会话标题、消息正文，以及回答当时的引用片段、耗时和拒答状态快照；不保存密钥、模型对象或整份 PDF。删除该文件会丢失全部聊天记录，但不影响知识库和索引。
- 删除 `vector_db/` 后需要重新建库；目录迁移时保留的既有索引仍在这里。
- `vector_db/document_artifacts/<artifact_key>/manifest.json` 的 `chunking_report` 是每份文档的切块审计统计；`chunks.json` 可核对章节前缀、页码和来源块 ID。
- `outputs/` 不是运行知识库的必要输入，但删除评测结果会丢失断点恢复与效果对比依据。
- 评测集、映射和本说明可纳入 Git；原始 PDF、解析缓存、索引与工具输出继续忽略。
- 旧评测批次迁移时核验了索引与元数据内容；`run.before-data-move.json` 保留原清单，
  `path-migration.json` 记录此次纯路径迁移。实际模型、索引或代码变更仍受恢复校验保护。

## backend 目录迁移

数据、模型和配置随 Python 工程移入 `backend/`。索引文件与原始资料未重新生成；默认模型的建库签名保留迁移前的逻辑标识，以继续复用现有向量。

已有评测批次的 `run.json` 仅更新了模型绝对路径，并在各批次保留 `run.before-backend-move.json`。最新批次的源码快照同步记录了这次建库签名路径兼容修改；更早批次原本已有其他源码差异，仍按原有规则拒绝混跑。每代索引清单的 `embedding_model` 展示路径同步更新，并保留 `build_manifest.before-backend-move.json`。
