# RAG-agent

用于学习 RAG 的 PDF 知识库项目，轻量工程化，覆盖从原始 PDF 到评测判分的完整链路。

```text
建库：PDF → MinerU 云端解析 → 本地规则后处理 → 章节感知切块 → Qwen3 Embedding + 中文 BM25 → 同代索引
问答：Dense/BM25 召回 → RRF 融合 → BGE 重排 → 证据门控 → 云端 LLM 回答
```

MinerU 在解析时接收完整 PDF；日常问答复用本地索引，只向回答模型发送问题和检索文本。
当前是文本 RAG，保存解析图片不等于支持图片理解。

## 功能总览

| 能力 | 说明 |
|---|---|
| **云端解析** | MinerU 解析 PDF，保留页码和来源；按内容哈希缓存，不重复上传 |
| **规则后处理** | 本地识别重复页眉页脚、跨页正文续接和编号标题；全程可审计并支持失败回退 |
| **章节切块** | 按连续章节组织正文，保护公式和表格原子块，保留来源块 ID 并输出完整性统计 |
| **增量建库** | 未变化文档复用向量；FAISS 与 BM25 候选索引共同验证后才切换 |
| **混合检索** | Qwen3 Dense 与中文 BM25 双路召回，经 RRF 去重融合和本地 BGE 重排 |
| **证据门控** | 无候选或低于已校准重排阈值时直接拒答，不调用回答模型 |
| **命令行问答** | 本地检索与重排 + 云端 LLM 生成，回答带最终引用来源编号 |
| **调试诊断** | 每轮问答可导出 HTML 报告，展示实际发送的 prompt、命中块是否被截断、相邻 chunk |
| **批量评测** | 跑 84 道题评测集，统计 hit@k、recall@k、MRR、延迟和 token 消耗 |
| **模型判分** | 用独立裁判模型按正确性/完整性/相关性三维度打分，支持断点续跑 |
| **离线导出** | 从已有批次纯本地重建 CSV、报告和汇总表，不调模型、不花 token |

## 目录与文档导航

```text
RAG-agent/
├── scripts/           # 七个运行入口
├── src/               # 核心 RAG 与评测、调试实现
├── data/              # 原始资料、解析缓存、向量库、评测输入和输出
├── tests/             # 离线自动化测试
├── model/             # 本地 Embedding 与 BGE 重排模型
├── frontend/          # 未来前端预留，目前没有可运行界面
├── .env.example       # 环境配置模板及参数注释
├── requirements.txt   # Python 依赖
└── README.md          # 本文件
```

各目录细节集中在对应文档，根目录只保留首次运行所需内容：

| 想了解什么 | 去哪里查看 |
|---|---|
| 每个入口做什么、所有命令参数、PyCharm 怎么运行 | [scripts/README.md](scripts/README.md) |
| 各源码模块的职责与学习顺序 | [src/README.md](src/README.md) |
| 数据目录结构、缓存和输出的保留规则 | [data/README.md](data/README.md) |
| 离线测试范围与运行命令 | [tests/README.md](tests/README.md) |
| 后续前端的设计边界 | [frontend/README.md](frontend/README.md) |

## 首次准备

### 1. 安装依赖

当前开发环境为 Windows、Python 3.12。在项目根目录的 PowerShell 终端执行：

```powershell
# 已有 .venv 时跳过第一条
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

命令直接使用项目解释器，无需提前激活虚拟环境。启动脚本会设置 `src` 导入路径，无需额外安装项目包。
不需要安装本地 MinerU 或下载其解析模型。依赖版本尚未在全新环境中验证；需要 GPU 时须准备适配本机的 PyTorch 构建，没有可用 CUDA 时 Embedding 使用 CPU。

### 2. 填写配置

首次创建 `.env`，已有文件不会被覆盖：

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

打开项目根目录 `.env`，主要填写：

- `MINERU_API_KEY`：云端 PDF 解析密钥。
- `LLM_API_KEY`：回答模型密钥，同时核对 `LLM_BASE_URL` 和 `LLM_MODEL`。
- `JUDGE_LLM_API_KEY`：可选的独立裁判模型密钥；同时配置 `JUDGE_LLM_BASE_URL` 和 `JUDGE_LLM_MODEL`。
- `EMBEDDING_MODEL`：本地向量模型目录，默认 `model/Qwen3-Embedding-0.6B`。
- `RERANKER_MODEL`：本地重排模型目录，默认 `model/bge-reranker-v2-m3`。

其他参数的含义见 [.env.example](.env.example)，不在这里重复配置清单。
两个服务使用不同的密钥变量；系统中的同名环境变量优先。真实 `.env` 被 Git 忽略。

### 3. 准备 PDF 和本地模型

将 PDF 直接放入 `data/raw/`，当前不递归扫描子目录。
将完整的 Sentence Transformers 模型放入 `model/Qwen3-Embedding-0.6B/`，包括权重、分词器、`modules.json` 和池化配置。
另行把 `BAAI/bge-reranker-v2-m3` 的完整 Transformers 模型目录放到
`model/bge-reranker-v2-m3/`。程序使用 `local_files_only=True`，不会自动下载模型。
模型相对路径基于项目根目录，也支持配置绝对路径。只运行 PDF 解析时不需要本地模型或 LLM 密钥。

临时没有重排模型时可在 `.env` 设置 `RERANK_ENABLED=false`。此时系统按 RRF 排名取前 5 条，
且只能在拒答阈值为 `None` 时运行；该模式用于临时排障，不代表正式效果。

## 快速开始

准备完成后，先建库，再聊天。不必提前单独运行解析，建库已包含该步骤。

```powershell
# 按需解析 PDF、切块和计算向量，发布知识库
.\.venv\Scripts\python.exe scripts/build_index.py

# 加载知识库，输入问题并调用回答模型
.\.venv\Scripts\python.exe scripts/chat.py
```

建库结束后查看 `data/processed/build_report.json`，确认哪些文档入库、失败或沿用了旧版。
聊天输入 `exit` 退出；更新知识库后需重启聊天入口。

升级前只有 FAISS 的旧索引不能用于混合检索，聊天会明确提示重建。重新执行建库会复用未变化文档
的解析缓存和向量产物，只在本地补建 BM25 和发布新索引，不会因此重新上传成功缓存的 PDF；
原本就失败或发生变化的文档仍按正常增量规则处理。

批量评测、模型判分、调试报告等操作，统一查阅 [scripts/README.md](scripts/README.md)。

## 需要了解的运行规则

- **三阶段处理**：MinerU 原始解析、`rules-v1` 本地后处理、文本切片彼此独立。原始 ZIP 与适配缓存不覆盖；规则版本变化只复用原始缓存重新执行本地后处理，不重新上传 PDF。`rules-v1` 不调用 LLM、不合并跨页表格、只保守处理有编号标题，失败时回退原始文档继续建库。
- **章节感知切块**：连续同章节先组织后切分，短小兄弟章节在不跨主标题和特殊结构边界时合并；正文目标约 1000 字，800–1500 字为主要范围。独立公式不截断，长表格只按行分片并重复图注/表头。每个 chunk 保留章节、页码、来源块 ID 和块类型；单文档产物 `manifest.json` 保存长度分布、短块原因、超长原子块和完整性校验。将 `indexing/chunker.py` 的 `SECTION_AWARE_CHUNKING` 改为 `False` 可回退旧逐块策略。
- **增量建库**：复用未变化文档的向量，再组装完整候选索引。默认允许部分解析失败；新文件失败时报告未入库，已有文档更新失败时可保留旧版。向量计算或写入异常仍会中止构建。
- **混合召回**：Dense 与 BM25 各取 30 条，按稳定 `chunk_id` 去重后使用等权 RRF 融合为 20 条，再由 BGE 重排取前 5 条。BM25 使用 `jieba.lcut_for_search()`，同时保留英文缩写、型号、版本号、年份和百分比。
- **RRF 原因**：余弦相似度与 BM25 分数不在同一量纲，不能直接相加；RRF 只组合各路排名，默认公式为 `weight / (60 + rank)`。
- **拒答阈值**：默认 `RERANK_REJECT_THRESHOLD=None`，表示尚未校准，不会凭经验硬拒答。应使用项目评测集统计可回答/不可回答问题的 Top-1 `rerank_score` 分布后再设置阈值；sigmoid 分数只是单调映射，不是真实概率。
- **索引保护**：每代同时保存并回读验证 FAISS、BM25 和 chunk ID 顺序，全部通过后才切换活动版本；历史索引和解析缓存不会自动删除。正式索引在 `data/vector_db/`，数据保留规则见 [数据目录说明](data/README.md)。
- **评测与判分**：结果写入 `data/outputs/`。跑评测、纯本地导出、裁判判分三个入口相互独立；已有批次可再用独立裁判模型按正确性、完整性和相关性判分，详见 [运行入口说明](scripts/README.md) 第 4–6 节。
- **当前边界**：聊天各轮独立检索，没有多轮历史；检索指标不等于答案正确率。离线测试验证程序行为，真实解析质量与回答效果仍需人工验收。

遇到解析或建库失败，先看终端阶段提示及 `data/processed/` 下的报告。
需要重新提交云端解析时，按 [运行入口说明](scripts/README.md) 中的 `--resubmit` 操作；重新提交可能消耗额外额度。
