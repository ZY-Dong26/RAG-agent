# RAG-agent

用于学习 RAG 的 PDF 知识库项目，轻量工程化，覆盖从原始 PDF 到评测判分的完整链路。

```text
建库：PDF → MinerU 云端解析 → 本地切块 → Qwen3 Embedding → FAISS 索引
问答：用户问题 → 本地向量检索 → 检索文本与问题 → 云端 LLM 回答
```

MinerU 在解析时接收完整 PDF；日常问答复用本地索引，只向回答模型发送问题和检索文本。
当前是文本 RAG，保存解析图片不等于支持图片理解。

## 功能总览

| 能力 | 说明 |
|---|---|
| **云端解析** | MinerU 解析 PDF，保留页码和来源；按内容哈希缓存，不重复上传 |
| **增量建库** | 未变化文档复用向量，只给新文档算 Embedding；候选索引验证后才切换 |
| **命令行问答** | 本地检索 + 云端 LLM 生成，回答带引用来源编号 |
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
├── model/             # 本地 Embedding 模型
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

其他参数的含义见 [.env.example](.env.example)，不在这里重复配置清单。
两个服务使用不同的密钥变量；系统中的同名环境变量优先。真实 `.env` 被 Git 忽略。

### 3. 准备 PDF 和向量模型

将 PDF 直接放入 `data/raw/`，当前不递归扫描子目录。
将完整的 Sentence Transformers 模型放入 `model/Qwen3-Embedding-0.6B/`，包括权重、分词器、`modules.json` 和池化配置。
模型相对路径基于项目根目录，也支持配置绝对路径。只运行 PDF 解析时不需要向量模型或 LLM 密钥。

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

批量评测、模型判分、调试报告等操作，统一查阅 [scripts/README.md](scripts/README.md)。

## 需要了解的运行规则

- **解析缓存**：按 PDF 内容、解析参数与适配版本等识别缓存。未变化文件复用结果，修改切块参数无需重新上传；普通重跑优先恢复原任务，不盲目创建新任务。
- **增量建库**：复用未变化文档的向量，再组装完整候选索引。默认允许部分解析失败；新文件失败时报告未入库，已有文档更新失败时可保留旧版。向量计算或写入异常仍会中止构建。
- **索引保护**：候选索引验证后才切换活动版本；历史索引和解析缓存不会自动删除。正式索引在 `data/vector_db/`，数据保留规则见 [数据目录说明](data/README.md)。
- **评测与判分**：结果写入 `data/outputs/`。跑评测、纯本地导出、裁判判分三个入口相互独立；已有批次可再用独立裁判模型按正确性、完整性和相关性判分，详见 [运行入口说明](scripts/README.md) 第 4–6 节。
- **当前边界**：聊天各轮独立检索，没有多轮历史；检索指标不等于答案正确率。离线测试验证程序行为，真实解析质量与回答效果仍需人工验收。

遇到解析或建库失败，先看终端阶段提示及 `data/processed/` 下的报告。
需要重新提交云端解析时，按 [运行入口说明](scripts/README.md) 中的 `--resubmit` 操作；重新提交可能消耗额外额度。
