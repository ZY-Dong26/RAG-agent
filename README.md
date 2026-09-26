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
| **本机问答 API** | FastAPI 启动时预热模型，向当前 Vue 页面和其他客户端提供状态与单轮问答接口 |
| **Vue 问答页面** | 展示本页会话、Markdown/公式回答、最终引用和分项耗时；通过 Vite 代理调用 API |
| **调试诊断** | 每轮问答可导出 HTML 报告，展示实际发送的 prompt、命中块是否被截断、相邻 chunk |
| **批量评测** | 跑 84 道题评测集，统计 hit@k、recall@k、MRR、延迟和 token 消耗 |
| **模型判分** | 用独立裁判模型按正确性/完整性/相关性三维度打分，支持断点续跑 |
| **离线导出** | 从已有批次纯本地重建 CSV、报告和汇总表，不调模型、不花 token |

## 项目结构

~~~text
RAG-agent/
├── backend/          # Python 核心、FastAPI、数据、模型和评测
│   └── README.md     # 后端准备、运行规则和实现细节
├── frontend/         # Vue 3 + Vite 问答页面
│   └── README.md     # 前端文件、状态和接口说明
├── start.ps1          # 在 Windows 上一键启动前后端并打开浏览器
└── README.md         # 项目总览（本文件）
~~~

前端经 `/api` 调用 FastAPI，后端统一调用 `RAGService.ask()`；命令行问答也复用同一服务。两端的职责、目录和学习顺序分别见 [后端说明](backend/README.md) 与 [前端说明](frontend/README.md)。

## 如何开始

首次运行需要 Windows/Python 3.12 的后端虚拟环境、Node.js 前端依赖、本地 Embedding 与重排模型，以及 `backend/.env` 中的云端服务配置。安装、模型放置、PDF 路径和可选 CUDA 设置按 [后端首次准备](backend/README.md#首次准备) 操作；前端依赖按 [前端启动说明](frontend/README.md#启动与验证) 安装。

### 一键启动（已有索引）

依赖、配置和索引准备好后，在项目根目录运行：

~~~powershell
.\start.ps1
~~~

脚本会打开后端、前端两个可见的 PowerShell 窗口，并打开浏览器。窗口中可以查看模型加载与请求日志；关闭两个窗口即可停止服务。脚本只负责启动，不会安装依赖或建库。

### 首次从 PDF 建库

把 PDF 放进 `backend/data/raw/`，从项目根目录运行：

~~~powershell
cd backend
.\.venv\Scripts\python.exe scripts/build_index.py
~~~

建库报告在 `backend/data/processed/build_report.json`。已有可用索引时可跳过这步。

### 手动启动后端

在一个终端进入 `backend/` 并等待模型和索引加载完成：

~~~powershell
cd backend
.\.venv\Scripts\python.exe scripts/serve_api.py
~~~

### 手动启动前端

另开一个终端进入 `frontend/`；首次运行先安装前端依赖：

~~~powershell
cd frontend
npm ci
npm run dev
~~~

打开终端显示的地址（默认 `http://127.0.0.1:5173/`）提问。后续依赖未变化时可跳过 `npm ci`。

前端显示的“已就绪”只说明本地索引和模型已加载；实际提问还依赖云端回答模型连接。也可在 `backend/` 运行 `.\.venv\Scripts\python.exe scripts/chat.py` 使用命令行入口。建库、解析、评测、判分与调试的完整参数见 [后端运行入口](backend/scripts/README.md)。

## 当前范围

- 这是文本 RAG。解析时保存图片不等于支持图片问答；日常问答只把问题与命中的检索文本发送给回答模型。
- 页面可以展示多轮消息，但每轮独立检索，后端不读取会话历史；页面刷新后，本页会话列表清空。
- 前端目前一次性展示完整回答，尚无流式输出、持久会话、模型或检索方式选择、文件上传和 Agent。
- 离线测试验证程序行为；检索指标与真实回答质量仍需结合评测集和人工检查。

## 文档导航

| 目的 | 文档 |
|---|---|
| 后端安装、配置、建库、问答和运行规则 | [backend/README.md](backend/README.md) |
| 前端目录、数据流、接口与排查 | [frontend/README.md](frontend/README.md) |
| 后端脚本参数与 PyCharm 运行方式 | [backend/scripts/README.md](backend/scripts/README.md) |
| 数据、缓存和索引的保留规则 | [backend/data/README.md](backend/data/README.md) |
| 离线测试范围与命令 | [backend/tests/README.md](backend/tests/README.md) |
