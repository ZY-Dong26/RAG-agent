# RAG-agent

PDF 知识库命令行问答：**MinerU 云端解析 → 本地切块 → Qwen3 Embedding → FAISS → 云端 LLM 回答**。

MinerU 在建库时接收完整 PDF，执行 OCR 和结构化解析；结果保存在本地。日常问答复用本地索引，不再次上传 PDF。LLM 会接收用户问题及检索到的文本。当前仍是文本 RAG，保存图片不等于支持图片理解。

## 1. 安装依赖

当前开发环境为 Windows、Python 3.12，在项目根目录的 PowerShell 中执行：

```powershell
# 已有 .venv 时跳过创建
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# 注册 src/rag_agent 包，便于测试、脚本和后续前端统一导入
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

不需要安装本地 MinerU 或下载它的解析模型。依赖固定了当前本机的直接依赖版本，尚未在全新环境中验证安装。PyTorch 的 GPU 构建需要按目标机器单独准备；当前本机为 `2.11.0+cu128`，依赖清单只指定基础版本。无可用 CUDA 时 Embedding 会使用 CPU。

## 2. 统一环境配置

所有配置集中在项目根目录 `.env`，用不同变量名区分服务：

- `LLM_API_KEY`：用于云端大模型回答。
- `MINERU_API_KEY`：用于 MinerU 文档解析。
- `EMBEDDING_MODEL`：本地向量模型目录。

首次克隆后从唯一示例创建配置，已有 `.env` 不覆盖：

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

```dotenv
# 大模型回答生成配置
LLM_API_KEY=填写LLM密钥
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
EMBEDDING_MODEL=model/Qwen3-Embedding-0.6B

# MinerU 云端文档解析配置
MINERU_API_KEY=填写MinerU密钥
MINERU_BASE_URL=https://mineru.net/api/v4
MINERU_MODEL=vlm
MINERU_LANGUAGE=ch
MINERU_OCR=true
MINERU_FORMULA=true
MINERU_TABLE=true
MINERU_TIMEOUT=60
MINERU_POLL_INTERVAL=5
MINERU_MAX_WAIT=1800
MINERU_RETRIES=3
MINERU_CACHE_REVISION=1
```

- 两个客户端只读取各自的配置项，不互相使用密钥；系统中的同名环境变量优先。
- `TIMEOUT` 是单次请求超时秒数；`MAX_WAIT` 是轮询等待预算，当前请求/重试可能使实际耗时略长。
- `RETRIES` 是查询/下载的最多尝试次数（包含第一次）。创建任务不会自动重试。
- API 地址目前限定为官方地址，不能直接改成协议不同的自建服务。
- `.env` 被 Git 忽略，`.env.example` 是不含真实密钥的唯一模板。

## 3. 准备文件与本地 Embedding 模型

PDF 放在 `data/raw/`，当前不递归读取子目录。云端客户端预检单文件不超过 200 MB、600 页，不填写局部页范围，按全文解析。

本地 Embedding 模型放在 `model/Qwen3-Embedding-0.6B/`。需要完整的 Sentence Transformers 模型目录，包括权重、分词器、`modules.json`、池化配置等。默认路径基于项目根目录，也支持 `.env` 中填写其他相对路径或绝对路径。

仅解析 PDF 不需要 Embedding 模型或 LLM 密钥。

## 4. 建议先单独验证一份扫描件

```powershell
.\.venv\Scripts\python.exe scripts/parse_documents.py --file "data/raw/基准测试报告.pdf"
```

该命令上传指定文件、等待解析、下载并验收结果，不更新向量索引。查看：

- `data/processed/parse_report.json`：本次各文件成功/失败、页数、空文本页、正文块及表格数。
- `data/processed/mineru/<缓存键>/manifest.json`：任务状态、`batch_id`、解析参数和产物校验值。
- 同目录下 `result.<标识>.zip`、`extracted.<标识>/`：保留的原始云端产物和解压结果。
- `documents.json`：统一格式的文本块和物理页码（从 1 开始）。

为保持引用页码准确，适配器优先读取 `layout.json` 中跨页合并前的 `preproc_blocks`，防止跨页段落或表格被统一归到前一页。

适配器当前支持官方云端 `*_content_list.json`（v1 内容列表）和 `layout.json` / `*_middle.json` 的 `pdf_info` 页记录。未知格式、页码缺失、页数不完整会明确失败，保留原始结果供检查，不猜测页码。完整覆盖中存在空文本页会在报告中列出，需人工确认是否为空白页；整份文档无文字时拒绝入库。

## 5. 构建知识库与聊天

```powershell
# 解析 data/raw 中所有 PDF，然后本地向量化
.\.venv\Scripts\python.exe scripts/build_index.py

# 问答
.\.venv\Scripts\python.exe scripts/chat.py
```

默认建库允许部分成功：新文档解析失败时记录为“未入库”，其他成功文档仍可发布；已入库文档更新失败时继续使用旧版并标记为 `stale`。未变化文档复用自己的切块和向量产物，不重新加载 Embedding 计算。每次仍会把全部有效文档组装为完整候选索引，回读验证后才切换活动版本。

```powershell
# 严格模式：data/raw 中任一 PDF 没有处理到最新版本时不发布
.\.venv\Scripts\python.exe scripts/build_index.py --strict

# 显式移除已经不在 data/raw 的旧文档；默认保留并标记 missing_source
.\.venv\Scripts\python.exe scripts/build_index.py --prune-missing
```

查看 `data/processed/build_report.json` 可以区分新增失败、使用旧版、来源缺失、复用和更新数量。解析调试入口 `parse_documents.py` 仍采用严格结果：指定文件中有一份失败就返回非零状态，但成功缓存不会丢失。

聊天输入 `exit`、`quit`、`q`、`退出` 或按 Ctrl+C 退出。当前各轮独立检索，没有多轮历史。

## 缓存与断点恢复

缓存键包含 **PDF 内容 SHA-256 + 解析参数 + 适配版本 + API 地址**，不包含切块参数或密钥。

- 从早期适配版本升级时：按相同文件与解析参数定位旧缓存，校验并复用原始 ZIP 做本地转换，保留原任务 ID，不重复上传。
- 文档没变：复用本地解析结果，校验原始 ZIP 和统一文本的 SHA-256；有完整缓存时无需 MinerU 密钥。
- 切块参数变化：本地重新切块和向量化，不重新上传。
- 文档删除：默认保留上次成功版本并标记 `missing_source`，防止临时移动造成误删；显式使用 `--prune-missing` 才从候选索引移除。
- 程序中断：重新执行相同命令，读取原 `batch_id` 查询进度；上传已被服务端接收时不会新建任务。
- 上传失败：下次先查原任务；只有它还在等文件时才重新上传。
- 查询/下载失败：有限重试，超限后退出；重新运行继续原任务。
- 创建任务超时或提交时被中断：标记 `submission_unknown` / `submitting`，不自动重发。

提交结果不确定、上传地址过期或云端解析确实失败时，先核对账号中的原任务。确定要新建任务后，对一个文件显式执行：

```powershell
.\.venv\Scripts\python.exe scripts/parse_documents.py --file "data/raw/基准测试报告.pdf" --resubmit
```

此操作可能产生重复云端任务；旧任务记录和原始 ZIP 会保留。不要把它作为普通网络重试使用。云端模型更新后如果想刷新全部解析，可主动调整 `MINERU_CACHE_REVISION`；该值变化会产生新缓存键和新任务。

```powershell
# 强制重新切块和向量化，仍复用云端解析缓存
.\.venv\Scripts\python.exe scripts/build_index.py --force
```

旧解析缓存不自动删除。相同 PDF 在云端不同时间解析可能变化，原始结果用于复现；服务未提供具体模型修订号时不声称锁定了云端模型版本。

## 索引发布与保护

新索引写入独立目录，回读验证数量、内容对应关系、维度及有效数值后，原子切换 `vector_db/current.json`：

```text
vector_db/
├── current.json
├── document_artifacts/<产物键>/     # 每份文档可独立复用的本地结果
│   ├── documents.json
│   ├── chunks.json
│   ├── vectors.npy
│   └── manifest.json
├── generations/<版本标识>/          # 每次发布的完整知识库快照
│   ├── index.faiss
│   ├── chunks_meta.json
│   └── build_manifest.json
└── index.faiss / chunks_meta.json   # 升级前旧文件如存在则保留
```

按文档产物键包含 PDF 内容、MinerU 参数、适配版本、切块参数和 Embedding 配置。PDF 新增或修改时只计算受影响文档；切块参数或 Embedding 模型变化时，所有相关产物都会失效并重新生成。`build_manifest.json` 保存当前每份文档的活动内容哈希、产物键和状态。

读取器每次加载固定一个版本，避免新索引配旧元数据。解析、向量化或发布前验证失败时，旧索引不变；以前的版本也不自动删除。首次升级前仍能读取根目录的旧索引，成功建库后以 `current.json` 为准。

`data/processed/chunks.json` 是方便查看的导出，活动索引的权威元数据位于版本目录。不要单独复制某个新 `index.faiss` 到旧目录。

## 安全与失败诊断

- API Token 只发送给固定官方 API，不随上传、下载和 CDN 跳转请求发送；错误文本会过滤 Token 和 URL。
- 上传地址是临时签名地址，任务缓存只保存在已忽略的本地目录，仍应避免分享整个缓存目录。
- ZIP 在写入前校验路径，拒绝目录穿越、绝对路径、符号链接、Windows 特殊路径和重复路径，并限制解压总量及文件数。
- `configuration`：检查 `.env` 的密钥及参数。
- `submission_unknown`：核对云端任务后再决定是否 `--resubmit`。
- `upload_failed`：检查网络或上传地址有效期，下次先查原任务。
- `query_failed` / `wait_timeout`：原任务保留；稍后重跑相同命令。
- `parse_failed`：查看解析报告中的已脱敏错误，再核对原文档。
- `download_failed`：原任务保留，下次重新查询并下载。
- `validation_failed`：查看保留的结果和页码统计；不发布不完整内容。

建库进程和相同文档任务使用操作系统文件锁，进程退出会释放锁；遗留 `.lock` 文件本身不表示任务仍在运行。

## 项目结构与测试

```text
src/
└── rag_agent/
    ├── config.py                       # 路径、Embedding 与 LLM 配置
    ├── ingestion/                      # PDF → 统一文档记录
    │   ├── mineru_settings.py          # 读取共享 .env 中的 MINERU_* 配置
    │   ├── mineru_client.py            # 云端任务、上传、查询、下载和缓存
    │   ├── mineru_adapter.py           # 安全解压、页覆盖和内容适配
    │   └── pypdf_fallback.py           # 无 OCR 的本地备用读取工具
    ├── indexing/                       # 文档 → chunk → 向量 → FAISS
    │   ├── chunker.py
    │   ├── embedder.py
    │   ├── artifact_store.py           # 单文档 documents/chunks/vectors 产物
    │   ├── faiss_store.py              # 版本化完整索引
    │   └── builder.py                  # 部分成功、复用和候选索引发布
    ├── qa/                             # 问题 → 检索 → 回答
    │   ├── retriever.py
    │   ├── generator.py
    │   └── service.py                  # CLI 与未来前端共用的 RAGService
    └── common/
        └── files.py                    # 原子 JSON 写入和进程锁
scripts/
├── parse_documents.py                  # 只解析
├── build_index.py                      # 增量建库
└── chat.py                             # 命令行界面
frontend/
└── README.md                           # 后续网页/桌面界面的边界说明
```

核心包不依赖命令行输入输出。`scripts/chat.py` 和未来前端都通过 `rag_agent.qa.service.RAGService` 调用问答流程；浏览器前端以后可在独立 API 层包装该服务，不应直接读取 FAISS 文件或调用 MinerU。

```powershell
# 离线测试：模拟云端，不使用 API Key、不上传 PDF、不加载真实模型
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

测试覆盖缓存、任务恢复、超时重试、鉴权隔离、ZIP 路径攻击、页覆盖、表格元数据、失败保护和索引切换。真实云端集成仍需填写密钥后验证，不能用模拟通过替代实际 OCR 质量验收。

源码、示例配置和测试可提交；真实环境文件、模型、PDF、解析缓存和索引被 Git 忽略。当前还未加入检索低分过滤、重排序和系统化答案评测。

协议参考：[MinerU 官方云端 API](https://mineru.net/doc/docs/index_en/)。云端调用的账号额度和服务政策以账号页面为准。
