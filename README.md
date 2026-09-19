# RAG-agent

基于本地 Qwen3-Embedding-0.6B、FAISS 和云端大模型的 PDF 知识库命令行问答项目。

流程：PDF → 按页提取文本 → 文本切块 → 本地向量化 → FAISS 检索 → 云端模型生成回答。
目前使用 FAISS，不依赖 Chroma 服务。建库在本地完成；问答时会把问题及检索到的文本发送到配置的云端接口。

## 1. 准备 Python 环境

当前本地运行环境为 Windows、Python 3.12。以下命令在项目根目录的 PowerShell 中执行。

```powershell
# 已有 .venv 时跳过创建
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

直接使用虚拟环境中的 Python，无需激活环境。依赖文件固定了当前本机的直接依赖版本，但不是包含所有间接依赖的完整锁文件，也尚未在全新环境中验证安装。

Embedding 会自动使用可用的 CUDA，否则使用 CPU。当前本机 PyTorch 为 `2.11.0+cu128`；依赖清单仅指定基础版本 `2.11.0`，不保证新环境安装后支持 GPU。需要 GPU 时，应根据目标机器安装相应的 PyTorch 构建。

## 2. 准备本地模型

将完整的 `Qwen3-Embedding-0.6B` 模型目录放到：

```text
model/Qwen3-Embedding-0.6B/
```

需要完整的 Sentence Transformers 模型目录，包括权重、分词器、`modules.json` 和池化配置等，不能只放一个权重文件。模型权重不随 Git 仓库分发，需要另行准备；当前代码要求本地目录，不会把模型名称自动下载为本地模型。

默认路径相对于项目根目录计算，不依赖启动命令时所在的目录。可通过 `.env` 中的 `EMBEDDING_MODEL` 指向其他目录，支持项目相对路径或绝对路径。

## 3. 配置环境变量

已有 `.env` 时保留原文件。首次配置可执行：

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

编辑 `.env`：

```dotenv
LLM_API_KEY=填写你自己的API密钥
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
EMBEDDING_MODEL=model/Qwen3-Embedding-0.6B
```

以上是项目默认配置；接口地址、模型名和密钥必须对应实际使用的服务。系统已有的同名环境变量优先于 `.env`。建库和本地检索不需要 LLM 密钥，云端问答需要。

`.env` 被 Git 忽略，`.env.example` 可以提交，示例文件中不应填写真实密钥。

## 4. 放入 PDF 并构建索引

把 PDF 放入 `data/raw/`。当前只读取该目录直接包含的 PDF，不递归遍历子目录。

```powershell
.\.venv\Scripts\python.exe scripts/build_index.py
```

生成：

- `data/processed/chunks.json`：文本块和来源信息。
- `vector_db/index.faiss`：向量索引。
- `vector_db/chunks_meta.json`：索引对应的文本和元数据。

当前脚本按修改时间复用缓存。删除或替换文件、更换 Embedding 模型、修改切块参数后，请强制重建：

```powershell
.\.venv\Scripts\python.exe scripts/build_index.py --force
```

建库前应确认存在可提取文字的 PDF。目前不支持 OCR，扫描版 PDF 可能无法提取文本；没有提取到文字的文件不会贡献有效文本块。

## 5. 启动问答

完成建库并配置 LLM 密钥后：

```powershell
.\.venv\Scripts\python.exe scripts/chat.py
```

输入问题后显示回答及检索来源。输入 `exit`、`quit`、`q`、`退出` 或按 Ctrl+C 退出。
当前每次提问独立检索和生成，不会把上一轮对话作为历史传入模型。

## 目录结构

```text
RAG-agent/
├── data/
│   ├── raw/                  # 本地原始 PDF
│   └── processed/            # 生成的 chunks.json
├── model/                    # 本地模型权重
├── vector_db/                # FAISS 索引和文本元数据
├── src/
│   ├── __init__.py           # 数字前缀模块的导入映射
│   ├── 00_config.py          # 路径和运行参数
│   ├── 01_document_loader.py
│   ├── 02_text_splitter.py
│   ├── 03_embedding.py
│   ├── 04_vector_store.py
│   ├── 05_retriever.py
│   └── 06_generator.py
├── scripts/
│   ├── build_index.py        # 建库入口
│   └── chat.py               # 问答入口
├── tests/                    # 自动化测试待补充
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## 常见问题与当前限制

- **找不到模型目录**：检查 `EMBEDDING_MODEL` 和完整模型文件是否就绪。
- **找不到索引**：先运行 `scripts/build_index.py`；索引和对应元数据需同时存在。
- **提示缺少 API Key**：检查 `.env` 中的 `LLM_API_KEY`，并确认使用上述虚拟环境运行。
- **放入 PDF 后查不到内容**：确认 PDF 能选中文字；扫描件需先 OCR，再导入。文档发生变化后可使用 `--force` 重建。
- **回答相关性不足**：目前采用固定数量的向量召回，未实现低相关性过滤、重排序及系统化质量评测。显示的来源是检索结果，不保证每条都被回答实际引用。
- **修改检索或切块参数**：在 `src/00_config.py` 调整；模型、切块参数变化后需要重建索引。

## Git 跟踪范围

源码、依赖清单、说明文档、`.env.example` 和目录占位文件可提交。`.env`、虚拟环境、IDE 配置、本地模型、原始 PDF、处理产物及向量索引均被忽略，仍保留在本地磁盘。

忽略规则不影响已经被 Git 跟踪的文件。模型和本地文档需要在新机器上另行准备。
