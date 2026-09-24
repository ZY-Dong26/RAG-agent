# 运行入口与命令速查

```text
scripts/
├── parse_documents.py   # MinerU 解析 + 本地规则后处理，不建向量库
├── build_index.py       # 增量建库：解析 + 切块 + FAISS/BM25 原子发布
├── chat.py              # 用当前索引做命令行问答
├── evaluate.py          # 批量回答评测集并统计检索指标（不调用裁判模型）
├── export_results.py    # 从已有 items 纯本地重新导出全部汇总产物
├── judge.py             # 用独立裁判模型给已有评测批次自动判分
└── debug_chat.py        # 交互问答，导出检索/提示词诊断报告
```

所有命令在项目根目录的 PowerShell 中执行，用项目自己的解释器，无需激活虚拟环境。首次使用顺序：**建库 → 问答 → 按需调试或评测**（单独解析是可选步骤，建库已包含）。各脚本都支持 `--help` 查看完整参数。

## 1. 建库：build_index.py

```powershell
# 常用：增量建库，复用未变化文档的向量，允许部分失败
.\.venv\Scripts\python.exe scripts/build_index.py
```

可选参数（可组合）：

| 参数 | 作用 |
|---|---|
| `--strict` | 任一文档解析失败就不发布新索引 |
| `--force` | 强制重新切块和算向量（仍复用 MinerU 解析缓存，不重新上传 PDF） |
| `--prune-missing` | 从索引移除源 PDF 已缺失的文档（不删磁盘文件） |

索引发布到 `data/vector_db/`，每代同时包含 FAISS、`bm25/`、chunk 元数据和构建清单。
旧代只有 FAISS 时必须重建；未变化文档复用解析与向量产物，本地补建 BM25 不会重新上传 PDF。
报告见 `data/processed/build_report.json`。

## 2. 问答：chat.py

```powershell
.\.venv\Scripts\python.exe scripts/chat.py
```

无参数。输入问题后执行 Dense/BM25 → RRF → BGE 重排 → 证据门控 → 云端生成；
拒答时不会调用回答模型。输入 `exit`/`quit`/`q`/`退出` 或 Ctrl+C 结束。
重新建库后需重启本入口才会加载新索引。

## 3. PDF 解析：parse_documents.py

```powershell
# 解析 data/raw/ 下全部 PDF，复用缓存
.\.venv\Scripts\python.exe scripts/parse_documents.py

# 只处理一份，或强制重新上传（会额外消耗 MinerU 额度）
.\.venv\Scripts\python.exe scripts/parse_documents.py --file "data/raw/xxx.pdf" --resubmit
```

MinerU 原始 ZIP、解压目录和适配缓存保存在 `data/processed/mineru/`；最近一次解析报告是 `data/processed/parse_report.json`，规则后处理报告是 `data/processed/postprocess_report.json`。后处理完全本地，失败时回退原始适配结果。

## 4. 批量评测：evaluate.py

```powershell
# 默认跑全量（84 题），结果写入新的时间戳目录
.\.venv\Scripts\python.exe scripts/evaluate.py

# 先校验题目格式与文档映射，不调用模型、不花费用
.\.venv\Scripts\python.exe scripts/evaluate.py --validate-only
```

常用参数：

| 参数 | 作用 |
|---|---|
| `--output <目录>` | 指定结果目录；目录存在且快照一致时跳过已做题目、续跑剩余；快照（数据集/索引/代码/模型/top_k）变了会拒绝混跑 |
| `--limit N` | 本次最多跑 N 题（试跑用）；续跑时表示"本次再跑 N 题" |
| `--top-k N` | BGE 重排后保留前 N 个片段；改 K 必须换新目录 |
| `--retry-failed` | 重跑状态为 error 的旧题（会再产生费用） |
| `--dataset` / `--source-map` | 换用自定义评测集与文档映射 |

续跑示例（接着上次没跑完的批次）：

```powershell
.\.venv\Scripts\python.exe scripts/evaluate.py --output "data/outputs/evaluation/testdata-baseline-20260923"
```

结果在 `data/outputs/evaluation/<批次>/`：`items/`（逐题权威记录）、`report.md`、`results.csv`、`summary.json` 和 `judge_summary.md`。CSV 末尾四列预留给模型判分；evaluate.py 本身不调用裁判模型。Ctrl+C 中断后可续跑，未保存的那次请求可能已计费。

## 5. 重新导出：export_results.py

```powershell
# 只读取已有 items，不加载模型、不产生 API 费用
.\.venv\Scripts\python.exe scripts/export_results.py "data/outputs/evaluation/<批次>"
```

该命令会重新生成 `results.jsonl`、`results.csv`、`report.md`、`summary.json` 和 `judge_summary.md`。

## 6. 自动判分：judge.py

先在 `.env` 中独立配置 `JUDGE_LLM_BASE_URL`、`JUDGE_LLM_API_KEY` 和 `JUDGE_LLM_MODEL`，再运行：

```powershell
# 判完所有尚未判分的 item；中断后原命令重跑会跳过已判题
.\.venv\Scripts\python.exe scripts/judge.py "data/outputs/evaluation/<批次>"

# 少量试跑；强制重判会再次产生费用
.\.venv\Scripts\python.exe scripts/judge.py "data/outputs/evaluation/<批次>" --limit 3
.\.venv\Scripts\python.exe scripts/judge.py "data/outputs/evaluation/<批次>" --force
```

裁判温度和最大输出长度在 `src/devtools/evaluation/judge_config.py` 中使用代码默认值；脚本只读取已有 item，并把 0–3 分的正确性、完整性、相关性及理由原子写回 `judge` 字段。随后复用同一 exporter 更新 CSV、逐题报告和判分汇总表。

## 7. 调试问答：debug_chat.py

```powershell
.\.venv\Scripts\python.exe scripts/debug_chat.py            # 问答并打印报告路径
.\.venv\Scripts\python.exe scripts/debug_chat.py --open      # 每轮结束自动打开 HTML 报告
```

报告输出到 `data/outputs/debug/`。

## 本地重排与阈值校准

- 默认模型目录为 `model/bge-reranker-v2-m3`，也可在 `.env` 用 `RERANKER_MODEL` 指定绝对或相对路径；程序不会联网下载。
- 临时排障可设置 `RERANK_ENABLED=false`，随后重启脚本。关闭时只按 RRF 排名，不能使用重排分数阈值。
- `RERANK_REJECT_THRESHOLD` 默认在 `src/rag_agent/config.py` 中为 `None`。先用含不可回答问题的评测集观察 Top-1 重排分数，再选择满足目标误拒率/漏拒率的阈值，不要直接采用 0.5。
- 只做离线程序回归可运行：`.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v`。

## PyCharm 运行配置

- 解释器：`.venv\Scripts\python.exe`；工作目录：项目根目录。
- Script path 选要运行的脚本；Parameters 栏只填脚本参数（如 `--limit 3 --output "..."`），普通建库/问答可留空。
