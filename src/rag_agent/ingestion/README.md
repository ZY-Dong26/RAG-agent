# 文档接入与规则后处理

MinerU 文档进入索引前分为三个独立阶段：

1. `mineru_client.py` 获取并缓存云端原始 ZIP；`mineru_adapter.py` 校验页覆盖并转换为项目统一文档记录。
2. `postprocessor.py` 在本地执行确定性的 `rules-v1` 结构后处理，返回 refined document 和审计报告。
3. `indexing/chunker.py` 按连续 `section_path` 切片，并跳过 `excluded_from_retrieval=true` 的块。

原始 ZIP、解压目录和适配后的 `documents.json` 保存在 MinerU 解析缓存中，不会被后处理覆盖。建库成功后，单文档产物目录另外保存 `refined_document.json` 和 `postprocess_report.json`，并把它们的 SHA-256 写入 manifest 验收。

## rules-v1

- 只压缩普通文本空格，不改写表格、公式、图片文字、代码或 LaTeX。
- 结合 bbox 或每页前后两个块，按跨页重复频率标记页眉、页脚和页码；只标记，不删除。
- 只合并相邻页边界上的高置信度正文续接，保留全部来源块 ID 和页码范围。
- 只按明确编号识别标题等级，并用标题栈生成 `section_path`。
- 未知块类型和 MinerU 原始块字段继续保留，并在报告中记录告警。
- 完整性校验或未知异常发生时 fail-open：返回原始文档深拷贝，继续现有切片和建库流程。

V1 不调用 LLM，不合并跨页表格，不猜测无编号标题，并优先避免误合并和误删除。LLM 标题裁决留到 V2。

## section-rules-v1 切块

- 同一文档内只组合连续的相同章节；不足 800 字的小节只向兼容的相邻同级兄弟合并，标题层级上升、主标题变化、参考文献、附录、代码和独立表格会停止合并。
- 普通正文优先按段落、句号和分号递归切分，目标约 1000 字，软上限 1200、硬上限 1500；不能安全合并的短块会在审计报告中说明原因。
- 独立公式和表格带 `atomic=true`。公式绝不截断并尽量携带相邻解释；长表格只按行分片，重复图注和表头，保留统一 `table_id` 与分片序号。V1 不猜测跨页表格连续性。
- chunk 文本带 `章节：...` 前缀；元数据保留 `section_path/section_paths`、`page_start/page_end`、`source_block_ids` 和 `block_types`。builder 随后补充稳定 `segment_index` 和内容哈希 `chunk_id`。
- 单文档 `manifest.json` 的 `chunking_report` 记录原始块数、chunk 长度 min/p50/p95/max、短块原因、超硬上限类型、公式/表格数量以及来源 ID 完整性。
- `SECTION_AWARE_CHUNKING=False` 是旧逐块策略的本地回退开关；两种策略都不调用云端服务。
