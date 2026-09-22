# RAG Inspector

仅供开发期查看每轮问答实际发送给云端模型的内容，不被正常 `scripts/chat.py` 导入。

```powershell
# 生成报告，但不自动打开浏览器
.\.venv\Scripts\python.exe scripts\debug_chat.py

# 每轮结束后自动打开 HTML 报告
.\.venv\Scripts\python.exe scripts\debug_chat.py --open
```

每轮结果写入 `debug_runs/<时间_问题>/`：

- `trace.json`：结构化检索结果、实际模型请求、回答和相邻文本块。
- `report.html`：便于人工阅读的本地诊断页。

报告不保存 API Key，但会保存用户问题和知识库原文，包含敏感资料时应及时删除。

删除本功能时移除以下内容即可，不需要改核心模块：

- `devtools/rag_inspector/`
- `scripts/debug_chat.py`
- `tests/test_rag_inspector.py`
- `debug_runs/`
