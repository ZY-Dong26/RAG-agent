# 离线自动化测试

这里不再划分子目录，按功能保存测试文件。它们检查代码行为，不是 84 道题的真实 RAG 效果评测。

```text
tests/
├── test_mineru.py            # 模拟云端任务、缓存恢复、鉴权与解析验收
├── test_postprocessor.py     # 页眉页脚、跨页段落、标题路径、审计与 fail-open
├── test_section_chunker.py   # 章节组合、公式/表格保护、元数据、回退与完整性
├── test_index_publication.py # 向量产物复用、部分成功与索引发布保护
├── test_bm25_store.py       # 中文分词、BM25 持久化与损坏检测
├── test_hybrid_retrieval.py # RRF 去重、单路候选与假 BGE 重排
├── test_qa_service.py       # 证据门控、拒答与生成调用隔离
├── test_retrieval_config.py # 混合检索默认值及模型路径解析
├── test_progress.py         # 进度提示、等待线程退出与错误信息保护
├── test_evaluation.py       # 评测指标、答案隔离、断点恢复与导出
├── test_judge_config.py     # 裁判配置隔离、环境变量优先级与参数校验
├── test_judging.py          # 独立导出、裁判 JSON 校验、逐题写回与续跑
└── test_rag_inspector.py     # 调试记录、邻近片段查询与报告生成
```

## 运行方法

在项目根目录执行：

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

测试通过模拟客户端、假模型和临时目录验证逻辑，不上传真实 PDF、不调用实际模型、不修改正式知识库。
离线测试通过不代表实际 OCR 质量或回答正确率达标；真实效果评测请使用 `scripts/evaluate.py`。
