# 离线自动化测试

这里不再划分子目录，按功能保存测试文件。它们检查代码行为，不是 84 道题的真实 RAG 效果评测。

```text
tests/
├── test_mineru.py            # 模拟云端任务、缓存恢复、鉴权与解析验收
├── test_index_publication.py # 向量产物复用、部分成功与索引发布保护
├── test_qa_service.py        # 检索与生成的衔接，以及无资料时的处理
├── test_progress.py         # 进度提示、等待线程退出与错误信息保护
├── test_evaluation.py       # 评测指标、答案隔离、断点恢复与导出
└── test_rag_inspector.py     # 调试记录、邻近片段查询与报告生成
```

## 运行方法

在项目根目录执行：

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

测试通过模拟客户端、假模型和临时目录验证逻辑，不上传真实 PDF、不调用实际模型、不修改正式知识库。
离线测试通过不代表实际 OCR 质量或回答正确率达标；真实效果评测请使用 `scripts/evaluate.py`。
