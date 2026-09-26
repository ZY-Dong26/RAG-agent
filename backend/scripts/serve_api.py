"""
serve_api.py —— 本机 FastAPI 问答服务入口

在 backend/ 目录运行。单进程复用索引与模型，监听本机地址；
如需修改端口，可使用 README 中的 uvicorn 命令。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_agent.common.progress import configure_progress


def main():
    """开启中文启动进度，再启动一个 ASGI worker。"""
    import uvicorn

    configure_progress()
    uvicorn.run("api.app:app", host="127.0.0.1", port=8000, workers=1)


if __name__ == "__main__":
    main()
