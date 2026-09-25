"""
files.py —— 存储辅助层：可靠保存任务进度与索引版本指针
职责：提供原子 JSON 写入、JSON 读取和跨进程文件锁。
为什么 RAG 需要这些工具？
    建库既有耗时的云端任务，也有本地索引。程序中途退出后，需要能恢复任务，
    还要防止读到写了一半的 JSON，或两个建库进程互相覆盖结果。
关键设计：先写临时文件，再替换正式文件；锁由操作系统管理，进程退出会释放。
"""
import json
import os
import uuid
from contextlib import contextmanager
from pathlib import Path


def atomic_json(path, value):
    """
    将对象完整保存为 JSON，再用一次文件替换发布。
    :param path: 最终 JSON 文件路径
    :param value: 可以被 JSON 序列化的字典、列表等对象
    为什么先写临时文件？直接覆盖正式文件时，中途退出可能留下半份 JSON。
    临时文件位于同一目录，使 os.replace 在同一文件系统内切换。
    注意：这是单个文件的原子替换，不是多个文件组成的数据库事务。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 使用随机后缀避免临时文件重名；同目录保证最终替换不跨文件系统。
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        # ensure_ascii=False 保留中文可读性，indent=2 方便学习时直接查看任务日志。
        json.dump(value, stream, ensure_ascii=False, indent=2)
        # 先把 Python 缓冲写给操作系统，再请求同步到磁盘，之后才发布正式文件名。
        stream.flush()
        os.fsync(stream.fileno())
    # 读者看到的是替换前或替换后的完整文件，不会看到正在写入的半份内容。
    os.replace(temporary, path)


def read_json(path):
    """
    按 UTF-8 读取 JSON 并还原为 Python 对象；文件损坏时让异常交给上层处理。
    """
    return json.loads(Path(path).read_text(encoding='utf-8'))


@contextmanager
def exclusive_lock(path):
    """
    独占文件锁：同一时刻只允许一个进程执行 with 代码块。
    :param path: 锁文件路径，例如某个缓存目录的 .lock
    contextmanager 把 yield 前后分成“进入”和“退出”两部分。
    锁状态由操作系统管理：进程退出会释放锁，磁盘上留下 .lock 文件也不代表仍被占用。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        # Windows 按字节区间加锁，因此确保锁文件至少有一个字节，并统一锁住第 0 个字节。
        if path.stat().st_size == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        try:
            # Windows 使用 msvcrt；其他系统使用 fcntl。非阻塞加锁失败时立即提示，不无限等待。
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RuntimeError('另一个进程正在处理相同任务，请等待它结束。') from None
        try:
            # 运行调用方 with 语句里的工作；无论正常结束还是抛异常，finally 都会执行解锁。
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)
