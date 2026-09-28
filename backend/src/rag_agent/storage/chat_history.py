"""chat_history.py —— 本机聊天记录的 SQLite 数据层

只保存会话元信息、问答正文和当时的回答快照，不保存模型、密钥或原始 PDF。
每次操作使用短连接；问答模型运行期间不打开写事务，成功后才原子写入一组消息。
"""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


DEFAULT_HISTORY_PATH = Path(__file__).resolve().parents[3] / "data" / "chat_history.sqlite3"


def _now() -> str:
    """返回可排序的 UTC 时间文本，供列表按最后更新时间排序。"""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


class ConversationNotFound(Exception):
    """请求中的会话 ID 不存在；API 层将它转换成 HTTP 404。"""


class ChatHistory:
    """负责建表和会话 CRUD；不依赖 RAGService，也不调用云端模型。"""

    def __init__(self, path: Path = DEFAULT_HISTORY_PATH):
        self.path = Path(path)

    @contextmanager
    def _connect(self):
        """每次操作独立连接，提交或回滚后立即关闭，避免长期占用 SQLite 锁。"""
        connection = sqlite3.connect(self.path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self):
        """启动时创建空数据库和两张表；已有记录不会被覆盖。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    text TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    snapshot_json TEXT
                );
                CREATE INDEX IF NOT EXISTS messages_conversation_order
                    ON messages(conversation_id, id);
            """)

    def list_conversations(self) -> list[dict]:
        """返回按最近活动排序的会话摘要，不读取消息正文。"""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, title, created_at, updated_at FROM conversations "
                "ORDER BY updated_at DESC, id DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def exists(self, conversation_id: str) -> bool:
        """在执行耗时问答前检查 ID；不存在时不调用模型。"""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
        return row is not None

    def get_conversation(self, conversation_id: str) -> dict | None:
        """取回一条会话及其原始消息快照；不存在则返回 None。"""
        with self._connect() as connection:
            conversation = connection.execute(
                "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            if conversation is None:
                return None
            rows = connection.execute(
                "SELECT id, role, text, created_at, snapshot_json FROM messages "
                "WHERE conversation_id = ? ORDER BY id", (conversation_id,)
            ).fetchall()
        messages = []
        for row in rows:
            message = {key: row[key] for key in ("id", "role", "text", "created_at")}
            if row["snapshot_json"]:
                message.update(json.loads(row["snapshot_json"]))
            messages.append(message)
        return {**dict(conversation), "messages": messages}

    def save_exchange(self, conversation_id: str | None, question: str, answer: str,
                      snapshot: dict) -> str:
        """一次事务写入用户与助手消息；首轮才创建会话，写失败时整组回滚。"""
        timestamp = _now()
        with self._connect() as connection:
            if conversation_id is None:
                conversation_id = str(uuid4())
                title = question[:29] + ("…" if len(question) > 29 else "")
                connection.execute(
                    "INSERT INTO conversations(id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                    (conversation_id, title, timestamp, timestamp),
                )
            else:
                changed = connection.execute(
                    "UPDATE conversations SET updated_at = ? WHERE id = ?",
                    (timestamp, conversation_id),
                ).rowcount
                if not changed:
                    raise ConversationNotFound(conversation_id)
            connection.execute(
                "INSERT INTO messages(conversation_id, role, text, created_at) VALUES (?, 'user', ?, ?)",
                (conversation_id, question, timestamp),
            )
            connection.execute(
                "INSERT INTO messages(conversation_id, role, text, created_at, snapshot_json) "
                "VALUES (?, 'assistant', ?, ?, ?)",
                (conversation_id, answer, timestamp, json.dumps(snapshot, ensure_ascii=False)),
            )
        return conversation_id

    def delete_conversation(self, conversation_id: str) -> bool:
        """仅级联删除该会话的消息，不接触知识库文件或索引。"""
        with self._connect() as connection:
            changed = connection.execute(
                "DELETE FROM conversations WHERE id = ?", (conversation_id,)
            ).rowcount
        return bool(changed)
