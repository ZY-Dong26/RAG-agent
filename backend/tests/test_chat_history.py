"""SQLite 聊天数据层离线测试：临时文件验证事务回滚与首轮建会话。"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from api.chat_history import ChatHistory


class ChatHistoryTests(unittest.TestCase):
    def test_failed_snapshot_rolls_back_new_conversation_and_both_messages(self):
        """快照不可序列化时，已插入的会话和用户消息也一起回滚。"""
        with tempfile.TemporaryDirectory() as directory:
            history = ChatHistory(Path(directory) / "chat.sqlite3")
            history.initialize()
            with self.assertRaises(TypeError):
                history.save_exchange(None, "测试问题", "回答", {"invalid": object()})
            self.assertEqual(history.list_conversations(), [])

            conversation_id = history.save_exchange(None, "测试问题", "回答", {"answerable": True})
            detail = history.get_conversation(conversation_id)
            self.assertEqual([message["role"] for message in detail["messages"]], ["user", "assistant"])
            self.assertEqual(detail["messages"][1]["answerable"], True)


if __name__ == "__main__":
    unittest.main()
