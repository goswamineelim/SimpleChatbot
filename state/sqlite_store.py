import sqlite3
import uuid
from pathlib import Path

DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "chats.db"

ZERO_USAGE = {
    "input_tokens": 0,
    "output_tokens": 0,
    "total_tokens": 0,
}


class ChatSession:
    """Represents a chat session with heading/title, messages, and token usage."""

    def __init__(
        self,
        chat_id=None,
        title="New Chat",
        messages=None,
        token_usage=None,
        created_at=None,
    ):
        self.id = chat_id or str(uuid.uuid4())[:8]
        self.title = title
        self.messages = list(messages) if messages is not None else []
        self.token_usage = (
            token_usage.copy() if token_usage is not None else ZERO_USAGE.copy()
        )
        self.created_at = created_at

    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "messages": self.messages,
            "token_usage": self.token_usage,
            "created_at": self.created_at,
        }


class SqliteChatStore:
    """Persistent SQLite store supporting multiple chats and history reconstruction."""

    def __init__(self, db_path=None):
        self.db_path = str(DEFAULT_DB_PATH) if db_path is None else str(db_path)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON;")
        self._init_db()

    def _init_db(self):
        with self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS chats (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    running_summary TEXT DEFAULT '',
                    input_tokens INTEGER DEFAULT 0,
                    output_tokens INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(chat_id) REFERENCES chats(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS app_state (
                    key TEXT PRIMARY KEY,
                    value TEXT
                );
                """
            )
            # Migration check: ensure running_summary column exists
            cursor = self._conn.execute("PRAGMA table_info(chats)")
            cols = [row["name"] for row in cursor.fetchall()]
            if "running_summary" not in cols:
                self._conn.execute(
                    "ALTER TABLE chats ADD COLUMN running_summary TEXT DEFAULT ''"
                )

            # Ensure at least one default chat exists
            cursor = self._conn.execute("SELECT COUNT(*) FROM chats")
            if cursor.fetchone()[0] == 0:
                default_id = str(uuid.uuid4())[:8]
                self._conn.execute(
                    "INSERT INTO chats (id, title) VALUES (?, ?)",
                    (default_id, "Chat 1"),
                )
                self._conn.execute(
                    "INSERT OR REPLACE INTO app_state (key, value) VALUES ('active_chat_id', ?)",
                    (default_id,),
                )

    def get_active_chat_id(self) -> str:
        row = self._conn.execute(
            "SELECT value FROM app_state WHERE key = 'active_chat_id'"
        ).fetchone()
        if row:
            active_id = row["value"]
            exists = self._conn.execute(
                "SELECT 1 FROM chats WHERE id = ?", (active_id,)
            ).fetchone()
            if exists:
                return active_id

        # Fallback to the latest updated chat
        chat_row = self._conn.execute(
            "SELECT id FROM chats ORDER BY updated_at DESC LIMIT 1"
        ).fetchone()
        if chat_row:
            active_id = chat_row["id"]
            with self._conn:
                self._conn.execute(
                    "INSERT OR REPLACE INTO app_state (key, value) VALUES ('active_chat_id', ?)",
                    (active_id,),
                )
            return active_id

        # If no chats exist, create one
        new_id = str(uuid.uuid4())[:8]
        with self._conn:
            self._conn.execute(
                "INSERT INTO chats (id, title) VALUES (?, ?)",
                (new_id, "Chat 1"),
            )
            self._conn.execute(
                "INSERT OR REPLACE INTO app_state (key, value) VALUES ('active_chat_id', ?)",
                (new_id,),
            )
        return new_id

    def set_active_chat_id(self, chat_id: str) -> bool:
        exists = self._conn.execute(
            "SELECT 1 FROM chats WHERE id = ?", (chat_id,)
        ).fetchone()
        if exists:
            with self._conn:
                self._conn.execute(
                    "INSERT OR REPLACE INTO app_state (key, value) VALUES ('active_chat_id', ?)",
                    (chat_id,),
                )
            return True
        return False

    def get_all_chats(self) -> list:
        rows = self._conn.execute(
            "SELECT id, title, input_tokens, output_tokens, total_tokens, created_at FROM chats ORDER BY created_at DESC"
        ).fetchall()
        chats = []
        for row in rows:
            chats.append(
                ChatSession(
                    chat_id=row["id"],
                    title=row["title"],
                    token_usage={
                        "input_tokens": row["input_tokens"],
                        "output_tokens": row["output_tokens"],
                        "total_tokens": row["total_tokens"],
                    },
                    created_at=row["created_at"],
                )
            )
        return chats

    def create_chat(self, title=None) -> ChatSession:
        chat_id = str(uuid.uuid4())[:8]
        with self._conn:
            count = self._conn.execute("SELECT COUNT(*) FROM chats").fetchone()[0]
            chat_title = title or f"Chat {count + 1}"
            self._conn.execute(
                "INSERT INTO chats (id, title) VALUES (?, ?)",
                (chat_id, chat_title),
            )
            self._conn.execute(
                "INSERT OR REPLACE INTO app_state (key, value) VALUES ('active_chat_id', ?)",
                (chat_id,),
            )
        return ChatSession(chat_id=chat_id, title=chat_title)

    def switch_chat(self, chat_id: str) -> bool:
        return self.set_active_chat_id(chat_id)

    def delete_chat(self, chat_id: str) -> bool:
        with self._conn:
            self._conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
            remaining = self._conn.execute(
                "SELECT id FROM chats ORDER BY updated_at DESC LIMIT 1"
            ).fetchone()
            if remaining:
                self._conn.execute(
                    "INSERT OR REPLACE INTO app_state (key, value) VALUES ('active_chat_id', ?)",
                    (remaining["id"],),
                )
            else:
                new_id = str(uuid.uuid4())[:8]
                self._conn.execute(
                    "INSERT INTO chats (id, title) VALUES (?, ?)",
                    (new_id, "Chat 1"),
                )
                self._conn.execute(
                    "INSERT OR REPLACE INTO app_state (key, value) VALUES ('active_chat_id', ?)",
                    (new_id,),
                )
        return True

    def get_active_chat(self) -> ChatSession:
        active_id = self.get_active_chat_id()
        row = self._conn.execute(
            "SELECT id, title, input_tokens, output_tokens, total_tokens, created_at FROM chats WHERE id = ?",
            (active_id,),
        ).fetchone()
        messages = self.get_messages(active_id)
        return ChatSession(
            chat_id=row["id"],
            title=row["title"],
            messages=messages,
            token_usage={
                "input_tokens": row["input_tokens"],
                "output_tokens": row["output_tokens"],
                "total_tokens": row["total_tokens"],
            },
            created_at=row["created_at"],
        )

    def get_messages(self, chat_id=None) -> list:
        """Reconstruct the entire chat from the start in chronological order."""
        target_id = chat_id or self.get_active_chat_id()
        rows = self._conn.execute(
            "SELECT role, content FROM messages WHERE chat_id = ? ORDER BY id ASC",
            (target_id,),
        ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in rows]

    def add_message(self, role: str, content: str, chat_id=None) -> dict:
        target_id = chat_id or self.get_active_chat_id()
        with self._conn:
            self._conn.execute(
                "INSERT INTO messages (chat_id, role, content) VALUES (?, ?, ?)",
                (target_id, role, content),
            )
            # Update heading dynamically on first user message if title is default
            chat = self._conn.execute(
                "SELECT title FROM chats WHERE id = ?", (target_id,)
            ).fetchone()
            if chat and (chat["title"].startswith("Chat ") or chat["title"] == "New Chat"):
                count = self._conn.execute(
                    "SELECT COUNT(*) FROM messages WHERE chat_id = ?",
                    (target_id,),
                ).fetchone()[0]
                if count == 1 and role == "user":
                    clean_title = content.strip().replace("\n", " ")[:30]
                    if len(content.strip()) > 30:
                        clean_title += "..."
                    self._conn.execute(
                        "UPDATE chats SET title = ? WHERE id = ?",
                        (clean_title, target_id),
                    )

            self._conn.execute(
                "UPDATE chats SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (target_id,),
            )
        return {"role": role, "content": content}

    def pop_message(self, chat_id=None):
        target_id = chat_id or self.get_active_chat_id()
        with self._conn:
            last_msg = self._conn.execute(
                "SELECT id, role, content FROM messages WHERE chat_id = ? ORDER BY id DESC LIMIT 1",
                (target_id,),
            ).fetchone()
            if last_msg:
                self._conn.execute(
                    "DELETE FROM messages WHERE id = ?", (last_msg["id"],)
                )
                return {"role": last_msg["role"], "content": last_msg["content"]}
        return None

    def get_token_usage(self, chat_id=None) -> dict:
        target_id = chat_id or self.get_active_chat_id()
        row = self._conn.execute(
            "SELECT input_tokens, output_tokens, total_tokens FROM chats WHERE id = ?",
            (target_id,),
        ).fetchone()
        if row:
            return {
                "input_tokens": row["input_tokens"],
                "output_tokens": row["output_tokens"],
                "total_tokens": row["total_tokens"],
            }
        return ZERO_USAGE.copy()

    def set_token_usage(self, usage: dict, chat_id=None):
        target_id = chat_id or self.get_active_chat_id()
        with self._conn:
            self._conn.execute(
                """
                UPDATE chats 
                SET input_tokens = ?, output_tokens = ?, total_tokens = ?, updated_at = CURRENT_TIMESTAMP 
                WHERE id = ?
                """,
                (
                    usage.get("input_tokens", 0),
                    usage.get("output_tokens", 0),
                    usage.get("total_tokens", 0),
                    target_id,
                ),
            )

    def get_running_summary(self, chat_id=None) -> str:
        target_id = chat_id or self.get_active_chat_id()
        row = self._conn.execute(
            "SELECT running_summary FROM chats WHERE id = ?", (target_id,)
        ).fetchone()
        if row and row["running_summary"]:
            return row["running_summary"]
        return ""

    def set_running_summary(self, summary: str, chat_id=None):
        target_id = chat_id or self.get_active_chat_id()
        with self._conn:
            self._conn.execute(
                "UPDATE chats SET running_summary = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (summary, target_id),
            )

    def reset_active_chat(self):
        target_id = self.get_active_chat_id()
        with self._conn:
            self._conn.execute("DELETE FROM messages WHERE chat_id = ?", (target_id,))
            self._conn.execute(
                """
                UPDATE chats 
                SET running_summary = '', input_tokens = 0, output_tokens = 0, total_tokens = 0, updated_at = CURRENT_TIMESTAMP 
                WHERE id = ?
                """,
                (target_id,),
            )
