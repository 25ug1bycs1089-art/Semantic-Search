import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


def _connect(db_path):
    database = Path(db_path)
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_db(db_path):
    with _connect(db_path) as connection:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                content_hash TEXT NOT NULL UNIQUE,
                indexed_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS passages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                passage_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                location TEXT NOT NULL,
                embedding TEXT NOT NULL,
                UNIQUE(document_id, passage_index)
            );
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                sources TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS passages_document_id_idx
                ON passages(document_id);
            CREATE INDEX IF NOT EXISTS messages_conversation_id_idx
                ON messages(conversation_id, id);
            """
        )


def document_exists(db_path, content_hash):
    with _connect(db_path) as connection:
        row = connection.execute(
            "SELECT 1 FROM documents WHERE content_hash = ?",
            (content_hash,),
        ).fetchone()
    return row is not None


def store_document(db_path, filename, content_hash, passages):
    now = datetime.now(timezone.utc).isoformat()
    document_id = str(uuid.uuid4())
    with _connect(db_path) as connection:
        connection.execute(
            "INSERT INTO documents (id, filename, content_hash, indexed_at) VALUES (?, ?, ?, ?)",
            (document_id, filename, content_hash, now),
        )
        connection.executemany(
            """
            INSERT INTO passages
                (document_id, passage_index, text, location, embedding)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    document_id,
                    index,
                    passage["text"],
                    passage["location"],
                    json.dumps(passage["embedding"]),
                )
                for index, passage in enumerate(passages)
            ],
        )
    return document_id


def list_documents(db_path):
    with _connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT d.id, d.filename, d.indexed_at, COUNT(p.id) AS passage_count
            FROM documents AS d
            LEFT JOIN passages AS p ON p.document_id = d.id
            GROUP BY d.id
            ORDER BY d.indexed_at DESC, d.filename
            """
        ).fetchall()
    return [dict(row) for row in rows]


def delete_document(db_path, document_id):
    with _connect(db_path) as connection:
        cursor = connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))
    return cursor.rowcount > 0


def get_passages(db_path):
    with _connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT p.text, p.location, p.embedding, d.filename
            FROM passages AS p
            JOIN documents AS d ON d.id = p.document_id
            ORDER BY d.filename, p.passage_index
            """
        ).fetchall()
    return [
        {
            "text": row["text"],
            "location": row["location"],
            "embedding": json.loads(row["embedding"]),
            "filename": row["filename"],
        }
        for row in rows
    ]


def list_conversations(db_path):
    with _connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT id, title, created_at, updated_at
            FROM conversations
            ORDER BY updated_at DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def get_messages(db_path, conversation_id):
    with _connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT role, content, sources, created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id
            """,
            (conversation_id,),
        ).fetchall()
    return [
        {
            "role": row["role"],
            "content": row["content"],
            "sources": json.loads(row["sources"]),
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def save_exchange(db_path, conversation_id, question, answer, sources):
    now = datetime.now(timezone.utc).isoformat()
    with _connect(db_path) as connection:
        if conversation_id is None:
            conversation_id = str(uuid.uuid4())
            title = " ".join(question.split())[:72] or "New chat"
            connection.execute(
                """
                INSERT INTO conversations (id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (conversation_id, title, now, now),
            )
        else:
            cursor = connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id),
            )
            if cursor.rowcount == 0:
                raise ValueError("The selected conversation no longer exists.")

        connection.executemany(
            """
            INSERT INTO messages (conversation_id, role, content, sources, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (conversation_id, "user", question, "[]", now),
                (conversation_id, "assistant", answer, json.dumps(sources), now),
            ],
        )
    return conversation_id
