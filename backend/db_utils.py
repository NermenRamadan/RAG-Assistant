"""SQLite persistence for chat history and uploaded-document records.

Kept intentionally simple: two tables, plain sqlite3, one short-lived
connection per call. That's enough for a single-instance graduation
project and avoids adding an ORM.
"""

import logging
import sqlite3
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DB_NAME = "rag_app.db"


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn


def create_application_logs() -> None:
    conn = get_db_connection()
    conn.execute(
        """CREATE TABLE IF NOT EXISTS application_logs (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               session_id TEXT,
               user_query TEXT,
               gpt_response TEXT,
               model TEXT,
               created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
           )"""
    )
    conn.commit()
    conn.close()


def create_document_store() -> None:
    conn = get_db_connection()
    conn.execute(
        """CREATE TABLE IF NOT EXISTS document_store (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               filename TEXT,
               upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
           )"""
    )
    conn.commit()
    conn.close()


def insert_application_logs(session_id: str, user_query: str, answer: str, model: str) -> None:
    conn = get_db_connection()
    conn.execute(
        "INSERT INTO application_logs (session_id, user_query, gpt_response, model) "
        "VALUES (?, ?, ?, ?)",
        (session_id, user_query, answer, model),
    )
    conn.commit()
    conn.close()


def get_chat_history(session_id: str) -> List[Dict[str, str]]:
    """Return prior turns for a session as a flat list of role/content dicts.

    This is passed straight into the LangChain history-aware retriever, which
    accepts (role, content) style messages.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT user_query, gpt_response FROM application_logs "
        "WHERE session_id = ? ORDER BY created_at",
        (session_id,),
    )
    messages: List[Dict[str, str]] = []
    for row in cursor.fetchall():
        messages.append({"role": "human", "content": row["user_query"]})
        messages.append({"role": "ai", "content": row["gpt_response"]})
    conn.close()
    return messages


def insert_document_record(filename: str) -> int:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO document_store (filename) VALUES (?)", (filename,))
    file_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return file_id


def delete_document_record(file_id: int) -> bool:
    conn = get_db_connection()
    conn.execute("DELETE FROM document_store WHERE id = ?", (file_id,))
    conn.commit()
    conn.close()
    return True


def get_all_documents() -> List[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, filename, upload_timestamp FROM document_store "
        "ORDER BY upload_timestamp DESC"
    )
    documents = cursor.fetchall()
    conn.close()
    return [dict(doc) for doc in documents]
