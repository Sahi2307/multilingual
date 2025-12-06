from __future__ import annotations

"""SQLite database utilities for the civic complaint system.

Tables (created on demand):

* users(id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT, email TEXT UNIQUE, phone TEXT,
        role TEXT, created_at TEXT)
* departments(id INTEGER PRIMARY KEY AUTOINCREMENT,
              name TEXT, category TEXT, officials TEXT)
* complaints(id TEXT PRIMARY KEY,
             user_id INTEGER,
             text TEXT,
             category TEXT,
             urgency TEXT,
             language TEXT,
             location TEXT,
             status TEXT,
             department_id INTEGER,
             created_at TEXT,
             updated_at TEXT,
             FOREIGN KEY(user_id) REFERENCES users(id),
             FOREIGN KEY(department_id) REFERENCES departments(id))
* status_updates(id INTEGER PRIMARY KEY AUTOINCREMENT,
                 complaint_id TEXT,
                 status TEXT,
                 remarks TEXT,
                 official_id INTEGER,
                 timestamp TEXT,
                 FOREIGN KEY(complaint_id) REFERENCES complaints(id))
* notifications(id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                message TEXT,
                read INTEGER,
                created_at TEXT,
                FOREIGN KEY(user_id) REFERENCES users(id))
* model_predictions(id INTEGER PRIMARY KEY AUTOINCREMENT,
                    complaint_id TEXT,
                    category_prob TEXT,
                    urgency_prob TEXT,
                    shap_values TEXT,
                    created_at TEXT,
                    FOREIGN KEY(complaint_id) REFERENCES complaints(id))

The schema is intentionally simple and uses TEXT-encoded JSON for
probability vectors and SHAP summaries.
"""

import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_DB_RELATIVE = Path("data") / "civic_system.db"
ENV_DB_PATH = "CIVIC_DB_PATH"


def _resolve_db_path(project_root: Optional[Path] = None) -> Path:
    """Resolve the SQLite database path.

    Order of precedence:
      1. ``CIVIC_DB_PATH`` environment variable (absolute or relative).
      2. ``project_root / data/civic_system.db``.

    Args:
        project_root: Optional project root path.

    Returns:
        Absolute path to the SQLite database file.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parents[1]

    env_path = os.getenv(ENV_DB_PATH)
    if env_path:
        db_path = Path(env_path)
        if not db_path.is_absolute():
            db_path = project_root / db_path
    else:
        db_path = project_root / DEFAULT_DB_RELATIVE

    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


@contextmanager
def get_connection(project_root: Optional[Path] = None) -> Generator[sqlite3.Connection, None, None]:
    """Context manager yielding a SQLite connection.

    The connection uses ``row_factory=sqlite3.Row`` and commits on
    successful exit, rolling back on exceptions.
    """
    db_path = _resolve_db_path(project_root)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception as exc:  # noqa: BLE001
        conn.rollback()
        logger.exception("Database error: %s", exc)
        raise
    finally:
        conn.close()


def init_db(project_root: Optional[Path] = None) -> None:
    """Create all required tables if they do not already exist."""
    with get_connection(project_root) as conn:
        cur = conn.cursor()

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                email TEXT UNIQUE,
                phone TEXT,
                role TEXT,
                created_at TEXT
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS departments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                category TEXT,
                officials TEXT
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS complaints (
                id TEXT PRIMARY KEY,
                user_id INTEGER,
                text TEXT,
                category TEXT,
                urgency TEXT,
                language TEXT,
                location TEXT,
                status TEXT,
                department_id INTEGER,
                created_at TEXT,
                updated_at TEXT,
                FOREIGN KEY(user_id) REFERENCES users(id),
                FOREIGN KEY(department_id) REFERENCES departments(id)
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS status_updates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                complaint_id TEXT,
                status TEXT,
                remarks TEXT,
                official_id INTEGER,
                timestamp TEXT,
                FOREIGN KEY(complaint_id) REFERENCES complaints(id)
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS notifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                message TEXT,
                read INTEGER,
                created_at TEXT,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS model_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                complaint_id TEXT,
                category_prob TEXT,
                urgency_prob TEXT,
                shap_values TEXT,
                created_at TEXT,
                FOREIGN KEY(complaint_id) REFERENCES complaints(id)
            )
            """
        )

        logger.info("Database initialised at %s", _resolve_db_path(project_root))


@dataclass
class User:
    """Simple user representation."""

    id: int
    name: str
    email: str
    phone: str
    role: str
    created_at: str


def get_or_create_user(
    name: str,
    email: str,
    phone: str,
    role: str = "citizen",
    project_root: Optional[Path] = None,
) -> User:
    """Fetch a user by email or create if it does not exist."""
    now = datetime.utcnow().isoformat()
    with get_connection(project_root) as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE email = ?", (email,))
        row = cur.fetchone()
        if row is not None:
            return User(**dict(row))  # type: ignore[arg-type]

        cur.execute(
            "INSERT INTO users(name, email, phone, role, created_at) VALUES(?,?,?,?,?)",
            (name, email, phone, role, now),
        )
        user_id = int(cur.lastrowid)
        return User(id=user_id, name=name, email=email, phone=phone, role=role, created_at=now)


def insert_department(
    name: str,
    category: str,
    officials: Optional[List[int]] = None,
    project_root: Optional[Path] = None,
) -> int:
    """Insert a department and return its ID.

    If a department with the same name and category already exists, its
    ID is returned instead of creating a duplicate.
    """
    with get_connection(project_root) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id FROM departments WHERE name = ? AND category = ?",
            (name, category),
        )
        row = cur.fetchone()
        if row is not None:
            return int(row["id"])

        officials_json = json.dumps(officials or [])
        cur.execute(
            "INSERT INTO departments(name, category, officials) VALUES(?,?,?)",
            (name, category, officials_json),
        )
        return int(cur.lastrowid)


def get_department_id_for_category(
    category: str,
    project_root: Optional[Path] = None,
) -> int:
    """Return department ID for a category, creating a default if needed."""
    default_name = f"Municipal Department - {category}"
    return insert_department(default_name, category, project_root=project_root)


def insert_complaint(
    complaint_id: str,
    user_id: int,
    text: str,
    category: str,
    urgency: str,
    language: str,
    location: str,
    status: str,
    department_id: int,
    project_root: Optional[Path] = None,
) -> None:
    """Insert a complaint record into the database."""
    now = datetime.utcnow().isoformat()
    with get_connection(project_root) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO complaints(
                id, user_id, text, category, urgency, language, location,
                status, department_id, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                complaint_id,
                user_id,
                text,
                category,
                urgency,
                language,
                location,
                status,
                department_id,
                now,
                now,
            ),
        )


def get_complaint(complaint_id: str, project_root: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Fetch a complaint by ID or return ``None`` if not found."""
    with get_connection(project_root) as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM complaints WHERE id = ?", (complaint_id,))
        row = cur.fetchone()
        return dict(row) if row is not None else None


def list_open_complaints_for_department(
    department_id: int,
    project_root: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Return all non-resolved complaints for a department."""
    with get_connection(project_root) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT * FROM complaints
            WHERE department_id = ? AND status != 'Resolved'
            ORDER BY created_at ASC
            """,
            (department_id,),
        )
        rows = cur.fetchall()
        return [dict(r) for r in rows]


def insert_status_update(
    complaint_id: str,
    status: str,
    remarks: str = "",
    official_id: Optional[int] = None,
    project_root: Optional[Path] = None,
) -> None:
    """Insert a new status update for a complaint and update its row."""
    now = datetime.utcnow().isoformat()
    with get_connection(project_root) as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO status_updates(complaint_id, status, remarks, official_id, timestamp) VALUES(?,?,?,?,?)",
            (complaint_id, status, remarks, official_id, now),
        )
        cur.execute(
            "UPDATE complaints SET status = ?, updated_at = ? WHERE id = ?",
            (status, now, complaint_id),
        )


def list_status_updates_for_complaint(
    complaint_id: str,
    project_root: Optional[Path] = None,
) -> list[dict[str, Any]]:
    """Return all status updates for a complaint ordered by time."""
    with get_connection(project_root) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM status_updates WHERE complaint_id = ? ORDER BY timestamp ASC",
            (complaint_id,),
        )
        rows = cur.fetchall()
        return [dict(r) for r in rows]


def insert_notification(
    user_id: int,
    message: str,
    project_root: Optional[Path] = None,
) -> None:
    """Insert an in-app notification for a user."""
    now = datetime.utcnow().isoformat()
    with get_connection(project_root) as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO notifications(user_id, message, read, created_at) VALUES(?,?,?,?)",
            (user_id, message, 0, now),
        )


def list_notifications(
    user_id: int,
    include_read: bool = False,
    project_root: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Return notifications for a user."""
    with get_connection(project_root) as conn:
        cur = conn.cursor()
        if include_read:
            cur.execute("SELECT * FROM notifications WHERE user_id = ? ORDER BY created_at DESC", (user_id,))
        else:
            cur.execute(
                "SELECT * FROM notifications WHERE user_id = ? AND read = 0 ORDER BY created_at DESC",
                (user_id,),
            )
        rows = cur.fetchall()
        return [dict(r) for r in rows]


def mark_notifications_read(user_id: int, project_root: Optional[Path] = None) -> None:
    """Mark all notifications for a user as read."""
    with get_connection(project_root) as conn:
        cur = conn.cursor()
        cur.execute("UPDATE notifications SET read = 1 WHERE user_id = ?", (user_id,))


def insert_model_prediction(
    complaint_id: str,
    category_prob: Dict[str, float],
    urgency_prob: Dict[str, float],
    shap_summary: Dict[str, Any],
    project_root: Optional[Path] = None,
) -> None:
    """Persist model prediction metadata for auditing and analysis."""
    now = datetime.utcnow().isoformat()
    with get_connection(project_root) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO model_predictions(
                complaint_id, category_prob, urgency_prob, shap_values, created_at
            ) VALUES (?,?,?,?,?)
            """,
            (
                complaint_id,
                json.dumps(category_prob),
                json.dumps(urgency_prob),
                json.dumps(shap_summary),
                now,
            ),
        )


__all__ = [
    "init_db",
    "get_or_create_user",
    "get_department_id_for_category",
    "insert_complaint",
    "get_complaint",
    "list_open_complaints_for_department",
    "insert_status_update",
    "list_status_updates_for_complaint",
    "insert_notification",
    "list_notifications",
    "mark_notifications_read",
    "insert_model_prediction",
]
