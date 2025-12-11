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

The schema is intentionally simple and uses TEXT-encoded JSON for
probability vectors and SHAP summaries.
"""

import bcrypt
import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_DB_RELATIVE = Path("data") / "civic_complaints.db"
ENV_DB_PATH = "CIVIC_DB_PATH"

# Database configuration
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'civic_complaints.db')
SCHEMA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'database_schema.sql')


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
    conn.execute("PRAGMA foreign_keys = ON")  # Enable foreign key constraints
    try:
        yield conn
        conn.commit()
    except Exception as exc:  # noqa: BLE001
        conn.rollback()
        logger.exception("Database error: %s", exc)
        raise
    finally:
        conn.close()


def _ensure_default_admin(cur: sqlite3.Cursor) -> None:
    """Ensure a default admin account exists.

    The account uses the credentials specified in the project brief:
    ``admin@civiccomplaints.gov`` / ``Admin@123``. The password is stored
    as a bcrypt hash. In a real deployment this password must be changed
    on first login.
    """
    email = "admin@civiccomplaints.gov"
    cur.execute("SELECT id FROM users WHERE email = ?", (email,))
    if cur.fetchone() is not None:
        return

    now = datetime.utcnow().isoformat()
    password_hash = bcrypt.hashpw("Admin@123".encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    cur.execute(
        """
        INSERT INTO users(name, email, phone, password_hash, role, location, is_active, created_at)
        VALUES(?,?,?,?,?,?,?,?)
        """,
        (
            "System Administrator",
            email,
            "",
            password_hash,
            "admin",
            "System",
            1,
            now,
        ),
    )


def initialize_database() -> None:
    """
    Initialize the database with schema and default data.
    Creates all tables and inserts default admin account and departments.
    """
    try:
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        
        # Read and execute schema
        with open(SCHEMA_PATH, 'r', encoding='utf-8') as f:
            schema_sql = f.read()
        
        with get_connection() as conn:
            cursor = conn.cursor()
            
            # Execute schema (split by semicolons for SQLite compatibility)
            for statement in schema_sql.split(';'):
                if statement.strip():
                    try:
                        cursor.execute(statement)
                    except sqlite3.OperationalError as e:
                        # Skip if table already exists
                        if 'already exists' not in str(e):
                            raise
            
            # Insert default departments
            departments = [
                ('Municipal Corporation - Water Supply', 'Water Supply', None, 'water@municipal.gov', '1800-111-001'),
                ('Municipal Corporation - Sanitation', 'Sanitation', None, 'sanitation@municipal.gov', '1800-111-002'),
                ('Municipal Corporation - Roads & Transport', 'Transportation', None, 'transport@municipal.gov', '1800-111-003')
            ]
            
            for dept in departments:
                cursor.execute("""
                    INSERT OR IGNORE INTO departments (name, category, head_official_id, contact_email, contact_phone)
                    VALUES (?, ?, ?, ?, ?)
                """, dept)
            
            # Create default admin account only if it doesn't exist
            cursor.execute("SELECT id FROM users WHERE email = ?", ('admin@civiccomplaints.gov',))
            if cursor.fetchone() is None:
                admin_password = "Admin@123"
                password_hash = bcrypt.hashpw(admin_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
                
                cursor.execute("""
                    INSERT INTO users (name, email, phone, password_hash, role, location, is_active, is_approved, must_change_password)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    'System Administrator',
                    'admin@civiccomplaints.gov',
                    '+919999999999',
                    password_hash,
                    'admin',
                    'Coimbatore',
                    True,
                    True,
                    True  # Force password change on first login
                ))
                logger.info("Default admin account created: admin@civiccomplaints.gov / Admin@123")
            
            logger.info("Database initialized successfully")
            
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


def init_db(project_root: Optional[Path] = None) -> None:
    """Create all required tables if they do not already exist."""
    with get_connection(project_root) as conn:
        cur = conn.cursor()

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                phone TEXT,
                password_hash TEXT,
                role TEXT NOT NULL DEFAULT 'citizen',
                department_id INTEGER,
                location TEXT,
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                last_login TEXT,
                FOREIGN KEY (department_id) REFERENCES departments(id)
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS departments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                head_official_id INTEGER,
                contact_email TEXT,
                contact_phone TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                officials TEXT,
                FOREIGN KEY (head_official_id) REFERENCES users(id)
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS complaints (
                id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                text TEXT NOT NULL,
                category TEXT NOT NULL,
                urgency TEXT NOT NULL,
                language TEXT NOT NULL,
                location TEXT,
                affected_population TEXT,
                status TEXT DEFAULT 'registered',
                assigned_to INTEGER,
                department_id INTEGER,
                queue_position INTEGER,
                estimated_resolution_date TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                resolved_at TEXT,
                FOREIGN KEY(user_id) REFERENCES users(id),
                FOREIGN KEY(assigned_to) REFERENCES users(id),
                FOREIGN KEY(department_id) REFERENCES departments(id)
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS status_updates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                complaint_id TEXT NOT NULL,
                old_status TEXT,
                status TEXT,
                remarks TEXT,
                official_id INTEGER,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(complaint_id) REFERENCES complaints(id),
                FOREIGN KEY(official_id) REFERENCES users(id)
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS notifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                complaint_id TEXT,
                message TEXT NOT NULL,
                notification_type TEXT,
                read INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id),
                FOREIGN KEY(complaint_id) REFERENCES complaints(id)
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS model_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                complaint_id TEXT NOT NULL,
                category_prob TEXT,
                urgency_prob TEXT,
                shap_values TEXT,
                feature_vector TEXT,
                processing_time REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(complaint_id) REFERENCES complaints(id)
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                session_token TEXT UNIQUE NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                expires_at TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                complaint_id TEXT NOT NULL,
                user_id INTEGER NOT NULL,
                rating INTEGER CHECK(rating >= 1 AND rating <= 5),
                comments TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(complaint_id) REFERENCES complaints(id),
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )

        # Seed a default admin account for first-time setups. In production,
        # the password should be changed immediately after deployment.
        _ensure_default_admin(cur)

        logger.info("Database initialised at %s", _resolve_db_path(project_root))


@dataclass
class User:
    """Simple user representation used in non-auth flows.

    This mirrors a subset of columns from the ``users`` table. The
    authentication subsystem exposes a richer user view including password
    hashes and activation flags.
    """

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
    password: Optional[str] = None,
    project_root: Optional[Path] = None,
) -> User:
    """Fetch a user by email or create if it does not exist.
    
    If password is not provided, a dummy hash is used (for auto-created users).
    """
    now = datetime.utcnow().isoformat()
    with get_connection(project_root) as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE email = ?", (email,))
        row = cur.fetchone()
        if row is not None:
            data = dict(row)
            return User(
                id=int(data["id"]),
                name=str(data.get("name", "")),
                email=str(data.get("email", "")),
                phone=str(data.get("phone", "")),
                role=str(data.get("role", "citizen")),
                created_at=str(data.get("created_at", "")),
            )

        # Generate password hash - use dummy hash for auto-created users
        if password:
            password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        else:
            # Dummy hash for users created without password (e.g., auto-created from complaints)
            password_hash = bcrypt.hashpw(b"dummy_auto_generated", bcrypt.gensalt()).decode()

        cur.execute(
            "INSERT INTO users(name, email, phone, role, password_hash, created_at) VALUES(?,?,?,?,?,?)",
            (name, email, phone, role, password_hash, now),
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

        # Insert department with basic info (officials not stored in schema)
        cur.execute(
            "INSERT INTO departments(name, category) VALUES(?,?)",
            (name, category),
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
                complaint_id, user_id, text, category, urgency, language, location,
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
        # Get the integer id for the complaint from the complaint_id string
        cur.execute("SELECT id FROM complaints WHERE complaint_id = ?", (complaint_id,))
        row = cur.fetchone()
        if not row:
            raise ValueError(f"Complaint {complaint_id} not found")
        complaint_row_id = row[0]
        
        cur.execute(
            "INSERT INTO status_updates(complaint_id, new_status, remarks, updated_by, timestamp) VALUES(?,?,?,?,?)",
            (complaint_row_id, status, remarks, official_id, now),
        )
        cur.execute(
            "UPDATE complaints SET status = ?, updated_at = ? WHERE complaint_id = ?",
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
            "INSERT INTO notifications(user_id, message, is_read, created_at) VALUES(?,?,?,?)",
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
                "SELECT * FROM notifications WHERE user_id = ? AND is_read = 0 ORDER BY created_at DESC",
                (user_id,),
            )
        rows = cur.fetchall()
        return [dict(r) for r in rows]


def mark_notifications_read(user_id: int, project_root: Optional[Path] = None) -> None:
    """Mark all notifications for a user as read."""
    with get_connection(project_root) as conn:
        cur = conn.cursor()
        cur.execute("UPDATE notifications SET is_read = 1 WHERE user_id = ?", (user_id,))


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
        # Get the integer id for the complaint from the complaint_id string
        cur.execute("SELECT id FROM complaints WHERE complaint_id = ?", (complaint_id,))
        row = cur.fetchone()
        if not row:
            raise ValueError(f"Complaint {complaint_id} not found")
        complaint_row_id = row[0]
        
        # Extract category and urgency predictions
        category_predicted = max(category_prob, key=category_prob.get) if category_prob else None
        category_confidence = max(category_prob.values()) if category_prob else 0.0
        urgency_predicted = max(urgency_prob, key=urgency_prob.get) if urgency_prob else None
        urgency_confidence = max(urgency_prob.values()) if urgency_prob else 0.0
        
        cur.execute(
            """
            INSERT INTO model_predictions(
                complaint_id, category_predicted, category_confidence, 
                category_probabilities, urgency_predicted, urgency_confidence, 
                urgency_probabilities, shap_values, timestamp
            ) VALUES (?,?,?,?,?,?,?,?,?)
            """,
            (
                complaint_row_id,
                category_predicted,
                category_confidence,
                json.dumps(category_prob),
                urgency_predicted,
                urgency_confidence,
                json.dumps(urgency_prob),
                json.dumps(shap_summary),
                now,
            ),
        )


def execute_query(query: str, params: Tuple = (), fetch: str = None) -> Optional[List[Dict]]:
    """
    Execute a SQL query with parameters.
    
    Args:
        query (str): SQL query string
        params (Tuple): Query parameters
        fetch (str): Fetch mode - 'one', 'all', or None for non-SELECT queries
    
    Returns:
        Optional[List[Dict]]: Query results as list of dictionaries, or None
    """
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            
            if fetch == 'one':
                row = cursor.fetchone()
                return dict(row) if row else None
            elif fetch == 'all':
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
            else:
                return None
    except Exception as e:
        logger.error(f"Query execution error: {e}")
        raise


def insert_record(table: str, data: Dict[str, Any]) -> Optional[int]:
    """
    Insert a record into specified table.
    
    Args:
        table (str): Table name
        data (Dict): Column-value pairs
    
    Returns:
        Optional[int]: ID of inserted record, or None on failure
    """
    try:
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?' for _ in data])
        query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, tuple(data.values()))
            return cursor.lastrowid
    except Exception as e:
        logger.error(f"Insert error in {table}: {e}")
        raise


def update_record(table: str, record_id: int, data: Dict[str, Any]) -> bool:
    """
    Update a record in specified table.
    
    Args:
        table (str): Table name
        record_id (int): Record ID to update
        data (Dict): Column-value pairs to update
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        set_clause = ', '.join([f"{k} = ?" for k in data.keys()])
        query = f"UPDATE {table} SET {set_clause} WHERE id = ?"
        
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, tuple(data.values()) + (record_id,))
            return cursor.rowcount > 0
    except Exception as e:
        logger.error(f"Update error in {table}: {e}")
        raise


def cleanup_expired_sessions() -> None:
    """Remove expired sessions from database."""
    try:
        query = "DELETE FROM sessions WHERE expires_at < ? OR is_active = FALSE"
        execute_query(query, (datetime.now(),))
        logger.info("Expired sessions cleaned up")
    except Exception as e:
        logger.error(f"Session cleanup error: {e}")


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
