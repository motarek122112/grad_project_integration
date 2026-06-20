from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIRECTORY = PROJECT_ROOT / "data"
DATABASE_PATH = DATA_DIRECTORY / "ishara_users.db"


def utc_now() -> str:
    """Return the current UTC date and time as an ISO-formatted string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def database_connection() -> Iterator[sqlite3.Connection]:
    """
    Open a SQLite database connection.

    The transaction is committed when the operation succeeds and rolled back
    when an exception occurs.
    """
    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(
        DATABASE_PATH,
        timeout=30,
    )

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


def init_database() -> None:
    """
    Create the users table and apply safe database migrations.

    Existing users are preserved when new columns are added.
    """
    with database_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                password_salt TEXT NOT NULL,
                age INTEGER,
                gender TEXT,
                profile_image_path TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_login_at TEXT
            )
            """
        )

        # Detect the columns that already exist in an older database.
        existing_columns = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(users)"
            ).fetchall()
        }

        # Safely upgrade older databases without deleting existing accounts.
        if "profile_image_path" not in existing_columns:
            connection.execute(
                """
                ALTER TABLE users
                ADD COLUMN profile_image_path TEXT
                """
            )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_users_email
            ON users(email)
            """
        )


def create_user(
    *,
    full_name: str,
    email: str,
    password_hash: str,
    password_salt: str,
) -> int:
    """Create a new user and return the generated user ID."""
    now = utc_now()

    with database_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO users (
                full_name,
                email,
                password_hash,
                password_salt,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                full_name.strip(),
                email.strip().lower(),
                password_hash,
                password_salt,
                now,
                now,
            ),
        )

        return int(cursor.lastrowid)


def get_user_by_email(
    email: str,
) -> Optional[dict[str, Any]]:
    """Return a user by email address."""
    with database_connection() as connection:
        row = connection.execute(
            """
            SELECT *
            FROM users
            WHERE email = ?
            """,
            (email.strip().lower(),),
        ).fetchone()

    return dict(row) if row else None


def get_user_by_id(
    user_id: int,
) -> Optional[dict[str, Any]]:
    """Return a user by database ID."""
    with database_connection() as connection:
        row = connection.execute(
            """
            SELECT *
            FROM users
            WHERE id = ?
            """,
            (int(user_id),),
        ).fetchone()

    return dict(row) if row else None


def update_last_login(
    user_id: int,
) -> None:
    """Update the user's most recent login timestamp."""
    now = utc_now()

    with database_connection() as connection:
        connection.execute(
            """
            UPDATE users
            SET
                last_login_at = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                now,
                now,
                int(user_id),
            ),
        )


def update_user_profile(
    *,
    user_id: int,
    full_name: str,
    age: int,
    gender: str,
    profile_image_path: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """
    Update the user's profile.

    When profile_image_path is None, the previously saved profile image is
    preserved.
    """
    normalized_name = full_name.strip()
    normalized_gender = gender.strip()

    if not normalized_name:
        raise ValueError("Full name cannot be empty.")

    if not normalized_gender:
        raise ValueError("Gender cannot be empty.")

    normalized_age = int(age)

    if normalized_age < 1 or normalized_age > 120:
        raise ValueError("Age must be between 1 and 120.")

    with database_connection() as connection:
        cursor = connection.execute(
            """
            UPDATE users
            SET
                full_name = ?,
                age = ?,
                gender = ?,
                profile_image_path = COALESCE(
                    ?,
                    profile_image_path
                ),
                updated_at = ?
            WHERE id = ?
            """,
            (
                normalized_name,
                normalized_age,
                normalized_gender,
                profile_image_path,
                utc_now(),
                int(user_id),
            ),
        )

        if cursor.rowcount == 0:
            return None

    return get_user_by_id(user_id)


def update_profile_image(
    *,
    user_id: int,
    profile_image_path: str,
) -> Optional[dict[str, Any]]:
    """Update only the user's profile-picture path."""
    clean_path = str(profile_image_path).strip()

    if not clean_path:
        raise ValueError("Profile-image path cannot be empty.")

    with database_connection() as connection:
        cursor = connection.execute(
            """
            UPDATE users
            SET
                profile_image_path = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                clean_path,
                utc_now(),
                int(user_id),
            ),
        )

        if cursor.rowcount == 0:
            return None

    return get_user_by_id(user_id)


def remove_profile_image(
    *,
    user_id: int,
) -> Optional[dict[str, Any]]:
    """Remove the stored profile-picture path from a user's profile."""
    with database_connection() as connection:
        cursor = connection.execute(
            """
            UPDATE users
            SET
                profile_image_path = NULL,
                updated_at = ?
            WHERE id = ?
            """,
            (
                utc_now(),
                int(user_id),
            ),
        )

        if cursor.rowcount == 0:
            return None

    return get_user_by_id(user_id)
