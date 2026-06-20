from __future__ import annotations

import base64
import hashlib
import hmac
import re
import secrets
import sqlite3
from typing import Any

import streamlit as st

from app.database import (
    create_user,
    get_user_by_email,
    get_user_by_id,
    update_last_login,
)


PBKDF2_ITERATIONS = 600_000
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _hash_password(password: str, salt: bytes | None = None) -> tuple[str, str]:
    password_salt = salt or secrets.token_bytes(32)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        password_salt,
        PBKDF2_ITERATIONS,
    )
    return (
        base64.b64encode(digest).decode("ascii"),
        base64.b64encode(password_salt).decode("ascii"),
    )


def _verify_password(password: str, expected_hash: str, encoded_salt: str) -> bool:
    try:
        salt = base64.b64decode(encoded_salt.encode("ascii"), validate=True)
    except (ValueError, TypeError):
        return False

    calculated_hash, _ = _hash_password(password, salt=salt)
    return hmac.compare_digest(calculated_hash, expected_hash)


def _public_user(user: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": int(user["id"]),
        "full_name": user["full_name"],
        "email": user["email"],
        "age": user.get("age"),
        "gender": user.get("gender"),
        "created_at": user.get("created_at"),
        "last_login_at": user.get("last_login_at"),
        "profile_image_path": user.get(
        "profile_image_path"
        ),
    }


def _set_authenticated_session(user: dict[str, Any]) -> None:
    st.session_state["authenticated"] = True
    st.session_state["user_id"] = int(user["id"])
    st.session_state["user"] = _public_user(user)


def restore_authenticated_user() -> None:
    if not st.session_state.get("authenticated", False):
        return

    user_id = st.session_state.get("user_id")
    if not user_id:
        logout_user()
        return

    user = get_user_by_id(int(user_id))
    if not user:
        logout_user()
        return

    st.session_state["user"] = _public_user(user)


def create_account(
    *,
    full_name: str,
    email: str,
    password: str,
    confirm_password: str,
    accepted_terms: bool,
) -> tuple[bool, str]:
    clean_name = full_name.strip()
    clean_email = email.strip().lower()

    if len(clean_name) < 2:
        return False, "Enter your full name."

    if not EMAIL_PATTERN.fullmatch(clean_email):
        return False, "Enter a valid email address."

    if len(password) < 8:
        return False, "The password must contain at least 8 characters."

    if not any(character.isalpha() for character in password):
        return False, "The password must contain at least one letter."

    if not any(character.isdigit() for character in password):
        return False, "The password must contain at least one number."

    if password != confirm_password:
        return False, "The two passwords do not match."

    if not accepted_terms:
        return False, "You must accept the responsible-use statement."

    password_hash, password_salt = _hash_password(password)

    try:
        user_id = create_user(
            full_name=clean_name,
            email=clean_email,
            password_hash=password_hash,
            password_salt=password_salt,
        )
    except sqlite3.IntegrityError:
        return False, "An account with this email address already exists."

    user = get_user_by_id(user_id)
    if not user:
        return False, "The account was created, but login failed."

    _set_authenticated_session(user)
    return True, "Account created. Complete your profile next."


def login_user(*, email: str, password: str) -> tuple[bool, str]:
    clean_email = email.strip().lower()

    if not clean_email or not password:
        return False, "Enter both your email and password."

    user = get_user_by_email(clean_email)
    if not user:
        return False, "Incorrect email or password."

    if not _verify_password(
        password,
        user["password_hash"],
        user["password_salt"],
    ):
        return False, "Incorrect email or password."

    update_last_login(int(user["id"]))
    refreshed_user = get_user_by_id(int(user["id"])) or user
    _set_authenticated_session(refreshed_user)
    return True, f"Welcome back, {refreshed_user['full_name']}."


def logout_user() -> None:
    authentication_keys = {
        "authenticated",
        "user_id",
        "user",
        "login_email",
        "login_password",
        "signup_name",
        "signup_email",
        "signup_password",
        "signup_confirm_password",
        "signup_terms",
    }
    for key in authentication_keys:
        st.session_state.pop(key, None)
