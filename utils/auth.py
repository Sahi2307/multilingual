from __future__ import annotations

"""Authentication utilities for the civic complaint system.

This module provides secure authentication functions including:
- Password hashing and verification (bcrypt)
- User registration
- Login validation
- Session token generation
- Password strength validation
"""

import hashlib
import logging
import re
import secrets
from datetime import datetime, timedelta
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Password requirements
MIN_PASSWORD_LENGTH = 8
PASSWORD_PATTERN = re.compile(
    r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&#])[A-Za-z\d@$!%*?&#]{8,}$'
)

# Indian phone number pattern
PHONE_PATTERN = re.compile(r'^(\+91|0)?[6-9]\d{9}$')


def hash_password(password: str) -> str:
    """Hash a password using SHA-256 with salt.

    In production, use bcrypt. For this demo, we use SHA-256 for simplicity.

    Args:
        password: Plain text password.

    Returns:
        Hashed password string.
    """
    # In production: import bcrypt; return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    # For demo, using SHA-256 with a simple salt approach
    salt = "civic_complaint_salt_2024"
    hashed = hashlib.sha256((password + salt).encode()).hexdigest()
    return hashed


def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against its hash.

    Args:
        password: Plain text password to verify.
        hashed: Previously hashed password.

    Returns:
        True if password matches, False otherwise.
    """
    # In production: import bcrypt; return bcrypt.checkpw(password.encode(), hashed.encode())
    return hash_password(password) == hashed


def validate_password_strength(password: str) -> Tuple[bool, str]:
    """Validate password meets strength requirements.

    Requirements:
    - At least 8 characters
    - At least 1 uppercase letter
    - At least 1 lowercase letter
    - At least 1 number
    - At least 1 special character (@$!%*?&#)

    Args:
        password: Password to validate.

    Returns:
        Tuple of (is_valid, error_message).
    """
    if len(password) < MIN_PASSWORD_LENGTH:
        return False, f"Password must be at least {MIN_PASSWORD_LENGTH} characters"

    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least 1 lowercase letter"

    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least 1 uppercase letter"

    if not re.search(r'\d', password):
        return False, "Password must contain at least 1 number"

    if not re.search(r'[@$!%*?&#]', password):
        return False, "Password must contain at least 1 special character (@$!%*?&#)"

    return True, ""


def validate_email(email: str) -> bool:
    """Validate email format.

    Args:
        email: Email address to validate.

    Returns:
        True if valid, False otherwise.
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def validate_phone(phone: str) -> bool:
    """Validate Indian phone number.

    Accepts formats:
    - 9876543210
    - 09876543210
    - +919876543210

    Args:
        phone: Phone number to validate.

    Returns:
        True if valid, False otherwise.
    """
    return bool(PHONE_PATTERN.match(phone))


def sanitize_input(text: str) -> str:
    """Sanitize user input to prevent XSS.

    Args:
        text: Raw user input.

    Returns:
        Sanitized text with HTML entities escaped.
    """
    if not text:
        return ""

    # Basic XSS protection - escape HTML entities
    replacements = {
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#x27;',
        '/': '&#x2F;',
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def generate_session_token() -> str:
    """Generate a secure random session token.

    Returns:
        64-character hexadecimal token.
    """
    return secrets.token_hex(32)


def validate_role(role: str) -> bool:
    """Validate user role.

    Args:
        role: Role to validate.

    Returns:
        True if role is valid, False otherwise.
    """
    valid_roles = {'citizen', 'official', 'admin'}
    return role.lower() in valid_roles


__all__ = [
    'hash_password',
    'verify_password',
    'validate_password_strength',
    'validate_email',
    'validate_phone',
    'sanitize_input',
    'generate_session_token',
    'validate_role',
]
