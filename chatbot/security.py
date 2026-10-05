from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
from pwdlib import PasswordHash

from chatbot.settings import S

# Define the hasher that handles the hashing.
password_hash = PasswordHash.recommended()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies if a raw password matches the provided hashed password"""

    return password_hash.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Generates a password hash starting from the provided raw password"""

    return password_hash.hash(password)


def create_access_token(data: dict[str, Any], expires_delta: timedelta | None = None) -> str:
    """Generates a signed JWT access token with an expiration timestamp"""

    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta if expires_delta else timedelta(minutes=S.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, S.SECRET_KEY, algorithm=S.ALGORITHM)
