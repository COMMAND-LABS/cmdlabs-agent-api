"""
API key verification utilities.
"""
from passlib.context import CryptContext

# Create bcrypt context for API key hashing
bcrypt_context = CryptContext(schemes=["sha256_crypt"])


def verify_api_key(plaintext_key: str, key_hash: str) -> bool:
    """
    Verify an API key against its hash.

    Args:
        plaintext_key: The plaintext API key to verify
        key_hash: The stored bcrypt hash

    Returns:
        True if the key matches, False otherwise
    """
    try:
        return bcrypt_context.verify(plaintext_key, key_hash)
    except Exception:
        return False
