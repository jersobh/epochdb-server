# src/auth.py
import os
import time
import sqlite3
import hashlib
import secrets
from typing import List, Optional, Set, Dict, Any, Tuple
from fastapi import HTTPException, status, Security, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

# Define permission constants
class Permission:
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"

class ScopedAPIKey(BaseModel):
    key_id: str
    permissions: List[str]
    tenant: Optional[str] = None
    namespace: Optional[str] = None
    expires_at: Optional[float] = None
    description: Optional[str] = None
    created_at: float

class KeyStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    key_id TEXT PRIMARY KEY,
                    key_hash TEXT NOT NULL,
                    permissions TEXT NOT NULL,
                    tenant TEXT,
                    namespace TEXT,
                    expires_at REAL,
                    description TEXT,
                    created_at REAL NOT NULL
                )
            """)
            conn.commit()

    def _hash_key(self, raw_key: str) -> str:
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def create_key(
        self,
        permissions: List[str],
        tenant: Optional[str] = None,
        namespace: Optional[str] = None,
        expires_at: Optional[float] = None,
        description: Optional[str] = None
    ) -> Tuple[str, str]:
        # Generate a unique key ID and raw API key
        key_id = f"k_{secrets.token_hex(4)}"
        raw_key = f"epoch_{secrets.token_urlsafe(32)}"
        key_hash = self._hash_key(raw_key)
        created_at = time.time()
        
        perms_str = ",".join(permissions)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO api_keys (key_id, key_hash, permissions, tenant, namespace, expires_at, description, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (key_id, key_hash, perms_str, tenant, namespace, expires_at, description, created_at)
            )
            conn.commit()

        return key_id, raw_key

    def validate_key(self, raw_key: str) -> Optional[ScopedAPIKey]:
        key_hash = self._hash_key(raw_key)
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM api_keys WHERE key_hash = ?",
                (key_hash,)
            )
            row = cursor.fetchone()
            if not row:
                return None
            
            # Check expiry
            expires_at = row["expires_at"]
            if expires_at and expires_at < time.time():
                return None
                
            permissions = [p.strip() for p in row["permissions"].split(",") if p.strip()]
            return ScopedAPIKey(
                key_id=row["key_id"],
                permissions=permissions,
                tenant=row["tenant"],
                namespace=row["namespace"],
                expires_at=expires_at,
                description=row["description"],
                created_at=row["created_at"]
            )

    def revoke_key(self, key_id: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM api_keys WHERE key_id = ?", (key_id,))
            conn.commit()
            return cursor.rowcount > 0

    def list_keys(self) -> List[ScopedAPIKey]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM api_keys ORDER BY created_at DESC")
            rows = cursor.fetchall()
            keys = []
            for row in rows:
                permissions = [p.strip() for p in row["permissions"].split(",") if p.strip()]
                keys.append(ScopedAPIKey(
                    key_id=row["key_id"],
                    permissions=permissions,
                    tenant=row["tenant"],
                    namespace=row["namespace"],
                    expires_at=row["expires_at"],
                    description=row["description"],
                    created_at=row["created_at"]
                ))
            return keys


# Global KeyStore instance helper
_keystore: Optional[KeyStore] = None

def get_keystore() -> KeyStore:
    global _keystore
    if _keystore is None:
        storage_dir = os.getenv("STORAGE_DIR", "./shared_memory")
        db_path = os.path.join(storage_dir, "auth.db")
        _keystore = KeyStore(db_path)
    return _keystore


# Security headers API definitions
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
internal_token_header = APIKeyHeader(name="X-Internal-Token", auto_error=False)

def verify_scoped_auth(required_permissions: Set[str]):
    async def auth_dependency(
        x_api_key: Optional[str] = Security(api_key_header),
        x_internal_token: Optional[str] = Security(internal_token_header)
    ) -> ScopedAPIKey:
        # Import dynamically to support test runtime property mocks
        try:
            import server as server_mod
        except ImportError:
            try:
                import src.server as server_mod
            except ImportError:
                server_mod = None

        legacy_api_key = getattr(server_mod, "API_KEY", None) or os.getenv("API_KEY")
        internal_auth_token = getattr(server_mod, "INTERNAL_AUTH_TOKEN", None) or os.getenv("INTERNAL_AUTH_TOKEN")
        
        token = x_api_key or x_internal_token
        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing auth headers."
            )
            
        # 1. Check legacy admin keys
        if legacy_api_key and token == legacy_api_key:
            return ScopedAPIKey(
                key_id="legacy_admin",
                permissions=[Permission.READ, Permission.WRITE, Permission.DELETE, Permission.ADMIN],
                created_at=0.0
            )
        if internal_auth_token and token == internal_auth_token:
            return ScopedAPIKey(
                key_id="legacy_internal",
                permissions=[Permission.READ, Permission.WRITE, Permission.DELETE, Permission.ADMIN],
                created_at=0.0
            )
            
        # 2. Check keystore keys
        keystore = get_keystore()
        key_info = keystore.validate_key(token)
        if not key_info:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired API Key."
            )
            
        # 3. Check permissions
        # If user has ADMIN permission, they pass any check
        has_admin = Permission.ADMIN in key_info.permissions
        if not has_admin:
            missing = required_permissions - set(key_info.permissions)
            if missing:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Key lacks required permissions: {', '.join(missing)}"
                )
                
        return key_info

    return auth_dependency
