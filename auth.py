"""User accounts for Backtest Studio: usernames, hashed passwords, roles and permissions.

Accounts live in <data dir>/users.json. The data dir is ./app_data next to this file, or
the folder in the BACKTEST_STUDIO_DATA environment variable. When a data repository is
configured (see storage.py) every change is also copied there, so accounts survive restarts. Passwords are never stored:
only a scrypt hash with a random salt per user. After MAX_FAILURES wrong passwords in a row
an account is locked for LOCKOUT_MINUTES.

Permissions (admins have all of them):
  run_code     run custom Python strategies (they execute on the machine hosting the app)
  local_files  read files by path on that machine (data paths, the bundled samples)
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import storage

MAX_FAILURES = 5
LOCKOUT_MINUTES = 5
MIN_PASSWORD = 8
USERNAME_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{2,31}$")
PERMISSIONS = ("run_code", "local_files")
_SCRYPT = dict(n=2 ** 14, r=8, p=1, dklen=32)
_lock = threading.Lock()  # Streamlit serves sessions from threads of one process


def data_dir() -> Path:
    d = Path(os.environ.get("BACKTEST_STUDIO_DATA") or Path(__file__).parent / "app_data")
    d.mkdir(parents=True, exist_ok=True)
    return d


class AuthError(ValueError):
    """A user-facing problem: bad username, weak password, last admin, ..."""


@dataclass
class User:
    username: str
    name: str
    role: str = "user"  # "admin" or "user"
    permissions: list[str] = field(default_factory=list)
    created: str = ""
    last_login: str = ""

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    def can(self, permission: str) -> bool:
        return self.is_admin or permission in self.permissions


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _hash(password: str, salt: bytes) -> str:
    return hashlib.scrypt(password.encode("utf-8"), salt=salt, **_SCRYPT).hex()


def check_username(username: str) -> str:
    u = username.strip().lower()
    if not USERNAME_RE.match(u):
        raise AuthError("Usernames are 3-32 characters: lowercase letters, numbers, '.', '_' or '-', "
                        "starting with a letter or number.")
    return u


def check_password(password: str) -> None:
    if len(password) < MIN_PASSWORD:
        raise AuthError(f"Passwords need at least {MIN_PASSWORD} characters.")
    if password.strip() != password:
        raise AuthError("Passwords can't start or end with a space.")


class UserStore:
    def __init__(self, path: Path | None = None):
        self.path = path or data_dir() / "users.json"

    # --- file ------------------------------------------------------------------------
    def _load(self) -> dict:
        if not self.path.exists():
            return {"users": {}}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self, data: dict, important: bool = True) -> None:
        """important=False: sign-in bookkeeping (last sign-in, wrong-password count); its
        permanent copy is made in the background and a failure there is ignored."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(self.path)  # atomic: a crash never leaves a half-written users file
        try:
            storage.push(self.path, data_dir(), wait=important)
        except storage.StorageError as exc:
            raise AuthError(f"Done, but it couldn't be copied to permanent storage, so it will be lost "
                            f"when the app restarts. {exc}") from None

    @staticmethod
    def _user(username: str, rec: dict) -> User:
        return User(username, rec.get("name") or username, rec.get("role", "user"), list(rec.get("permissions", [])),
                    rec.get("created", ""), rec.get("last_login", ""))

    # --- queries -----------------------------------------------------------------------
    def has_users(self) -> bool:
        return bool(self._load()["users"])

    def list_users(self) -> list[User]:
        return [self._user(u, r) for u, r in sorted(self._load()["users"].items())]

    def get(self, username: str) -> User | None:
        rec = self._load()["users"].get(username.strip().lower())
        return self._user(username.strip().lower(), rec) if rec else None

    # --- changes -------------------------------------------------------------------------
    def create_user(self, username: str, password: str, name: str = "", role: str = "user",
                    permissions: list[str] | None = None) -> User:
        u = check_username(username)
        check_password(password)
        if role not in ("admin", "user"):
            raise AuthError("Role must be admin or user.")
        bad = set(permissions or []) - set(PERMISSIONS)
        if bad:
            raise AuthError(f"Unknown permissions: {sorted(bad)}")
        with _lock:
            data = self._load()
            if u in data["users"]:
                raise AuthError(f"There's already a user called {u}.")
            salt = secrets.token_bytes(16)
            data["users"][u] = {"name": name.strip() or u, "role": role, "permissions": sorted(set(permissions or [])),
                                "salt": salt.hex(), "hash": _hash(password, salt), "created": _now().isoformat(),
                                "last_login": "", "failures": 0, "locked_until": ""}
            self._save(data)
        return self.get(u)

    def authenticate(self, username: str, password: str) -> User:
        """The user if the password is right; AuthError otherwise (same message whether or not the
        username exists, so the form doesn't reveal which accounts exist)."""
        u = username.strip().lower()
        with _lock:
            data = self._load()
            rec = data["users"].get(u)
            if rec is None:
                _hash(password, b"\0" * 16)  # same work as a real check: no timing hint
                raise AuthError("Wrong username or password.")
            locked = rec.get("locked_until")
            if locked and datetime.fromisoformat(locked) > _now():
                mins = max(1, int((datetime.fromisoformat(locked) - _now()).total_seconds() // 60) + 1)
                raise AuthError(f"Too many wrong passwords. This account is locked for about {mins} more minute(s).")
            if not hmac.compare_digest(_hash(password, bytes.fromhex(rec["salt"])), rec["hash"]):
                rec["failures"] = rec.get("failures", 0) + 1
                if rec["failures"] >= MAX_FAILURES:
                    rec["locked_until"] = (_now() + timedelta(minutes=LOCKOUT_MINUTES)).isoformat()
                    rec["failures"] = 0
                self._save(data, important=False)
                raise AuthError("Wrong username or password.")
            rec.update(failures=0, locked_until="", last_login=_now().isoformat())
            self._save(data, important=False)
        return self.get(u)

    def set_password(self, username: str, new_password: str, current_password: str | None = None) -> None:
        """Change a password. Pass current_password when users change their own."""
        check_password(new_password)
        if current_password is not None:
            self.authenticate(username, current_password)
        with _lock:
            data = self._load()
            rec = self._require(data, username)
            salt = secrets.token_bytes(16)
            rec.update(salt=salt.hex(), hash=_hash(new_password, salt), failures=0, locked_until="")
            self._save(data)

    def update(self, username: str, *, name: str | None = None, role: str | None = None,
               permissions: list[str] | None = None) -> User:
        with _lock:
            data = self._load()
            rec = self._require(data, username)
            if role is not None:
                if role not in ("admin", "user"):
                    raise AuthError("Role must be admin or user.")
                if rec.get("role") == "admin" and role != "admin" and self._admins(data) == 1:
                    raise AuthError("This is the only admin. Make someone else an admin first.")
                rec["role"] = role
            if permissions is not None:
                bad = set(permissions) - set(PERMISSIONS)
                if bad:
                    raise AuthError(f"Unknown permissions: {sorted(bad)}")
                rec["permissions"] = sorted(set(permissions))
            if name is not None:
                rec["name"] = name.strip() or username
            self._save(data)
        return self.get(username)

    def unlock(self, username: str) -> None:
        with _lock:
            data = self._load()
            self._require(data, username).update(failures=0, locked_until="")
            self._save(data, important=False)

    def delete_user(self, username: str) -> None:
        with _lock:
            data = self._load()
            rec = self._require(data, username)
            if rec.get("role") == "admin" and self._admins(data) == 1:
                raise AuthError("This is the only admin, so it can't be deleted.")
            del data["users"][username.strip().lower()]
            self._save(data)

    @staticmethod
    def _admins(data: dict) -> int:
        return sum(r.get("role") == "admin" for r in data["users"].values())

    @staticmethod
    def _require(data: dict, username: str) -> dict:
        rec = data["users"].get(username.strip().lower())
        if rec is None:
            raise AuthError(f"No user called {username}.")
        return rec
