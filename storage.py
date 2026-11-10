"""Permanent storage for accounts and saved strategies: a private GitHub repository.

Streamlit Cloud wipes the app's disk on every restart, so users.json and everyone's saved
strategies would vanish. When a data repository is configured, this module:

  - on startup, downloads users.json and users/<name>/strategies/*.py into the data dir
    (the first time, when the repository has no users.json yet, it uploads what's on disk);
  - after every change, uploads the changed file as a commit.

The rest of the app keeps reading and writing the data dir as before. Saved backtest runs
are not stored here.

Configure it in the app's secrets (or the BACKTEST_STUDIO_GITHUB_REPO / _TOKEN / _BRANCH
environment variables):

    [storage]
    repo = "your-name/backtest-studio-data"   # must be PRIVATE: it holds password hashes
    token = "github_pat_..."                  # fine-grained token: Contents read and write, this repo only

With nothing configured every function here does nothing, and files stay on the local disk.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://api.github.com"
TIMEOUT = 20
RETRY_SECONDS = 30  # after a failed connection, wait this long before trying again
USERS_FILE = "users.json"
STRATEGY_RE = re.compile(r"^users/[a-z0-9][a-z0-9_.-]{2,31}/strategies/[A-Za-z0-9_-]+\.py$")
REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


class StorageError(RuntimeError):
    """The data repository couldn't be read or written. Never contains the token."""


def _blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()  # noqa: S324 - git's own file id


class GitHubRemote:
    """The few GitHub calls needed: list, read, write and delete files in one repository."""

    def __init__(self, repo: str, token: str, branch: str | None = None):
        repo = repo.strip().removeprefix("https://github.com/").removesuffix(".git").strip("/")
        if not REPO_RE.match(repo):
            raise StorageError('The storage repo must look like "owner/name".')
        self.repo, self._token, self.branch = repo, token.strip(), (branch or "").strip() or None
        self.shas: dict[str, str] = {}  # path -> file id on GitHub, needed to replace or delete it

    def _api(self, method: str, path: str, body: dict | None = None, raw: bool = False) -> tuple[int, bytes]:
        req = urllib.request.Request(
            f"{API}/repos/{self.repo}{path}", method=method,
            data=json.dumps(body).encode("utf-8") if body is not None else None,
            headers={"Authorization": f"Bearer {self._token}", "X-GitHub-Api-Version": "2022-11-28",
                     "Accept": "application/vnd.github.raw+json" if raw else "application/vnd.github+json",
                     "User-Agent": "backtest-studio"})
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:  # noqa: S310 - fixed https URL
                return resp.status, resp.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise StorageError(f"GitHub can't be reached ({getattr(exc, 'reason', exc)}).") from None

    def _fail(self, status: int, raw: bytes, doing: str) -> StorageError:
        try:
            message = json.loads(raw).get("message", "")
        except (ValueError, AttributeError):
            message = ""
        hint = {401: " The token is wrong or has expired.",
                403: " The token needs 'Contents: Read and write' on this repository.",
                404: " Check the repo name, and that the token has access to it."}.get(status, "")
        return StorageError(f"GitHub refused {doing} ({status} {message}).{hint}")

    def _file(self, path: str) -> str:
        return "/contents/" + urllib.parse.quote(path)

    def _ref(self) -> str:
        return f"?ref={urllib.parse.quote(self.branch)}" if self.branch else ""

    def require_private(self) -> None:
        status, raw = self._api("GET", "")
        if status != 200:
            raise self._fail(status, raw, "opening the storage repo")
        if not json.loads(raw).get("private"):
            raise StorageError(f"{self.repo} is public. Accounts are only stored in a PRIVATE repository: "
                               "make it private in its GitHub settings.")

    def list(self) -> dict[str, str]:
        """Every file in the repository: {path: file id}. Empty for a brand-new repository."""
        status, raw = self._api("GET", f"/git/trees/{urllib.parse.quote(self.branch or 'HEAD')}?recursive=1")
        if status in (404, 409):  # no commits yet
            self.shas = {}
            return {}
        if status != 200:
            raise self._fail(status, raw, "listing the storage repo")
        self.shas = {e["path"]: e["sha"] for e in json.loads(raw).get("tree", []) if e.get("type") == "blob"}
        return dict(self.shas)

    def read(self, path: str) -> bytes:
        status, raw = self._api("GET", self._file(path) + self._ref(), raw=True)
        if status != 200:
            raise self._fail(status, raw, f"reading {path}")
        return raw

    def _current_sha(self, path: str) -> str | None:
        status, raw = self._api("GET", self._file(path) + self._ref())
        return json.loads(raw).get("sha") if status == 200 else None

    def write(self, path: str, data: bytes, message: str) -> None:
        if self.shas.get(path) == _blob_sha(data):
            return  # already identical on GitHub: no empty commit
        for attempt in (1, 2):
            body = {"message": message, "content": base64.b64encode(data).decode("ascii")}
            if self.branch:
                body["branch"] = self.branch
            if self.shas.get(path):
                body["sha"] = self.shas[path]
            status, raw = self._api("PUT", self._file(path), body)
            if status in (200, 201):
                self.shas[path] = json.loads(raw)["content"]["sha"]
                return
            if status in (409, 422) and attempt == 1:  # our file id is out of date: fetch it and retry
                sha = self._current_sha(path)
                self.shas.pop(path, None)
                if sha:
                    self.shas[path] = sha
                continue
            raise self._fail(status, raw, f"saving {path}")

    def delete(self, path: str, message: str) -> None:
        sha = self.shas.get(path) or self._current_sha(path)
        if not sha:
            return
        body = {"message": message, "sha": sha}
        if self.branch:
            body["branch"] = self.branch
        status, raw = self._api("DELETE", self._file(path), body)
        if status not in (200, 404):
            raise self._fail(status, raw, f"deleting {path}")
        self.shas.pop(path, None)


# --- the app's one connection ---------------------------------------------------------------
_lock = threading.RLock()
_remote: GitHubRemote | None = None
_base: Path | None = None
_state = "off"   # off: not configured | ok: connected and restored | error: configured but unreachable
_detail = ""
_last_try = 0.0


def status() -> dict:
    """{"state": off / ok / error, "repo": ..., "detail": the problem when state is error}."""
    return {"state": _state, "repo": _remote.repo if _remote else "", "detail": _detail}


def disconnect() -> None:
    global _remote, _base, _state, _detail, _last_try
    with _lock:
        _remote, _base, _state, _detail, _last_try = None, None, "off", "", 0.0


def connect(config, base: Path, make_remote=GitHubRemote) -> dict:
    """Connect to the data repository and restore its files into `base` (once per process;
    after a failure it retries every RETRY_SECONDS). `config` is the [storage] section of the
    app's secrets, or None. Returns status()."""
    global _remote, _base, _state, _detail, _last_try
    get = (lambda k: config.get(k)) if config else (lambda k: None)
    repo = get("repo") or os.environ.get("BACKTEST_STUDIO_GITHUB_REPO")
    token = get("token") or os.environ.get("BACKTEST_STUDIO_GITHUB_TOKEN")
    branch = get("branch") or os.environ.get("BACKTEST_STUDIO_GITHUB_BRANCH")
    base = Path(base).resolve()
    with _lock:
        if not repo or not token:
            if _state != "off":
                disconnect()
            return status()
        if _state == "ok" and _base == base:
            return status()
        if _state == "error" and time.monotonic() - _last_try < RETRY_SECONDS:
            return status()
        _last_try = time.monotonic()
        try:
            remote = make_remote(str(repo), str(token), str(branch) if branch else None)
            _remote = remote
            remote.require_private()
            _restore(remote, base)
            _base, _state, _detail = base, "ok", ""
        except StorageError as exc:
            _state, _detail = "error", str(exc)
        return status()


def _local_files(base: Path) -> list[str]:
    found = [USERS_FILE] if (base / USERS_FILE).exists() else []
    found += sorted(p.relative_to(base).as_posix() for p in base.glob("users/*/strategies/*.py"))
    return [p for p in found if p == USERS_FILE or STRATEGY_RE.match(p)]


def _restore(remote: GitHubRemote, base: Path) -> None:
    files = remote.list()
    if USERS_FILE not in files:  # first use: the repository takes what this disk has
        for rel in _local_files(base):
            remote.write(rel, (base / rel).read_bytes(), "Initial copy from Backtest Studio")
        return
    users = remote.read(USERS_FILE)
    try:
        if not isinstance(json.loads(users).get("users"), dict):
            raise ValueError
    except (ValueError, AttributeError):
        raise StorageError(f"{USERS_FILE} in {remote.repo} isn't a valid accounts file.") from None
    _write_atomic(base / USERS_FILE, users)
    for rel in files:
        if STRATEGY_RE.match(rel):
            _write_atomic(base / rel, remote.read(rel))


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def push(path: Path, base: Path, wait: bool = True) -> None:
    """Copy one changed (or deleted) file under `base` to the data repository.

    wait=True uploads now and raises StorageError if it fails (also when the repository is
    configured but unreachable, so the caller can say the change isn't permanent).
    wait=False uploads in the background and never raises: for changes that don't matter
    much, like the last sign-in time."""
    if _state == "off":
        return
    if _state == "error":
        if wait:
            raise StorageError(_detail)
        return
    try:
        rel = Path(path).resolve().relative_to(_base).as_posix()
    except ValueError:
        return  # not in the data dir this connection restored (e.g. a test's own file)
    if Path(base).resolve() != _base or not (rel == USERS_FILE or STRATEGY_RE.match(rel)):
        return
    if wait:
        _upload(rel)
    else:
        threading.Thread(target=_upload_quietly, args=(rel,), daemon=True).start()


def _upload(rel: str) -> None:
    with _lock:
        if _state != "ok":
            return
        local = _base / rel
        what = "accounts" if rel == USERS_FILE else rel.split("/")[1] + "/" + Path(rel).stem
        if local.exists():  # read inside the lock: the newest contents always win
            _remote.write(rel, local.read_bytes(), f"Update {what}")
        else:
            _remote.delete(rel, f"Delete {what}")


def _upload_quietly(rel: str) -> None:
    try:
        _upload(rel)
    except StorageError:
        pass  # the next change to this file uploads it again
