"""Permanent storage (storage.py): accounts and saved strategies survive a wiped disk.
GitHub is replaced by an in-memory fake, so nothing here touches the network."""

import json
import shutil

import pytest

import storage
import workspace
from auth import AuthError, UserStore

CFG = {"repo": "me/data", "token": "secret-token"}


class FakeGitHub:
    """What the data repository holds, shared by every FakeRemote: {path: bytes}."""

    def __init__(self, private=True):
        self.files, self.private, self.down, self.commits = {}, private, False, []

    def remote(self, repo, token, branch=None):
        return FakeRemote(self, repo, token, branch)


class FakeRemote(storage.GitHubRemote):
    def __init__(self, hub, repo, token, branch=None):
        super().__init__(repo, token, branch)
        self.hub = hub

    def _api(self, method, path, body=None, raw=False):
        hub = self.hub
        if hub.down:
            raise storage.StorageError("GitHub can't be reached (test).")
        path = path.split("?")[0]
        if path == "":
            return 200, json.dumps({"private": hub.private}).encode()
        if path.startswith("/git/trees/"):
            if not hub.files:
                return 409, b'{"message": "Git Repository is empty."}'
            tree = [{"path": p, "type": "blob", "sha": storage._blob_sha(d)} for p, d in hub.files.items()]
            return 200, json.dumps({"tree": tree}).encode()
        name = path.removeprefix("/contents/")
        have = hub.files.get(name)
        if method == "GET":
            if have is None:
                return 404, b'{"message": "Not Found"}'
            return 200, have if raw else json.dumps({"sha": storage._blob_sha(have)}).encode()
        if (have is not None) != ("sha" in body) or (have is not None and body["sha"] != storage._blob_sha(have)):
            return 409, b'{"message": "sha does not match"}'
        hub.commits.append(body["message"])
        if method == "DELETE":
            del hub.files[name]
            return 200, b"{}"
        import base64
        data = base64.b64decode(body["content"])
        hub.files[name] = data
        return 201, json.dumps({"content": {"sha": storage._blob_sha(data)}}).encode()


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("BACKTEST_STUDIO_DATA", str(tmp_path / "disk"))
    for var in ("REPO", "TOKEN", "BRANCH"):
        monkeypatch.delenv(f"BACKTEST_STUDIO_GITHUB_{var}", raising=False)
    storage.disconnect()
    yield tmp_path / "disk"
    storage.disconnect()


def restart(data_dir, hub):
    """What a Streamlit Cloud restart does: the disk is wiped, then the app reconnects."""
    storage.disconnect()
    shutil.rmtree(data_dir, ignore_errors=True)
    data_dir.mkdir(parents=True)
    return storage.connect(CFG, data_dir, make_remote=hub.remote)


def test_off_by_default(data_dir):
    assert storage.connect(None, data_dir)["state"] == "off"
    UserStore().create_user("alice", "s3cret-pass")  # works exactly as before
    assert UserStore().get("alice")


def test_users_and_strategies_survive_a_wiped_disk(data_dir):
    hub = FakeGitHub()
    assert restart(data_dir, hub)["state"] == "ok"
    store = UserStore()
    store.create_user("boss", "s3cret-pass", role="admin")
    store.create_user("alice", "s3cret-pass", "Alice", permissions=["run_code"])
    store.set_password("alice", "new-s3cret-pass")
    workspace.save_strategy("alice", "EMA trend v2", "def signals(df):\n    return df.close * 0\n")
    assert "secret-token" not in json.dumps(hub.commits) and "s3cret" not in hub.files["users.json"].decode()

    assert restart(data_dir, hub)["state"] == "ok"
    store = UserStore()
    alice = store.authenticate("alice", "new-s3cret-pass")
    assert alice.name == "Alice" and alice.can("run_code") and store.get("boss").is_admin
    assert workspace.list_strategies("alice") == ["EMA_trend_v2"]
    assert "def signals" in workspace.load_strategy("alice", "EMA trend v2")


def test_deletes_are_permanent_too(data_dir):
    hub = FakeGitHub()
    restart(data_dir, hub)
    store = UserStore()
    store.create_user("boss", "s3cret-pass", role="admin")
    store.create_user("bob", "s3cret-pass")
    workspace.save_strategy("bob", "idea", "x = 1")
    store.delete_user("bob")
    workspace.delete_strategy("bob", "idea")
    restart(data_dir, hub)
    assert UserStore().get("bob") is None and [u.username for u in UserStore().list_users()] == ["boss"]
    assert not any(p.startswith("users/bob/") for p in hub.files)


def test_first_connection_uploads_existing_accounts(data_dir):
    UserStore().create_user("alice", "s3cret-pass")  # made before storage was set up
    workspace.save_strategy("alice", "old", "x = 1")
    hub = FakeGitHub()
    assert storage.connect(CFG, data_dir, make_remote=hub.remote)["state"] == "ok"
    assert set(hub.files) == {"users.json", "users/alice/strategies/old.py"}
    restart(data_dir, hub)
    assert UserStore().authenticate("alice", "s3cret-pass")


def test_public_repo_is_refused(data_dir):
    hub = FakeGitHub(private=False)
    s = restart(data_dir, hub)
    assert s["state"] == "error" and "PRIVATE" in s["detail"] and not hub.files


def test_unreachable_storage_never_overwrites_it_and_says_so(data_dir, monkeypatch):
    hub = FakeGitHub()
    restart(data_dir, hub)
    UserStore().create_user("alice", "s3cret-pass")
    kept = dict(hub.files)

    hub.down = True
    assert restart(data_dir, hub)["state"] == "error"
    with pytest.raises(AuthError, match="permanent storage"):
        UserStore().create_user("temp", "s3cret-pass")
    with pytest.raises(AuthError, match="Wrong username"):  # alice isn't on this disk: plain failure, no crash
        UserStore().authenticate("alice", "s3cret-pass")
    assert hub.files == kept

    hub.down = False
    monkeypatch.setattr(storage, "RETRY_SECONDS", 0)
    assert storage.connect(CFG, data_dir, make_remote=hub.remote)["state"] == "ok"
    assert UserStore().authenticate("alice", "s3cret-pass") and UserStore().get("temp") is None


def test_sign_in_bookkeeping_is_copied_in_the_background(data_dir):
    hub = FakeGitHub()
    restart(data_dir, hub)
    UserStore().create_user("alice", "s3cret-pass")
    UserStore().authenticate("alice", "s3cret-pass")
    import threading
    for t in threading.enumerate():
        if t is not threading.current_thread() and t.daemon:
            t.join(timeout=5)
    assert json.loads(hub.files["users.json"])["users"]["alice"]["last_login"]


def test_only_account_and_strategy_files_are_restored(data_dir):
    hub = FakeGitHub()
    hub.files = {"users.json": b'{"users": {}}', "users/alice/strategies/ok.py": b"x = 1",
                 "users/alice/runs/r1/meta.json": b"{}", "users/../../evil.py": b"bad", "README.md": b"hi"}
    restart(data_dir, hub)
    found = sorted(p.relative_to(data_dir).as_posix() for p in data_dir.rglob("*") if p.is_file())
    assert found == ["users.json", "users/alice/strategies/ok.py"]


def test_broken_accounts_file_is_not_restored(data_dir):
    hub = FakeGitHub()
    hub.files = {"users.json": b"not json"}
    assert restart(data_dir, hub)["state"] == "error" and not (data_dir / "users.json").exists()
