"""Accounts, per-user storage and the sign-in screens."""

import os

import pytest

import auth
from auth import AuthError, UserStore


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("BACKTEST_STUDIO_DATA", str(tmp_path))
    return tmp_path


@pytest.fixture
def store(data_dir):
    return UserStore()


def test_passwords_are_hashed_never_stored(store, data_dir):
    store.create_user("alice", "s3cret-pass", "Alice")
    raw = (data_dir / "users.json").read_text()
    assert "s3cret-pass" not in raw and '"hash"' in raw and '"salt"' in raw
    assert store.authenticate("Alice ", "s3cret-pass").username == "alice"  # case / spaces in the username
    with pytest.raises(AuthError, match="Wrong username or password"):
        store.authenticate("alice", "wrong-pass")


def test_same_message_for_unknown_users(store):
    store.create_user("alice", "s3cret-pass")
    with pytest.raises(AuthError) as a:
        store.authenticate("alice", "nope-nope")
    with pytest.raises(AuthError) as b:
        store.authenticate("nobody", "nope-nope")
    assert str(a.value) == str(b.value)


def test_lockout_after_repeated_failures(store, monkeypatch):
    store.create_user("bob", "password123")
    for _ in range(auth.MAX_FAILURES):
        with pytest.raises(AuthError):
            store.authenticate("bob", "wrong-pass")
    with pytest.raises(AuthError, match="locked"):
        store.authenticate("bob", "password123")  # even the right password, while locked
    store.unlock("bob")
    assert store.authenticate("bob", "password123").username == "bob"


def test_validation(store):
    with pytest.raises(AuthError):
        store.create_user("x", "password123")  # too short
    with pytest.raises(AuthError):
        store.create_user("../evil", "password123")
    with pytest.raises(AuthError):
        store.create_user("carol", "short")
    store.create_user("carol", "password123")
    with pytest.raises(AuthError, match="already"):
        store.create_user("Carol", "password123")
    with pytest.raises(AuthError):
        store.create_user("dave", "password123", permissions=["fly"])


def test_last_admin_is_protected(store):
    store.create_user("admin", "password123", role="admin")
    store.create_user("bob", "password123")
    with pytest.raises(AuthError):
        store.delete_user("admin")
    with pytest.raises(AuthError):
        store.update("admin", role="user")
    store.update("bob", role="admin")
    store.update("admin", role="user")  # fine now that bob is an admin
    assert not store.get("admin").is_admin


def test_permissions(store):
    u = store.create_user("bob", "password123")
    assert not u.can("run_code") and not u.can("local_files")
    u = store.update("bob", permissions=["run_code"])
    assert u.can("run_code") and not u.can("local_files")
    admin = store.create_user("root", "password123", role="admin")
    assert admin.can("run_code") and admin.can("local_files")


def test_change_own_password_needs_the_current_one(store):
    store.create_user("bob", "password123")
    with pytest.raises(AuthError):
        store.set_password("bob", "new-password", current_password="wrong-one")
    store.set_password("bob", "new-password", current_password="password123")
    assert store.authenticate("bob", "new-password")


def test_workspaces_are_private(data_dir):
    import workspace as ws
    ws.save_strategy("alice", "My EMA", "def signals(df): ...")
    assert ws.list_strategies("alice") == ["My_EMA"] and ws.list_strategies("bob") == []
    rid = ws.save_run("alice", "Run 1", {"a": 1}, {"Sharpe": "1.2"}, {"trade_log.csv": b"x", "../../evil.csv": b"y"})
    assert {r["id"] for r in ws.list_runs("alice")} == {rid} and ws.list_runs("bob") == []
    assert not (data_dir / "evil.csv").exists()  # file names can't climb out of the folder
    with pytest.raises(ValueError):
        ws.run_file("bob", f"../alice/runs/{rid}", "trade_log.csv")
    ws.delete_run("alice", rid)
    assert ws.list_runs("alice") == []


def test_manage_users_cli(data_dir, monkeypatch, capsys):
    import manage_users
    monkeypatch.setattr(manage_users, "ask_password", lambda: "password123")
    assert manage_users.main(["add", "alice", "--admin"]) == 0
    assert manage_users.main(["add", "bob", "--allow", "run_code"]) == 0
    assert manage_users.main(["list"]) == 0
    out = capsys.readouterr().out
    assert "alice" in out and "admin" in out and "run_code" in out
    assert manage_users.main(["delete", "alice"]) == 1  # the only admin


# --- the app ------------------------------------------------------------------------------------
def _app(data_dir):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    os.environ["BACKTEST_STUDIO_DATA"] = str(data_dir)
    return AppTest.from_file("app.py", default_timeout=300)


def test_first_run_creates_the_admin(data_dir):
    at = _app(data_dir)
    at.run()
    assert any("Create the admin account" in s.value for s in at.subheader)
    at.text_input[0].set_value("boss")
    at.text_input[1].set_value("The Boss")
    at.text_input[2].set_value("password123")
    at.text_input[3].set_value("password123")
    at.button[0].click().run()
    assert not at.exception
    assert UserStore().get("boss").is_admin
    assert any(r.label == "Mode" for r in at.radio)  # signed straight in


def test_sign_in_screen(data_dir):
    UserStore().create_user("bob", "password123")
    at = _app(data_dir)
    at.run()
    assert any("Sign in" in s.value for s in at.subheader) and not at.radio  # nothing else until signed in
    at.text_input[0].set_value("bob")
    at.text_input[1].set_value("wrong-password")
    at.button[0].click().run()
    assert any("Wrong username or password" in e.value for e in at.error)
    at.text_input[1].set_value("password123")
    at.button[0].click().run()
    assert not at.exception and any(r.label == "Mode" for r in at.radio)
    labels = [b.label for b in at.button]
    assert "Saved runs" in labels and "Account" in labels and "Users" not in labels  # not an admin


def test_users_without_permissions_cant_run_code_or_read_paths(data_dir):
    UserStore().create_user("bob", "password123")
    at = _app(data_dir)
    at.session_state["user"] = "bob"
    at.run()
    strategy = next(s for s in at.selectbox if s.label == "Strategy")
    assert "Custom (Python)" not in strategy.options and "QuantConnect algorithm (Python)" not in strategy.options
    assert not any("file paths" in t.label for t in at.text_input)


def test_admin_sees_user_management(data_dir):
    UserStore().create_user("boss", "password123", role="admin")
    at = _app(data_dir)
    at.session_state["user"] = "boss"
    at.run()
    next(b for b in at.button if b.label == "Users").click().run()
    assert not at.exception and any("Manage users" in s.value for s in at.subheader)
    next(b for b in at.button if b.label == "← Back").click().run()
    assert not any("Manage users" in s.value for s in at.subheader)


def test_account_pages_keep_the_backtest_settings(data_dir):
    UserStore().create_user("boss", "password123", role="admin")
    at = _app(data_dir)
    at.session_state["user"] = "boss"
    at.run()
    next(s for s in at.selectbox if s.label == "Strategy").set_value("EMA crossover").run()
    next(b for b in at.button if b.label == "Saved runs").click().run()
    assert any("My saved runs" in s.value for s in at.subheader)
    next(b for b in at.button if b.label == "← Back").click().run()
    assert next(s for s in at.selectbox if s.label == "Strategy").value == "EMA crossover"


def test_save_a_run_and_open_it_again(data_dir, tmp_path):
    from test_frontend import _write_csv, bars
    UserStore().create_user("boss", "password123", role="admin")
    path = _write_csv(tmp_path, "USDJPY.csv", bars(seed=1))
    at = _app(data_dir)
    at.session_state["user"] = "boss"
    at.run()
    at.text_input[0].set_value(path).run()
    at.selectbox[0].set_value("As uploaded").run()
    next(b for b in at.button if b.label == "Run backtest").click().run()
    next(b for b in at.button if b.label == "Save this run").click().run()
    assert any(s.value.startswith("Saved.") for s in at.success)
    import workspace
    runs = workspace.list_runs("boss")
    assert len(runs) == 1 and {"statistics.csv", "trade_log.csv", "daily.csv", "report.html"} <= set(runs[0]["files"])
    next(b for b in at.button if b.label == "Saved runs").click().run()
    assert not at.exception and any("My saved runs" in s.value for s in at.subheader)
    assert not any("Nothing saved yet" in i.value for i in at.info)


def test_admin_comes_from_the_apps_secrets(data_dir):
    at = _app(data_dir)
    at.secrets["admin"] = {"username": "owner", "password": "cloud-password-1", "name": "Owner"}
    at.run()
    assert not at.exception
    owner = UserStore().get("owner")
    assert owner is not None and owner.is_admin
    assert any("Sign in" in s.value for s in at.subheader)  # no open "create the admin" screen
    at.text_input[0].set_value("owner")
    at.text_input[1].set_value("cloud-password-1")
    at.button[0].click().run()
    assert any(r.label == "Mode" for r in at.radio)


def test_bad_admin_secret_is_reported_not_crashing(data_dir):
    at = _app(data_dir)
    at.secrets["admin"] = {"username": "owner", "password": "short"}
    at.run()
    assert not at.exception and any("couldn't be created" in e.value for e in at.error)
