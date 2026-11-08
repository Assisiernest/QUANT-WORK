"""Each user's own space: saved strategies and saved backtest runs.

    <data dir>/users/<username>/strategies/<name>.py
    <data dir>/users/<username>/runs/<YYYYmmdd-HHMMSS>_<title>/meta.json (+ the run's files)

Names are reduced to safe characters and every path is checked to stay inside the
user's folder, so one user can't read or overwrite another's files.
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from auth import check_username, data_dir

MAX_STRATEGY_BYTES = 200_000


def _slug(text: str, limit: int = 60) -> str:
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", text.strip()).strip("_")[:limit]
    if not s:
        raise ValueError("Use at least one letter or number in the name.")
    return s


def user_dir(username: str) -> Path:
    d = data_dir() / "users" / check_username(username)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _inside(base: Path, path: Path) -> Path:
    path = path.resolve()
    if base.resolve() not in path.parents and path != base.resolve():
        raise ValueError("That path is outside your folder.")
    return path


# --- strategies ---------------------------------------------------------------------------
def _strategies(username: str) -> Path:
    d = user_dir(username) / "strategies"
    d.mkdir(exist_ok=True)
    return d


def list_strategies(username: str) -> list[str]:
    return sorted(p.stem for p in _strategies(username).glob("*.py"))


def save_strategy(username: str, name: str, code: str) -> str:
    if len(code.encode("utf-8")) > MAX_STRATEGY_BYTES:
        raise ValueError("That strategy is too long to save (200 KB at most).")
    base = _strategies(username)
    path = _inside(base, base / f"{_slug(name)}.py")
    path.write_text(code, encoding="utf-8")
    return path.stem


def load_strategy(username: str, name: str) -> str:
    base = _strategies(username)
    return _inside(base, base / f"{_slug(name)}.py").read_text(encoding="utf-8")


def delete_strategy(username: str, name: str) -> None:
    base = _strategies(username)
    _inside(base, base / f"{_slug(name)}.py").unlink(missing_ok=True)


# --- runs ----------------------------------------------------------------------------------
def _runs(username: str) -> Path:
    d = user_dir(username) / "runs"
    d.mkdir(exist_ok=True)
    return d


def save_run(username: str, title: str, settings: dict, headline: dict, files: dict[str, bytes]) -> str:
    """Store a run: settings, headline numbers and its files (CSVs, HTML report). Returns its id."""
    stamp = datetime.now(timezone.utc)
    run_id = f"{stamp:%Y%m%d-%H%M%S}_{_slug(title, 40)}"
    base = _runs(username)
    folder = _inside(base, base / run_id)
    folder.mkdir()
    for name, data in files.items():
        suffix = re.sub(r"[^A-Za-z0-9.]", "", Path(name).suffix)[:10]
        _inside(folder, folder / f"{_slug(Path(name).stem)}{suffix}").write_bytes(data)
    meta = {"id": run_id, "title": title, "created": stamp.isoformat(), "settings": settings,
            "headline": headline,
            "files": sorted(f"{_slug(Path(n).stem)}{re.sub(r'[^A-Za-z0-9.]', '', Path(n).suffix)[:10]}" for n in files)}
    (folder / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return run_id


def list_runs(username: str) -> list[dict]:
    runs = []
    for meta in _runs(username).glob("*/meta.json"):
        try:
            runs.append(json.loads(meta.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(runs, key=lambda m: m.get("created", ""), reverse=True)


def run_file(username: str, run_id: str, filename: str) -> bytes:
    base = _runs(username)
    folder = _inside(base, base / run_id)
    return _inside(folder, folder / filename).read_bytes()


def delete_run(username: str, run_id: str) -> None:
    base = _runs(username)
    folder = _inside(base, base / run_id)
    if folder != base.resolve() and folder.exists():
        shutil.rmtree(folder)
