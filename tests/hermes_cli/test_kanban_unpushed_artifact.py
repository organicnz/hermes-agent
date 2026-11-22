"""``complete_task`` refuses a declared landing that is not on the named ref.

The dangling-42c2b998 class (t_b4510271): a card closed ``done`` with its
commit only on a local worktree; ``git ls-remote origin main`` never resolved
to the landed SHA, the worktree was discarded, and a downstream sync copied
the orphaned commit into a second repository. The completion gate refuses
fail-closed until the artifact is on the named ref of origin.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


def _git(cwd: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True,
    )
    assert proc.returncode == 0, f"git {' '.join(args)} failed: {proc.stderr}"
    return proc.stdout.strip()


@pytest.fixture
def conn(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    with kbc.connect() as c:
        yield c


@pytest.fixture
def landing_repo(tmp_path):
    """A git repo with a bare origin, shaped like the failure: a base commit
    pushed to origin/main, then a guard commit stranded on a discarded
    detached worktree (present in the local object store, absent from
    origin)."""
    origin = tmp_path / "origin.git"
    origin.mkdir()
    _git(origin, "init", "--bare", "-q", "-b", "main")
    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "-q", str(origin), str(clone))
    _git(clone, "config", "user.email", "t@t")
    _git(clone, "config", "user.name", "t")

    def commit(msg: str, path: str = "app.txt") -> str:
        (clone / path).write_text(msg + "\n")
        _git(clone, "add", "-A")
        _git(clone, "commit", "-q", "-m", msg)
        return _git(clone, "rev-parse", "HEAD")

    base = commit("base")
    _git(clone, "push", "-q", "origin", "main")

    # The guard: a commit on a DETACHED worktree, never pushed.
    wt = tmp_path / "wt"
    _git(clone, "worktree", "add", "--detach", "-q", str(wt))
    (wt / "guard.txt").write_text("the guard\n")
    _git(wt, "add", "-A")
    _git(wt, "commit", "-q", "-m", "land the guard")
    guard = _git(wt, "rev-parse", "HEAD")
    # Discard the worktree: the commit is now dangling (local object only).
    _git(clone, "worktree", "remove", "--force", str(wt))

    return {"origin": origin, "clone": clone, "base": base, "guard": guard}


def _ready_task(conn) -> str:
    tid = kb.create_task(conn, title="land the guard", assignee="coder")
    assert kb.claim_task(conn, tid, claimer=kb._claimer_id()) is not None
    return tid


def _landing(repo: Path, ref: str, sha: str) -> dict:
    return {"landed": {"repo": str(repo), "ref": ref, "sha": sha}}


def test_dangling_landing_refused(conn, landing_repo):
    tid = _ready_task(conn)
    with pytest.raises(kb.UnpushedArtifactError):
        kb.complete_task(
            conn, tid, summary="landed the guard",
            metadata=_landing(landing_repo["clone"], "main", landing_repo["guard"]),
        )
    assert conn.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()["status"] == "running"
    rows = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'completion_blocked_unpushed_artifact'",
        (tid,),
    ).fetchall()
    assert len(rows) == 1
    import json
    payload = json.loads(rows[0]["payload"])
    assert payload["reason"] == "sha_not_on_ref"
    assert payload["origin_tip"] == landing_repo["base"]
    assert payload["sha"] == landing_repo["guard"]


def test_pushed_landing_passes(conn, landing_repo):
    tid = _ready_task(conn)
    guard = landing_repo["guard"]
    _git(landing_repo["clone"], "push", "-q", "origin", f"{guard}:main")
    assert kb.complete_task(
        conn, tid, summary="landed the guard (pushed)",
        metadata=_landing(landing_repo["clone"], "main", guard),
    ) is True
    assert conn.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()["status"] == "done"
    assert conn.execute(
        "SELECT COUNT(*) c FROM task_events WHERE task_id = ? AND kind = 'completion_blocked_unpushed_artifact'",
        (tid,),
    ).fetchone()["c"] == 0


def test_landing_sha_ancestor_of_tip_passes(conn, landing_repo):
    """The spec: the ref must resolve to the landed SHA *or an ancestor*."""
    tid = _ready_task(conn)
    guard = landing_repo["guard"]
    _git(landing_repo["clone"], "push", "-q", "origin", f"{guard}:main")
    # Advance the ref past the landed commit.
    _git(landing_repo["clone"], "pull", "-q", "origin", "main")
    (landing_repo["clone"] / "later.txt").write_text("later\n")
    _git(landing_repo["clone"], "add", "-A")
    _git(landing_repo["clone"], "commit", "-q", "-m", "later work")
    _git(landing_repo["clone"], "push", "-q", "origin", "main")
    assert kb.complete_task(
        conn, tid, summary="guard landed; ref advanced past it",
        metadata=_landing(landing_repo["clone"], "main", guard),
    ) is True
    assert conn.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()["status"] == "done"


def test_ref_not_on_origin_refused(conn, landing_repo):
    tid = _ready_task(conn)
    with pytest.raises(kb.UnpushedArtifactError) as exc:
        kb.complete_task(
            conn, tid, summary="landed on develop",
            metadata=_landing(landing_repo["clone"], "develop", landing_repo["base"]),
        )
    assert exc.value.reason == "ref_not_on_origin"
    assert conn.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()["status"] == "running"


def test_repo_not_found_refused(conn, tmp_path):
    tid = _ready_task(conn)
    with pytest.raises(kb.UnpushedArtifactError) as exc:
        kb.complete_task(
            conn, tid, summary="landed",
            metadata=_landing(tmp_path / "does-not-exist", "main", "0" * 40),
        )
    assert exc.value.reason == "repo_not_found"


def test_one_bad_line_in_list_refuses(conn, landing_repo):
    """Every declared landing line must verify — one good + one dangling is a refusal."""
    tid = _ready_task(conn)
    _git(landing_repo["clone"], "push", "-q", "origin", "main")
    with pytest.raises(kb.UnpushedArtifactError) as exc:
        kb.complete_task(
            conn, tid, summary="two landings",
            metadata={"landed": [
                {"repo": str(landing_repo["clone"]), "ref": "main", "sha": landing_repo["base"]},
                {"repo": str(landing_repo["clone"]), "ref": "main", "sha": landing_repo["guard"]},
            ]},
        )
    assert exc.value.reason == "sha_not_on_ref"
    assert conn.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()["status"] == "running"


def test_completion_without_landing_declaration_untouched(conn):
    """The gate is declared-opt-in: a non-landing completion passes."""
    tid = _ready_task(conn)
    assert kb.complete_task(conn, tid, summary="no landing declared") is True
    assert conn.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()["status"] == "done"
