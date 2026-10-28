"""File-backed store for AI version manifests, state and the audit log (Phase 18).

Layout (``AI_VERSIONS_DIR``)::

    manifests/<version_id>.json   immutable, created with exclusive-create, never rewritten
    state.json                    {revision, active_version_id, versions{id: mutable status data}}
    audit.jsonl                   append-only event log (fsync'd)
    evidence/<version_id>/*.json  Phase 17 reports + functional-test result that justified a verdict
    .lock                         short-lived cross-process lock file

Why files and not a database: a few dozen small records, human-readable, easy to back up with the
existing backup scripts, and no new infrastructure.  Consistency comes from (a) one writer at a time
(thread lock + lock file), (b) atomic ``os.replace`` of ``state.json`` and (c) a monotonically
increasing ``revision`` so a caller holding a stale view is rejected instead of overwriting.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

from app.core.exceptions import AppError, ErrorCode

logger = logging.getLogger(__name__)

LOCK_TIMEOUT_S = 5.0
STALE_LOCK_S = 30.0

_thread_lock = threading.RLock()


def empty_state() -> dict[str, Any]:
    return {"revision": 0, "active_version_id": None, "versions": {}}


class Transaction:
    """Mutable view of ``state.json`` handed out by :meth:`AIVersionRepository.transaction`."""

    def __init__(self, state: dict[str, Any]) -> None:
        self.state = state
        self.audit_events: list[dict[str, Any]] = []

    def audit(self, event: dict[str, Any]) -> None:
        self.audit_events.append(event)


class AIVersionRepository:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.manifests = self.root / "manifests"
        self.state_path = self.root / "state.json"
        self.audit_path = self.root / "audit.jsonl"
        self.evidence_root = self.root / "evidence"
        self.lock_path = self.root / ".lock"

    # -- setup / manifests ----------------------------------------------------------------
    def ensure(self) -> None:
        try:
            self.manifests.mkdir(parents=True, exist_ok=True)
            self.evidence_root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise AppError(ErrorCode.AI_VERSION_STORE_ERROR, "AI version store is not writable.") from exc

    def write_manifest(self, manifest: dict[str, Any]) -> None:
        """Exclusive create: an existing manifest is never overwritten."""
        self.ensure()
        path = self.manifests / f"{manifest['version_id']}.json"
        try:
            with open(path, "x", encoding="utf-8") as fh:
                json.dump(manifest, fh, indent=2, sort_keys=True)
                fh.flush()
                os.fsync(fh.fileno())
        except FileExistsError as exc:
            raise AppError(ErrorCode.AI_VERSION_DUPLICATE, "A manifest with this version id already exists.") from exc
        except OSError as exc:
            raise AppError(ErrorCode.AI_VERSION_STORE_ERROR, "Could not write the version manifest.") from exc

    def read_manifest(self, version_id: str) -> Optional[dict[str, Any]]:
        if not _safe_id(version_id):
            return None
        path = self.manifests / f"{version_id}.json"
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise AppError(ErrorCode.AI_VERSION_STORE_ERROR, f"Manifest {version_id} is unreadable.") from exc

    def manifest_ids(self) -> list[str]:
        if not self.manifests.is_dir():
            return []
        return sorted(p.stem for p in self.manifests.glob("*.json"))

    # -- state ----------------------------------------------------------------------------------
    def read_state(self) -> dict[str, Any]:
        if not self.state_path.is_file():
            return empty_state()
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if not isinstance(state, dict) or "versions" not in state:
                raise ValueError("bad shape")
            state.setdefault("revision", 0)
            state.setdefault("active_version_id", None)
            return state
        except (OSError, ValueError) as exc:
            raise AppError(ErrorCode.AI_VERSION_STORE_ERROR, "AI version state is unreadable.") from exc

    def state_mtime_ns(self) -> int:
        try:
            return self.state_path.stat().st_mtime_ns
        except OSError:
            return 0

    @contextmanager
    def transaction(self, expected_revision: Optional[int] = None) -> Iterator[Transaction]:
        """Serialised read-modify-write. Nothing is persisted if the block raises."""
        self.ensure()
        with _thread_lock, self._file_lock():
            state = self.read_state()
            if expected_revision is not None and state["revision"] != expected_revision:
                raise AppError(ErrorCode.AI_VERSION_CONFLICT,
                               "The version store changed since it was read; retry the operation.")
            tx = Transaction(state)
            yield tx
            # audit first: an event without effect is detectable, an effect without event is not
            for event in tx.audit_events:
                self.append_audit(event)
            state["revision"] = int(state["revision"]) + 1
            tmp = self.state_path.with_suffix(".json.tmp")
            try:
                with open(tmp, "w", encoding="utf-8") as fh:
                    json.dump(state, fh, indent=2, sort_keys=True)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp, self.state_path)
            except OSError as exc:
                raise AppError(ErrorCode.AI_VERSION_STORE_ERROR, "Could not persist the version state.") from exc

    @contextmanager
    def _file_lock(self) -> Iterator[None]:
        deadline = time.monotonic() + LOCK_TIMEOUT_S
        fd: Optional[int] = None
        while fd is None:
            try:
                fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    if time.time() - self.lock_path.stat().st_mtime > STALE_LOCK_S:
                        logger.warning("Removing stale AI version lock file")
                        self.lock_path.unlink(missing_ok=True)
                        continue
                except OSError:
                    pass
                if time.monotonic() > deadline:
                    raise AppError(ErrorCode.AI_VERSION_CONFLICT,
                                   "Another version operation is in progress; retry shortly.")
                time.sleep(0.02)
            except OSError as exc:
                raise AppError(ErrorCode.AI_VERSION_STORE_ERROR, "Could not lock the version store.") from exc
        try:
            yield
        finally:
            os.close(fd)
            try:
                self.lock_path.unlink(missing_ok=True)
            except OSError:
                logger.warning("Could not remove AI version lock file")

    # -- audit ---------------------------------------------------------------------------------------
    def append_audit(self, event: dict[str, Any]) -> None:
        self.ensure()
        try:
            with open(self.audit_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, sort_keys=True, default=str) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
        except OSError as exc:
            raise AppError(ErrorCode.AI_VERSION_STORE_ERROR, "Could not write the audit log.") from exc

    def read_audit(self, limit: int = 100, version_id: Optional[str] = None) -> list[dict[str, Any]]:
        if not self.audit_path.is_file():
            return []
        events: list[dict[str, Any]] = []
        for line in self.audit_path.read_text(encoding="utf-8").splitlines():
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if version_id is None or ev.get("version_id") == version_id:
                events.append(ev)
        return events[-limit:]

    # -- evidence ---------------------------------------------------------------------------------------
    def evidence_dir(self, version_id: str) -> Path:
        if not _safe_id(version_id):
            raise AppError(ErrorCode.AI_VERSION_INVALID, "Invalid version id.")
        return self.evidence_root / version_id


def _safe_id(version_id: str) -> bool:
    return bool(version_id) and all(c.isalnum() or c in "-_." for c in version_id) and ".." not in version_id
