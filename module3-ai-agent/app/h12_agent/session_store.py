"""SQLite-backed isolated assessment session memory with retention and context window policy (PI 2.3)."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Literal, Optional

from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.core.exceptions import AppError, ErrorCode
from app.h12_agent.lcms_tools import ToolOutcome
from app.h12_agent.types import (
    AgentRole,
    AssessmentMode,
    CallerScope,
    EvidenceRef,
    StoppingReason,
)

SessionStatus = Literal["active", "closed", "escalated"]
SpeakerRole = Literal["student", "agent", "system"]


class SessionTurnRecord(BaseModel):
    """One persisted turn inside an isolated assessment session."""

    turn_index: int
    speaker: SpeakerRole
    message_text: str
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    tool_outcomes: list[ToolOutcome] = Field(default_factory=list)
    created_at: str


class ContextWindowPolicyView(BaseModel):
    """Applied context retention and window truncation parameters."""

    window_turns: int
    max_context_chars: int
    retention_days: int
    expires_at: str
    total_turns: int
    included_turns: int
    summarized_turns: int
    truncated: bool


class SessionContextWindow(BaseModel):
    """Model-facing session memory window isolated to one student/lesson/attempt."""

    session_id: str
    context_summary: str
    recent_turns: list[SessionTurnRecord] = Field(default_factory=list)
    rendered_context: str
    policy: ContextWindowPolicyView


class SessionRecord(BaseModel):
    """Full persisted metadata for one isolated assessment session."""

    session_id: str
    student_id: str
    lesson_id: str
    lesson_version: str
    attempt_id: str
    course_id: Optional[str] = None
    assignment_id: Optional[str] = None
    rubric_id: Optional[str] = None
    assessment_mode: AssessmentMode
    status: SessionStatus
    turn_count: int
    max_turns: int
    stopping_reason: Optional[StoppingReason] = None
    rubric_progress: dict[str, Any] = Field(default_factory=dict)
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    context_summary: str = ""
    retention_days: int
    created_at: str
    updated_at: str
    expires_at: str


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def _summarize_older_turns(older: list[SessionTurnRecord], max_chars: int = 1200) -> str:
    if not older:
        return ""
    lines = [
        f"T{t.turn_index}[{t.speaker}]: {' '.join(t.message_text.split())[:140]}"
        for t in older
    ]
    joined = " | ".join(lines)
    if len(joined) > max_chars:
        return joined[-max_chars:]
    return joined


class SessionMemoryStore:
    """SQLite store enforcing strict isolation by ``(student_id, lesson_id, attempt_id)``."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        settings = get_settings()
        self._db_path = (db_path or settings.agent_session_db_path).strip()
        self._lock = threading.Lock()
        self._memory_conn: Optional[sqlite3.Connection] = None
        if self._db_path == ":memory:":
            self._memory_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._memory_conn.row_factory = sqlite3.Row
            self._memory_conn.execute("PRAGMA foreign_keys = ON")
        else:
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        if self._memory_conn is not None:
            with self._memory_conn:
                yield self._memory_conn
            return

        conn = sqlite3.connect(self._db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def close(self) -> None:
        with self._lock:
            if self._memory_conn is not None:
                self._memory_conn.close()
                self._memory_conn = None

    def _init_schema(self) -> None:
        with self._lock, self._connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS h12_sessions (
                    session_id TEXT PRIMARY KEY,
                    student_id TEXT NOT NULL,
                    lesson_id TEXT NOT NULL,
                    lesson_version TEXT NOT NULL,
                    attempt_id TEXT NOT NULL,
                    course_id TEXT,
                    assignment_id TEXT,
                    rubric_id TEXT,
                    assessment_mode TEXT NOT NULL,
                    status TEXT NOT NULL,
                    turn_count INTEGER NOT NULL,
                    max_turns INTEGER NOT NULL,
                    stopping_reason TEXT,
                    rubric_progress_json TEXT NOT NULL,
                    evidence_refs_json TEXT NOT NULL,
                    context_summary TEXT NOT NULL,
                    retention_days INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    UNIQUE(student_id, lesson_id, attempt_id, assessment_mode)
                );

                CREATE TABLE IF NOT EXISTS h12_session_turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL REFERENCES h12_sessions(session_id) ON DELETE CASCADE,
                    turn_index INTEGER NOT NULL,
                    speaker TEXT NOT NULL,
                    message_text TEXT NOT NULL,
                    evidence_refs_json TEXT NOT NULL,
                    tool_outcomes_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(session_id, turn_index)
                );

                CREATE TABLE IF NOT EXISTS h12_tool_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL,
                    session_id TEXT,
                    agent_role TEXT NOT NULL,
                    caller_scope TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    source_mode TEXT NOT NULL,
                    detail TEXT NOT NULL,
                    duration_ms REAL NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    @staticmethod
    def _row_to_session(row: sqlite3.Row) -> SessionRecord:
        return SessionRecord(
            session_id=row["session_id"],
            student_id=row["student_id"],
            lesson_id=row["lesson_id"],
            lesson_version=row["lesson_version"],
            attempt_id=row["attempt_id"],
            course_id=row["course_id"],
            assignment_id=row["assignment_id"],
            rubric_id=row["rubric_id"],
            assessment_mode=row["assessment_mode"],
            status=row["status"],
            turn_count=row["turn_count"],
            max_turns=row["max_turns"],
            stopping_reason=row["stopping_reason"],
            rubric_progress=json.loads(row["rubric_progress_json"]),
            evidence_refs=[EvidenceRef.model_validate(item) for item in json.loads(row["evidence_refs_json"])],
            context_summary=row["context_summary"],
            retention_days=row["retention_days"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            expires_at=row["expires_at"],
        )

    @staticmethod
    def _row_to_turn(row: sqlite3.Row) -> SessionTurnRecord:
        return SessionTurnRecord(
            turn_index=row["turn_index"],
            speaker=row["speaker"],
            message_text=row["message_text"],
            evidence_refs=[EvidenceRef.model_validate(item) for item in json.loads(row["evidence_refs_json"])],
            tool_outcomes=[ToolOutcome.model_validate(item) for item in json.loads(row["tool_outcomes_json"])],
            created_at=row["created_at"],
        )

    @staticmethod
    def _verify_owner_row(
        row: Optional[sqlite3.Row],
        *,
        session_id: str,
        student_id: str,
        lesson_id: Optional[str] = None,
        attempt_id: Optional[str] = None,
    ) -> sqlite3.Row:
        if row is None:
            raise AppError(ErrorCode.SESSION_NOT_FOUND, f"Session '{session_id}' was not found.")
        if row["student_id"] != student_id:
            raise AppError(
                ErrorCode.AGENT_SCOPE_FORBIDDEN,
                "Cross-student session access is forbidden.",
            )
        if lesson_id is not None and row["lesson_id"] != lesson_id:
            raise AppError(
                ErrorCode.AGENT_SCOPE_FORBIDDEN,
                "Session does not belong to the requested lesson.",
            )
        if attempt_id is not None and row["attempt_id"] != attempt_id:
            raise AppError(
                ErrorCode.AGENT_SCOPE_FORBIDDEN,
                "Session does not belong to the requested attempt.",
            )
        return row

    def create_or_get_session(
        self,
        *,
        student_id: str,
        lesson_id: str,
        lesson_version: str = "v1",
        attempt_id: str,
        assessment_mode: AssessmentMode,
        course_id: Optional[str] = None,
        assignment_id: Optional[str] = None,
        rubric_id: Optional[str] = None,
        max_turns: Optional[int] = None,
        retention_days: Optional[int] = None,
        now_utc: Optional[datetime] = None,
    ) -> tuple[SessionRecord, bool]:
        """Create a new isolated session or return the existing one for the same composite key."""
        if assessment_mode == "classroom_exam":
            raise AppError(
                ErrorCode.ASSESSMENT_MODE_FORBIDDEN,
                "AI assessment sessions cannot be created during classroom examinations.",
            )
        settings = get_settings()
        eff_max_turns = max_turns if max_turns is not None else settings.agent_max_dialogue_turns
        eff_retention = retention_days if retention_days is not None else settings.agent_session_retention_days
        now = now_utc or _utc_now()
        now_iso = _iso(now)
        expires_iso = _iso(now + timedelta(days=eff_retention))

        with self._lock, self._connection() as conn:
            existing = conn.execute(
                """
                SELECT * FROM h12_sessions
                WHERE student_id = ? AND lesson_id = ? AND attempt_id = ? AND assessment_mode = ?
                """,
                (student_id, lesson_id, attempt_id, assessment_mode),
            ).fetchone()
            if existing is not None:
                return self._row_to_session(existing), False

            session_id = f"sess-{uuid.uuid4().hex[:16]}"
            conn.execute(
                """
                INSERT INTO h12_sessions (
                    session_id, student_id, lesson_id, lesson_version, attempt_id,
                    course_id, assignment_id, rubric_id, assessment_mode, status,
                    turn_count, max_turns, stopping_reason, rubric_progress_json,
                    evidence_refs_json, context_summary, retention_days,
                    created_at, updated_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', 0, ?, NULL, '{}', '[]', '', ?, ?, ?, ?)
                """,
                (
                    session_id,
                    student_id,
                    lesson_id,
                    lesson_version,
                    attempt_id,
                    course_id,
                    assignment_id,
                    rubric_id,
                    assessment_mode,
                    eff_max_turns,
                    eff_retention,
                    now_iso,
                    now_iso,
                    expires_iso,
                ),
            )
            created_row = conn.execute(
                "SELECT * FROM h12_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            assert created_row is not None
            return self._row_to_session(created_row), True

    def get_session_for_owner(
        self,
        session_id: str,
        *,
        student_id: str,
        lesson_id: Optional[str] = None,
        attempt_id: Optional[str] = None,
    ) -> SessionRecord:
        with self._lock, self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM h12_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            verified = self._verify_owner_row(
                row,
                session_id=session_id,
                student_id=student_id,
                lesson_id=lesson_id,
                attempt_id=attempt_id,
            )
            return self._row_to_session(verified)

    def append_turn(
        self,
        session_id: str,
        *,
        student_id: str,
        speaker: SpeakerRole,
        message_text: str,
        evidence_refs: Optional[list[EvidenceRef]] = None,
        tool_outcomes: Optional[list[ToolOutcome]] = None,
        window_turns: Optional[int] = None,
        now_utc: Optional[datetime] = None,
    ) -> SessionTurnRecord:
        """Append one turn to an active session, updating turn count, evidence and summary."""
        clean_msg = (message_text or "").strip()
        if not clean_msg:
            raise AppError(ErrorCode.VALIDATION_ERROR, "message_text must not be empty.", status_code=422)

        settings = get_settings()
        eff_window = window_turns if window_turns is not None else settings.agent_context_window_turns
        now = now_utc or _utc_now()
        now_iso = _iso(now)
        new_refs = evidence_refs or []
        outcomes = tool_outcomes or []

        with self._lock, self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM h12_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            verified = self._verify_owner_row(row, session_id=session_id, student_id=student_id)
            if verified["status"] != "active":
                raise AppError(
                    ErrorCode.SESSION_STATE_CONFLICT,
                    f"Session '{session_id}' is already {verified['status']} and cannot accept new turns.",
                )
            if speaker == "student" and verified["turn_count"] >= verified["max_turns"]:
                raise AppError(
                    ErrorCode.SESSION_STATE_CONFLICT,
                    f"Session '{session_id}' reached max_turns ({verified['max_turns']}).",
                )

            max_idx_row = conn.execute(
                "SELECT COALESCE(MAX(turn_index), 0) AS max_idx FROM h12_session_turns WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            next_idx = int(max_idx_row["max_idx"]) + 1

            conn.execute(
                """
                INSERT INTO h12_session_turns (
                    session_id, turn_index, speaker, message_text,
                    evidence_refs_json, tool_outcomes_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    next_idx,
                    speaker,
                    clean_msg,
                    json.dumps([r.model_dump(mode="json") for r in new_refs], ensure_ascii=False),
                    json.dumps([o.model_dump(mode="json") for o in outcomes], ensure_ascii=False),
                    now_iso,
                ),
            )

            existing_refs = [
                EvidenceRef.model_validate(item)
                for item in json.loads(verified["evidence_refs_json"])
            ]
            merged_by_id: dict[str, EvidenceRef] = {r.chunk_id: r for r in existing_refs}
            for ref in new_refs:
                prev = merged_by_id.get(ref.chunk_id)
                if prev is None or ref.score > prev.score:
                    merged_by_id[ref.chunk_id] = ref
            merged_refs = list(merged_by_id.values())

            turn_rows = conn.execute(
                "SELECT * FROM h12_session_turns WHERE session_id = ? ORDER BY turn_index ASC",
                (session_id,),
            ).fetchall()
            all_turns = [self._row_to_turn(r) for r in turn_rows]
            older_turns = all_turns[:-eff_window] if len(all_turns) > eff_window else []
            summary = _summarize_older_turns(older_turns)
            new_turn_count = verified["turn_count"] + (1 if speaker == "student" else 0)
            new_expires = _iso(now + timedelta(days=int(verified["retention_days"])))

            conn.execute(
                """
                UPDATE h12_sessions
                SET turn_count = ?, evidence_refs_json = ?, context_summary = ?,
                    updated_at = ?, expires_at = ?
                WHERE session_id = ?
                """,
                (
                    new_turn_count,
                    json.dumps([r.model_dump(mode="json") for r in merged_refs], ensure_ascii=False),
                    summary,
                    now_iso,
                    new_expires,
                    session_id,
                ),
            )
            inserted = conn.execute(
                "SELECT * FROM h12_session_turns WHERE session_id = ? AND turn_index = ?",
                (session_id, next_idx),
            ).fetchone()
            assert inserted is not None
            return self._row_to_turn(inserted)

    def update_rubric_progress(
        self,
        session_id: str,
        *,
        student_id: str,
        rubric_progress: dict[str, Any],
        now_utc: Optional[datetime] = None,
    ) -> SessionRecord:
        now_iso = _iso(now_utc or _utc_now())
        with self._lock, self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM h12_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            self._verify_owner_row(row, session_id=session_id, student_id=student_id)
            conn.execute(
                "UPDATE h12_sessions SET rubric_progress_json = ?, updated_at = ? WHERE session_id = ?",
                (json.dumps(rubric_progress, ensure_ascii=False), now_iso, session_id),
            )
            updated = conn.execute(
                "SELECT * FROM h12_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            assert updated is not None
            return self._row_to_session(updated)

    def close_session(
        self,
        session_id: str,
        *,
        student_id: str,
        stopping_reason: StoppingReason,
        escalate: bool = False,
        now_utc: Optional[datetime] = None,
    ) -> SessionRecord:
        now_iso = _iso(now_utc or _utc_now())
        target_status: SessionStatus = "escalated" if escalate else "closed"
        with self._lock, self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM h12_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            verified = self._verify_owner_row(row, session_id=session_id, student_id=student_id)
            if verified["status"] != "active":
                raise AppError(
                    ErrorCode.SESSION_STATE_CONFLICT,
                    f"Session '{session_id}' is already {verified['status']}.",
                )
            conn.execute(
                """
                UPDATE h12_sessions
                SET status = ?, stopping_reason = ?, updated_at = ?
                WHERE session_id = ?
                """,
                (target_status, stopping_reason, now_iso, session_id),
            )
            updated = conn.execute(
                "SELECT * FROM h12_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            assert updated is not None
            return self._row_to_session(updated)

    def list_turns(self, session_id: str, *, student_id: str) -> list[SessionTurnRecord]:
        with self._lock, self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM h12_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            self._verify_owner_row(row, session_id=session_id, student_id=student_id)
            rows = conn.execute(
                "SELECT * FROM h12_session_turns WHERE session_id = ? ORDER BY turn_index ASC",
                (session_id,),
            ).fetchall()
            return [self._row_to_turn(r) for r in rows]

    def build_model_context(
        self,
        session_id: str,
        *,
        student_id: str,
        window_turns: Optional[int] = None,
        max_chars: Optional[int] = None,
    ) -> SessionContextWindow:
        """Build a bounded, strictly isolated model-facing conversation context."""
        settings = get_settings()
        eff_window = window_turns if window_turns is not None else settings.agent_context_window_turns
        eff_max_chars = max_chars if max_chars is not None else settings.agent_session_max_context_chars

        with self._lock, self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM h12_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            verified = self._verify_owner_row(row, session_id=session_id, student_id=student_id)
            rows = conn.execute(
                "SELECT * FROM h12_session_turns WHERE session_id = ? ORDER BY turn_index ASC",
                (session_id,),
            ).fetchall()
            all_turns = [self._row_to_turn(r) for r in rows]

        total = len(all_turns)
        recent = all_turns[-eff_window:] if total > eff_window else all_turns
        older = all_turns[:-len(recent)] if len(recent) < total else []
        summary = _summarize_older_turns(older) or verified["context_summary"]

        parts: list[str] = []
        if summary:
            parts.append(f"[Prior Context Summary]\n{summary}")
        for turn in recent:
            parts.append(f"[Turn {turn.turn_index} - {turn.speaker}]\n{turn.message_text}")

        rendered = "\n\n".join(parts)
        truncated = len(older) > 0
        if len(rendered) > eff_max_chars:
            rendered = rendered[-eff_max_chars:]
            truncated = True

        policy_view = ContextWindowPolicyView(
            window_turns=eff_window,
            max_context_chars=eff_max_chars,
            retention_days=int(verified["retention_days"]),
            expires_at=verified["expires_at"],
            total_turns=total,
            included_turns=len(recent),
            summarized_turns=len(older),
            truncated=truncated,
        )
        return SessionContextWindow(
            session_id=session_id,
            context_summary=summary,
            recent_turns=recent,
            rendered_context=rendered,
            policy=policy_view,
        )

    def record_tool_audit(
        self,
        *,
        request_id: str,
        session_id: Optional[str],
        agent_role: AgentRole,
        caller_scope: CallerScope,
        outcome: ToolOutcome,
        now_utc: Optional[datetime] = None,
    ) -> None:
        now_iso = _iso(now_utc or _utc_now())
        with self._lock, self._connection() as conn:
            conn.execute(
                """
                INSERT INTO h12_tool_audit (
                    request_id, session_id, agent_role, caller_scope,
                    tool_name, outcome, source_mode, detail, duration_ms, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    session_id,
                    agent_role,
                    caller_scope,
                    outcome.tool_name,
                    outcome.outcome,
                    outcome.source_mode,
                    outcome.detail,
                    outcome.duration_ms,
                    now_iso,
                ),
            )

    def list_tool_audit(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock, self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM h12_tool_audit WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def purge_expired(self, now_utc: Optional[datetime] = None) -> int:
        now_iso = _iso(now_utc or _utc_now())
        with self._lock, self._connection() as conn:
            cur = conn.execute("DELETE FROM h12_sessions WHERE expires_at <= ?", (now_iso,))
            return cur.rowcount


_session_store: Optional[SessionMemoryStore] = None


def get_session_memory_store() -> SessionMemoryStore:
    global _session_store
    if _session_store is None:
        _session_store = SessionMemoryStore()
    return _session_store
