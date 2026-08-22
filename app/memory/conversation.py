"""
Lightweight in-process conversation memory, keyed by session_id.

This is intentionally simple (a dict guarded by a lock) rather than a
database — it's enough to give the chatbot multi-turn context within a
single process, and swapping in Redis/Postgres later only means changing
this one module. Sessions are trimmed to `max_history_turns` and expire
after `session_ttl_minutes` of inactivity so memory usage stays bounded.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field

from app.config import settings


@dataclass
class Turn:
    user: str
    assistant: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class Session:
    session_id: str
    turns: list[Turn] = field(default_factory=list)
    last_active: float = field(default_factory=time.time)


class ConversationStore:
    def __init__(self, max_history_turns: int | None = None, ttl_minutes: int | None = None):
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        self._max_history_turns = max_history_turns or settings.max_history_turns
        self._ttl_seconds = (ttl_minutes or settings.session_ttl_minutes) * 60

    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        with self._lock:
            self._sessions[session_id] = Session(session_id=session_id)
        return session_id

    def _get_or_create(self, session_id: str | None) -> Session:
        with self._lock:
            self._evict_expired_locked()
            if session_id and session_id in self._sessions:
                return self._sessions[session_id]
            new_id = session_id or str(uuid.uuid4())
            session = Session(session_id=new_id)
            self._sessions[new_id] = session
            return session

    def get_history(self, session_id: str | None) -> list[Turn]:
        if not session_id:
            return []
        with self._lock:
            self._evict_expired_locked()
            session = self._sessions.get(session_id)
            return list(session.turns) if session else []

    def append_turn(self, session_id: str | None, user: str, assistant: str) -> str:
        session = self._get_or_create(session_id)
        with self._lock:
            session.turns.append(Turn(user=user, assistant=assistant))
            session.last_active = time.time()
            if len(session.turns) > self._max_history_turns:
                session.turns = session.turns[-self._max_history_turns :]
            return session.session_id

    def reset(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def _evict_expired_locked(self) -> None:
        now = time.time()
        expired = [sid for sid, s in self._sessions.items() if now - s.last_active > self._ttl_seconds]
        for sid in expired:
            del self._sessions[sid]


# Process-wide singleton — swap for a real backing store if you need to
# scale the API across multiple processes/instances.
conversation_store = ConversationStore()
