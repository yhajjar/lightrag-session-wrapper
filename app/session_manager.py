from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from .models import SessionData


def utcnow() -> datetime:
    """Return timezone-aware UTC now."""
    return datetime.now(timezone.utc)


class SessionManager:
    """Manages session lifecycle, document tracking, and cleanup."""

    def __init__(self, cleanup_interval: int = 3600, max_session_age: int = 24):
        self._sessions: Dict[str, SessionData] = {}
        self._lock = asyncio.Lock()
        self.cleanup_interval = cleanup_interval
        self.max_session_age = max_session_age

    async def create_session(self, session_id: str) -> SessionData:
        """Ensure a session exists and return it."""
        async with self._lock:
            if session_id not in self._sessions:
                now = utcnow()
                self._sessions[session_id] = SessionData(
                    session_id=session_id,
                    document_ids=[],
                    created_at=now,
                    last_activity=now,
                    metadata={"document_count": 0},
                )
            return self._sessions[session_id]

    async def add_document(self, session_id: str, doc_id: str) -> None:
        """Associate a document with a session, creating the session if needed."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                now = utcnow()
                session = SessionData(
                    session_id=session_id,
                    document_ids=[],
                    created_at=now,
                    last_activity=now,
                    metadata={"document_count": 0},
                )
                self._sessions[session_id] = session
            if doc_id not in session.document_ids:
                session.document_ids.append(doc_id)
                session.metadata["document_count"] = len(session.document_ids)
            session.last_activity = utcnow()

    async def get_session(self, session_id: str) -> Optional[SessionData]:
        """Return a session by id."""
        async with self._lock:
            return self._sessions.get(session_id)

    async def get_session_documents(self, session_id: str) -> List[str]:
        """Return document ids for the session."""
        async with self._lock:
            session = self._sessions.get(session_id)
            return list(session.document_ids) if session else []

    async def delete_session(self, session_id: str) -> bool:
        """Remove a session from the manager."""
        async with self._lock:
            return self._sessions.pop(session_id, None) is not None

    async def update_activity(self, session_id: str) -> None:
        """Update session last activity timestamp."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.last_activity = utcnow()
                session.metadata["document_count"] = len(session.document_ids)

    async def cleanup_expired_sessions(self) -> int:
        """Remove sessions past the max age. Returns number of removed sessions."""
        expiry = timedelta(hours=self.max_session_age)
        now = utcnow()
        async with self._lock:
            expired = [
                session_id
                for session_id, session in self._sessions.items()
                if now - session.last_activity > expiry
            ]
            for session_id in expired:
                self._sessions.pop(session_id, None)
        return len(expired)

    async def get_all_sessions(self) -> List[str]:
        """Return a list of active session ids."""
        async with self._lock:
            return list(self._sessions.keys())
