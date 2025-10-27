import pytest
from datetime import timedelta

from app.session_manager import SessionManager, utcnow


@pytest.mark.asyncio
async def test_create_and_add_document():
    manager = SessionManager(max_session_age=1)
    await manager.create_session("session-1")
    await manager.add_document("session-1", "doc-1")

    session = await manager.get_session("session-1")
    assert session is not None
    assert session.document_ids == ["doc-1"]
    assert session.metadata["document_count"] == 1


@pytest.mark.asyncio
async def test_cleanup_expired_sessions():
    manager = SessionManager(max_session_age=0)
    await manager.create_session("session-expired")
    session = await manager.get_session("session-expired")
    assert session is not None

    # Age the session beyond the TTL
    session.last_activity = utcnow() - timedelta(hours=1)

    removed = await manager.cleanup_expired_sessions()
    assert removed == 1
    assert await manager.get_session("session-expired") is None


@pytest.mark.asyncio
async def test_pending_track_resolution():
    manager = SessionManager()
    await manager.add_document("session-2", "track:placeholder")
    await manager.mark_pending_track("session-2", "track:placeholder", "upload_abc")
    pending = await manager.get_pending_tracks("session-2")
    assert pending == {"track:placeholder": "upload_abc"}

    await manager.resolve_pending_track("session-2", "track:placeholder", ["doc-real"])
    session = await manager.get_session("session-2")
    assert session is not None
    assert session.document_ids == ["doc-real"]
    assert await manager.get_pending_tracks("session-2") == {}
