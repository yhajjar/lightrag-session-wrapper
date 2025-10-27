import pytest
import respx
from httpx import AsyncClient, Response

from app.config import settings
from app.main import app, session_manager


@pytest.mark.asyncio
async def test_session_flow():
    base_url = settings.lightrag_url.rstrip("/")

    async with AsyncClient(app=app, base_url="http://testserver") as client:
        with respx.mock(assert_all_called=False) as mock:
            mock.post(f"{base_url}/documents/upload").respond(
                200, json={"document_ids": ["doc-xyz789"]}
            )
            mock.post(f"{base_url}/query/data").respond(
                200,
                json={
                    "status": "success",
                    "data": {
                        "chunks": [
                            {
                                "content": "Chunk content",
                                "file_path": "doc1.pdf",
                                "source_id": "doc-xyz789",
                                "chunk_id": "chunk-123",
                            }
                        ],
                        "references": [
                            {"file_path": "doc1.pdf", "source_id": "doc-xyz789"}
                        ],
                    },
                },
            )
            mock.delete(f"{base_url}/documents/doc-xyz789").respond(204)

            response = await client.post(
                "/session/test-session/upload",
                files={"file": ("doc1.pdf", b"hello world", "application/pdf")},
            )
            assert response.status_code == 201
            payload = response.json()
            assert payload["document_ids"] == ["doc-xyz789"]
            assert payload["pending"] is False
            assert payload["track_id"] is None

            docs_response = await client.get("/session/test-session/documents")
            assert docs_response.status_code == 200
            docs_payload = docs_response.json()
            assert docs_payload["document_ids"] == ["doc-xyz789"]

            query_response = await client.post(
                "/session/test-session/query",
                json={"query": "What are the points?", "mode": "mix"},
            )
            assert query_response.status_code == 200
            query_payload = query_response.json()
            assert "Based on documents" in query_payload["response"]
            assert query_payload["references"][0]["source_id"] == "doc-xyz789"

            delete_response = await client.delete("/session/test-session")
            assert delete_response.status_code == 200
            delete_payload = delete_response.json()
            assert delete_payload["deleted_count"] == 1

        sessions_response = await client.get("/sessions")
        assert sessions_response.status_code == 200
        sessions_payload = sessions_response.json()
        assert "test-session" not in sessions_payload["sessions"]


@pytest.mark.asyncio
async def test_health_endpoint_handles_lightrag_failure():
    base_url = settings.lightrag_url.rstrip("/")
    async with AsyncClient(app=app, base_url="http://testserver") as client:
        with respx.mock(assert_all_called=False) as mock:
            mock.get(f"{base_url}/health").respond(503, json={"status": "error"})
            response = await client.get("/health")
            assert response.status_code == 200
            payload = response.json()
        assert payload["status"] == "degraded"
        assert payload["lightrag_status"] == "error"


@pytest.mark.asyncio
async def test_upload_resolves_track_id(monkeypatch):
    base_url = settings.lightrag_url.rstrip("/")
    track_id = "upload_123"

    async with AsyncClient(app=app, base_url="http://testserver") as client:
        with respx.mock(assert_all_called=False) as mock:
            mock.post(f"{base_url}/documents/upload").respond(
                200,
                json={
                    "status": "success",
                    "message": "Processing in background.",
                    "track_id": track_id,
                },
            )
            mock.get(f"{base_url}/documents/track_status/{track_id}").respond(
                200,
                json={
                    "track_id": track_id,
                    "documents": [
                        {"id": "doc-processed", "status": "completed"}
                    ],
                    "total_count": 1,
                },
            )

            response = await client.post(
                "/session/track-session/upload",
                files={"file": ("doc2.pdf", b"hello world", "application/pdf")},
            )
            assert response.status_code == 201
            payload = response.json()
            assert payload["document_ids"] == ["doc-processed"]
            assert payload["track_id"] == track_id
            assert payload["pending"] is False
            assert payload["status"] == "success"
        await session_manager.delete_session("track-session")


@pytest.mark.asyncio
async def test_upload_pending_placeholder(monkeypatch):
    base_url = settings.lightrag_url.rstrip("/")
    track_id = "upload_pending"

    original_timeout = settings.upload_status_timeout
    original_background_timeout = settings.upload_status_background_timeout
    try:
        settings.upload_status_timeout = 0
        settings.upload_status_background_timeout = 0

        async with AsyncClient(app=app, base_url="http://testserver") as client:
            with respx.mock(assert_all_called=False) as mock:
                mock.post(f"{base_url}/documents/upload").respond(
                    200,
                    json={
                        "status": "success",
                        "message": "Processing in background.",
                        "track_id": track_id,
                    },
                )
                mock.get(f"{base_url}/documents/track_status/{track_id}").respond(
                    200,
                    json={"track_id": track_id, "documents": [], "total_count": 0},
                )

                response = await client.post(
                    "/session/pending-session/upload",
                    files={"file": ("doc3.pdf", b"hello world", "application/pdf")},
                )
                assert response.status_code == 201
                payload = response.json()
                assert payload["pending"] is True
                assert payload["track_id"] == track_id
                assert payload["document_ids"][0].startswith("track:")
                assert payload["status"] == "processing"
        await session_manager.delete_session("pending-session")
    finally:
        settings.upload_status_timeout = original_timeout
        settings.upload_status_background_timeout = original_background_timeout


@pytest.mark.asyncio
async def test_query_matches_by_file_name():
    base_url = settings.lightrag_url.rstrip("/")

    async with AsyncClient(app=app, base_url="http://testserver") as client:
        with respx.mock(assert_all_called=False) as mock:
            mock.post(f"{base_url}/documents/upload").respond(
                200,
                json={"document_ids": ["doc-file"]},
            )
            mock.post(f"{base_url}/query/data").respond(
                200,
                json={
                    "status": "success",
                    "data": {
                        "chunks": [
                            {
                                "content": "Key findings",
                                "file_path": "ANNUAL SUMMARY 2024.PDF",
                                "source_id": "chunk-001",
                                "chunk_id": "chunk-001",
                            }
                        ],
                        "references": [],
                        "entities": [],
                        "relationships": [],
                    },
                    "metadata": {
                        "processing_info": {"final_chunks_count": 1}
                    },
                },
            )

            response = await client.post(
                "/session/file-session/upload",
                files={
                    "file": ("Reports/Annual Summary 2024.PDF", b"content", "application/pdf")
                },
            )
            assert response.status_code == 201

            query_response = await client.post(
                "/session/file-session/query",
                json={"query": "Summarize the annual report", "mode": "mix"},
            )
            assert query_response.status_code == 200
            data = query_response.json()
            assert data["filtered_data"]["data"]["chunks"][0]["content"] == "Key findings"

        await session_manager.delete_session("file-session")
