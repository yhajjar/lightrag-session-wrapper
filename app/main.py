from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List

import httpx
from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from starlette import status

from .config import settings
from .filtering import filter_by_session
from .lightrag_client import LightRagClient
from .models import (
    DeleteSessionResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    Reference,
    SessionData,
    SessionDocumentsResponse,
    SessionsListResponse,
    UploadResponse,
)
from .session_manager import SessionManager

logger = logging.getLogger("lightrag_session_wrapper")
logging.basicConfig(level=settings.log_level.upper())

session_manager = SessionManager(
    cleanup_interval=settings.session_cleanup_interval,
    max_session_age=settings.session_max_age_hours,
)
lightrag_client: LightRagClient | None = None

cleanup_task: asyncio.Task | None = None
router = APIRouter(prefix="/session", tags=["sessions"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    global cleanup_task, lightrag_client
    lightrag_client = LightRagClient(
        base_url=settings.lightrag_url,
        api_key=settings.lightrag_api_key,
    )
    cleanup_task = asyncio.create_task(_session_cleanup_loop())
    try:
        yield
    finally:
        if cleanup_task:
            cleanup_task.cancel()
            try:
                await cleanup_task
            except asyncio.CancelledError:
                pass
        if lightrag_client:
            await lightrag_client.close()
            lightrag_client = None


app = FastAPI(
    title="LightRAG Session Wrapper",
    description="Session-isolated wrapper for LightRAG REST API",
    version="0.1.0",
    lifespan=lifespan,
)

if settings.cors_origin_list:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


async def _session_cleanup_loop() -> None:
    """Background loop that removes expired sessions."""
    interval = max(5, settings.session_cleanup_interval)
    while True:
        try:
            removed = await session_manager.cleanup_expired_sessions()
            if removed:
                logger.info("Cleaned up %s expired sessions", removed)
        except Exception:
            logger.exception("Failed to cleanup sessions")
        await asyncio.sleep(interval)


async def get_existing_session(session_id: str) -> SessionData:
    session = await session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found",
        )
    return session


def _extract_document_ids(payload: Dict[str, Any]) -> List[str]:
    candidates: List[str] = []
    for key in ("document_ids", "documents", "ids"):
        items = payload.get(key)
        if isinstance(items, list):
            candidates.extend([str(item) for item in items])
        elif isinstance(items, str):
            candidates.append(items)
    if not candidates:
        for key in ("document_id", "id"):
            doc_id = payload.get(key)
            if doc_id:
                candidates.append(str(doc_id))
    return candidates


def _build_references(filtered_payload: Dict[str, Any]) -> List[Reference]:
    references = []
    data = filtered_payload.get("data") or {}
    raw_refs = data.get("references")
    if isinstance(raw_refs, list) and raw_refs:
        for idx, ref in enumerate(raw_refs, start=1):
            references.append(
                Reference(
                    reference_id=str(ref.get("reference_id", idx)),
                    file_path=ref.get("file_path"),
                    source_id=ref.get("source_id"),
                    metadata={k: v for k, v in ref.items() if k not in ("reference_id", "file_path", "source_id")},
                )
            )
    else:
        chunks = data.get("chunks") or []
        for idx, chunk in enumerate(chunks, start=1):
            references.append(
                Reference(
                    reference_id=str(idx),
                    file_path=chunk.get("file_path"),
                    source_id=chunk.get("source_id"),
                    metadata={"chunk_id": chunk.get("chunk_id")},
                )
            )
    return references


def _compose_answer(chunks: List[Dict[str, Any]]) -> str:
    """Construct a simple response string from filtered chunks."""
    if not chunks:
        return "No session-relevant context found for this query."

    parts: List[str] = []
    for chunk in chunks[:3]:
        content = chunk.get("content")
        if content:
            parts.append(content.strip())
    if not parts:
        return "No session-relevant context found for this query."

    merged = "\n\n".join(parts)
    return f"Based on documents in this session:\n\n{merged}"


def _get_lightrag_client() -> LightRagClient:
    if lightrag_client is None:
        raise RuntimeError("LightRAG client not initialised")
    return lightrag_client


@router.post(
    "/{session_id}/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    session_id: str,
    request: Request,
    file: UploadFile = File(...),
):
    if file is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="File payload missing"
        )
    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty"
        )

    session = await session_manager.create_session(session_id)
    user_agent = request.headers.get("user-agent")
    if user_agent:
        session.metadata["user_agent"] = user_agent

    try:
        client = _get_lightrag_client()
        lightrag_response = await client.upload_document(
            filename=file.filename or "document",
            content=content,
            content_type=file.content_type or "application/octet-stream",
        )
    except httpx.HTTPStatusError as exc:
        logger.exception("LightRAG upload failed: %s", exc.response.text)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Unable to upload document to LightRAG",
        ) from exc

    document_ids = _extract_document_ids(lightrag_response)
    if not document_ids:
        logger.warning("LightRAG upload response missing document ids: %s", lightrag_response)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LightRAG did not return document identifiers",
        )

    for doc_id in document_ids:
        await session_manager.add_document(session_id, doc_id)

    message = f"{len(document_ids)} documents uploaded successfully"
    return UploadResponse(
        status="success",
        session_id=session_id,
        document_ids=document_ids,
        message=message,
    )


@router.post(
    "/{session_id}/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
)
async def query_session(
    session_id: str,
    query: QueryRequest,
    session: SessionData = Depends(get_existing_session),
):
    payload = query.model_dump()
    try:
        client = _get_lightrag_client()
        rag_response = await client.query_data(payload)
    except httpx.HTTPStatusError as exc:
        logger.exception("LightRAG query failed: %s", exc.response.text)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Unable to query LightRAG",
        ) from exc

    filtered = filter_by_session(rag_response, session.document_ids)
    data_section = filtered.get("data") or {}
    chunks = data_section.get("chunks") or []

    response_text = _compose_answer(chunks)
    references = _build_references(filtered)

    await session_manager.update_activity(session_id)
    return QueryResponse(
        response=response_text,
        references=references,
        session_id=session_id,
        document_count=len(session.document_ids),
        filtered_data=filtered,
    )


@router.get(
    "/{session_id}/documents",
    response_model=SessionDocumentsResponse,
    status_code=status.HTTP_200_OK,
)
async def list_session_documents(
    session_id: str,
    session: SessionData = Depends(get_existing_session),
):
    await session_manager.update_activity(session_id)
    return SessionDocumentsResponse(
        session_id=session_id,
        document_count=len(session.document_ids),
        document_ids=session.document_ids,
    )


@router.delete(
    "/{session_id}",
    response_model=DeleteSessionResponse,
    status_code=status.HTTP_200_OK,
)
async def delete_session(
    session_id: str,
    session: SessionData = Depends(get_existing_session),
):
    deleted_count = 0
    error_count = 0
    for doc_id in session.document_ids:
        try:
            client = _get_lightrag_client()
            success = await client.delete_document(doc_id)
            if success:
                deleted_count += 1
            else:
                error_count += 1
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "Failed to delete LightRAG document %s: %s", doc_id, exc.response.text
            )
            error_count += 1
    await session_manager.delete_session(session_id)
    return DeleteSessionResponse(
        status="session deleted",
        session_id=session_id,
        deleted_count=deleted_count,
        error_count=error_count,
    )


app.include_router(router)


@app.get("/sessions", response_model=SessionsListResponse)
async def list_sessions():
    sessions = await session_manager.get_all_sessions()
    return SessionsListResponse(sessions=sessions, total=len(sessions))


@app.get("/health", response_model=HealthResponse)
async def health_check():
    status_text = "healthy"
    try:
        client = _get_lightrag_client()
        lightrag_status_payload = await client.health()
        status_text = lightrag_status_payload.get("status", "healthy")
    except httpx.HTTPStatusError as exc:
        try:
            status_text = exc.response.json().get("status", "unhealthy")
        except ValueError:
            status_text = "unhealthy"
    except httpx.HTTPError:
        status_text = "unhealthy"
    active_sessions = len(await session_manager.get_all_sessions())
    return HealthResponse(
        status="healthy" if status_text == "healthy" else "degraded",
        lightrag_status=status_text,
        active_sessions=active_sessions,
    )
