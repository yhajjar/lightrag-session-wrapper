from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Tuple

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
from .utils import normalize_file_name

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
            metadata = {
                k: v
                for k, v in ref.items()
                if k not in ("reference_id", "file_path", "source_id")
            }
            original_reference_id = ref.get("reference_id")
            if original_reference_id is not None:
                metadata.setdefault("original_reference_id", original_reference_id)
            references.append(
                Reference(
                    reference_id=str(idx),
                    file_path=ref.get("file_path"),
                    source_id=ref.get("source_id"),
                    metadata=metadata,
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


def _compose_answer(chunks: List[Dict[str, Any]], query_text: str) -> str:
    """Construct a simple response string from filtered chunks."""
    normalized_query = (query_text or "").strip()
    if not chunks:
        if normalized_query:
            display_query = (
                normalized_query
                if len(normalized_query) <= 120
                else f"{normalized_query[:117]}..."
            )
            return f"No documents in this session contain information about '{display_query}'."
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


async def _register_file_names(session_id: str, *file_names: str) -> None:
    """Store normalized file names for session filtering."""
    unique = {
        name
        for name in (normalize_file_name(value) for value in file_names)
        if name
    }
    for name in unique:
        await session_manager.add_file_name(session_id, name)


async def _fetch_documents_for_track(
    track_id: str, *, timeout: float, interval: float
) -> List[Dict[str, Any]]:
    """Attempt to resolve documents associated with a LightRAG track id."""
    if timeout < 0:
        timeout = 0.0
    if interval <= 0:
        interval = 1.0

    client = _get_lightrag_client()
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    attempt = 0

    while True:
        attempt += 1
        try:
            payload = await client.get_track_status(track_id)
        except httpx.HTTPError as exc:
            logger.warning(
                "Failed to fetch track status for %s on attempt %d: %s",
                track_id,
                attempt,
                exc,
            )
            if timeout == 0:
                return []
        else:
            documents = payload.get("documents") or []
            resolved = []
            statuses = set()
            for doc in documents:
                doc_id = doc.get("id")
                status_value = doc.get("status")
                if status_value is not None:
                    statuses.add(str(status_value).lower())
                if doc_id:
                    resolved.append(
                        {
                            "id": str(doc_id),
                            "file_path": doc.get("file_path")
                            or (doc.get("metadata") or {}).get("file_path"),
                            "status": status_value,
                            "metadata": doc.get("metadata") or {},
                        }
                    )
            if resolved:
                logger.info(
                    "Resolved track %s to document ids %s on attempt %d",
                    track_id,
                    [rec["id"] for rec in resolved],
                    attempt,
                )
                return resolved
            if statuses and statuses.issubset({"failed", "error"}):
                logger.warning(
                    "Track %s reached terminal failure states: %s", track_id, statuses
                )
                return []

        if timeout == 0:
            return []

        now = loop.time()
        if now >= deadline:
            logger.info(
                "Timeout while waiting for track %s to produce document ids", track_id
            )
            return []

        sleep_for = min(interval, max(0.1, deadline - now))
        await asyncio.sleep(sleep_for)


def _schedule_track_resolution(session_id: str, placeholder_id: str, track_id: str) -> None:
    """Fire-and-forget background task to resolve pending track ids."""

    async def _runner() -> None:
        try:
            records = await _fetch_documents_for_track(
                track_id,
                timeout=settings.upload_status_background_timeout,
                interval=settings.upload_status_poll_interval,
            )
            if records:
                for record in records:
                    metadata = record.get("metadata") or {}
                    await _register_file_names(
                        session_id,
                        record.get("file_path"),
                        metadata.get("file_name"),
                        metadata.get("original_file_name"),
                        metadata.get("source_file"),
                        metadata.get("source_path"),
                    )
                doc_ids = [record["id"] for record in records if record.get("id")]
                await session_manager.resolve_pending_track(
                    session_id, placeholder_id, doc_ids
                )
                logger.info(
                    "Background resolved track %s for session %s to %s",
                    track_id,
                    session_id,
                    doc_ids,
                )
        except Exception:
            logger.exception(
                "Unexpected error while resolving track %s for session %s",
                track_id,
                session_id,
            )

    asyncio.create_task(_runner())


async def _resolve_pending_tracks(session_id: str, *, timeout: float) -> None:
    """Refresh placeholder document ids for a session, if possible."""
    pending = await session_manager.get_pending_tracks(session_id)
    if not pending:
        return
    for placeholder_id, track_id in pending.items():
        try:
            records = await _fetch_documents_for_track(
                track_id,
                timeout=timeout,
                interval=settings.upload_status_poll_interval,
            )
        except Exception:
            logger.exception(
                "Failed to resolve pending track %s for session %s",
                track_id,
                session_id,
            )
            continue
        if records:
            for record in records:
                metadata = record.get("metadata") or {}
                await _register_file_names(
                    session_id,
                    record.get("file_path"),
                    metadata.get("file_name"),
                    metadata.get("original_file_name"),
                    metadata.get("source_file"),
                    metadata.get("source_path"),
                )
            doc_ids = [record["id"] for record in records if record.get("id")]
            await session_manager.resolve_pending_track(
                session_id, placeholder_id, doc_ids
            )


def _build_reference_lookup(references: List[Reference]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for reference in references:
        display_id = reference.reference_id
        if display_id:
            lookup[str(display_id)] = display_id
        original_id = reference.metadata.get("original_reference_id")
        if original_id is not None:
            lookup[str(original_id)] = display_id
    return lookup


def _select_context_chunks(
    filtered_payload: Dict[str, Any],
    max_chunks: int = 6,
    max_chars: int = 8000,
) -> List[Tuple[Dict[str, Any], str]]:
    data = filtered_payload.get("data") or {}
    chunks = data.get("chunks") or []
    selected: List[Tuple[Dict[str, Any], str]] = []
    total_chars = 0

    for chunk in chunks:
        content = (chunk.get("content") or "").strip()
        if not content:
            continue
        selected.append((chunk, content))
        total_chars += len(content)
        if len(selected) >= max_chunks or total_chars >= max_chars:
            break

    return selected


def _build_context_text(
    chunks: List[Tuple[Dict[str, Any], str]], reference_lookup: Dict[str, str]
) -> str:
    lines: List[str] = []
    for chunk, content in chunks:
        reference_id = chunk.get("reference_id")
        label = ""
        if reference_id is not None:
            mapped = reference_lookup.get(str(reference_id))
            if mapped:
                label = f"[{mapped}] "
        if label or content:
            lines.append(f"{label}{content}")
    return "\n\n".join(lines).strip()


def _format_reference_section(references: List[Reference]) -> str:
    lines: List[str] = []
    for reference in references:
        label = reference.reference_id or ""
        descriptor = reference.file_path or reference.source_id or "Document"
        lines.append(f"[{label}] {descriptor}")
    return "\n".join(lines).strip()


async def _generate_llm_summary(
    query_request: QueryRequest,
    filtered_payload: Dict[str, Any],
    references: List[Reference],
) -> str | None:
    reference_lookup = _build_reference_lookup(references)
    context_chunks = _select_context_chunks(filtered_payload)
    if not context_chunks:
        return None

    context_text = _build_context_text(context_chunks, reference_lookup)
    if not context_text:
        return None

    reference_section = _format_reference_section(references)

    prompt_parts = [
        "You are a helpful assistant that answers questions using only the provided context.",
        "If the context does not contain the answer, reply with: Context does not contain relevant information.",
        "Cite supporting statements using bracketed reference numbers like [1] that correspond to the reference list.",
        "",
        "Context:",
        context_text,
    ]
    if reference_section:
        prompt_parts.extend(["", "Reference list:", reference_section])
    prompt_parts.extend(
        [
            "",
            f"Question: {query_request.query}",
            "",
            "Answer:",
        ]
    )

    payload: Dict[str, Any] = {
        "query": "\n".join(prompt_parts),
        "mode": "bypass",
        "include_references": False,
        "response_type": query_request.response_type or "Multiple Paragraphs",
        "stream": False,
    }

    if query_request.top_k is not None:
        payload["top_k"] = query_request.top_k

    if query_request.conversation_history:
        payload["conversation_history"] = query_request.conversation_history

    try:
        client = _get_lightrag_client()
        llm_response = await client.query(payload)
    except httpx.HTTPError:
        logger.exception("LightRAG bypass summarisation failed")
        return None

    answer = llm_response.get("response")
    if isinstance(answer, str) and answer.strip():
        return answer.strip()
    return None


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

    await _register_file_names(session_id, file.filename)

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
    track_id = lightrag_response.get("track_id")
    pending = False
    track_records: List[Dict[str, Any]] = []

    if not document_ids and track_id:
        track_records = await _fetch_documents_for_track(
            track_id,
            timeout=settings.upload_status_timeout,
            interval=settings.upload_status_poll_interval,
        )
        if track_records:
            document_ids = [record["id"] for record in track_records if record.get("id")]

    if not document_ids:
        if track_id:
            pending = True
            placeholder_id = f"track:{track_id}"
            document_ids = [placeholder_id]
            await session_manager.mark_pending_track(session_id, placeholder_id, track_id)
            _schedule_track_resolution(session_id, placeholder_id, track_id)
            logger.info(
                "Tracking pending document for session %s via track_id %s",
                session_id,
                track_id,
            )
        else:
            logger.warning(
                "LightRAG upload response missing document ids: %s", lightrag_response
            )
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="LightRAG did not return document identifiers",
            )

    for doc_id in document_ids:
        await session_manager.add_document(session_id, doc_id)

    if track_records:
        for record in track_records:
            metadata = record.get("metadata") or {}
            await _register_file_names(
                session_id,
                record.get("file_path"),
                metadata.get("file_name"),
                metadata.get("original_file_name"),
                metadata.get("source_file"),
                metadata.get("source_path"),
            )

    message = lightrag_response.get("message")
    if not message:
        if pending:
            message = "File accepted for background processing."
        else:
            count = len(document_ids)
            plural = "document" if count == 1 else "documents"
            message = f"{count} {plural} uploaded successfully"

    status_text = "success" if not pending else "processing"
    return UploadResponse(
        status=status_text,
        session_id=session_id,
        document_ids=document_ids,
        message=message,
        track_id=track_id,
        pending=pending,
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
    await _resolve_pending_tracks(session_id, timeout=0.0)
    refreshed_session = await session_manager.get_session(session_id)
    if refreshed_session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found",
        )
    session = refreshed_session
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

    file_names = session.metadata.get("file_names", [])
    filtered = filter_by_session(rag_response, session.document_ids, file_names)
    data_section = filtered.get("data") or {}
    chunks = data_section.get("chunks") or []

    references = _build_references(filtered)
    response_text = await _generate_llm_summary(query, filtered, references)
    if response_text is None:
        response_text = _compose_answer(chunks, query.query)

    logger.info(
        "Session %s query='%s' doc_count=%d filtered_chunks=%d",
        session_id,
        query.query,
        len(session.document_ids),
        len(chunks),
    )

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
    await _resolve_pending_tracks(session_id, timeout=0.0)
    refreshed_session = await session_manager.get_session(session_id)
    if refreshed_session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found",
        )
    session = refreshed_session
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
    await _resolve_pending_tracks(
        session_id, timeout=settings.upload_status_poll_interval
    )
    refreshed_session = await session_manager.get_session(session_id)
    if refreshed_session is None:
        return DeleteSessionResponse(
            status="session deleted",
            session_id=session_id,
            deleted_count=0,
            error_count=0,
        )
    session = refreshed_session
    deleted_count = 0
    error_count = 0
    for doc_id in session.document_ids:
        if doc_id.startswith("track:"):
            logger.warning(
                "Cannot delete pending LightRAG document %s for session %s",
                doc_id,
                session_id,
            )
            error_count += 1
            continue
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
