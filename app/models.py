from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class SessionData(BaseModel):
    """Represents in-memory session state."""

    model_config = ConfigDict(extra="ignore")

    session_id: str
    document_ids: List[str] = Field(default_factory=list)
    created_at: datetime
    last_activity: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    query: str
    mode: str = "mix"
    include_references: bool = True
    top_k: int = 10
    response_type: Optional[str] = None
    stream: Optional[bool] = None
    hl_keywords: Optional[List[str]] = None
    ll_keywords: Optional[List[str]] = None
    conversation_history: Optional[List[Dict[str, Any]]] = None
    user_prompt: Optional[str] = None


class Reference(BaseModel):
    reference_id: Optional[str] = None
    file_path: Optional[str] = None
    source_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    response: str
    references: List[Reference] = Field(default_factory=list)
    session_id: str
    document_count: int
    filtered_data: Dict[str, Any] = Field(default_factory=dict)


class UploadResponse(BaseModel):
    status: str
    session_id: str
    document_ids: List[str]
    message: str
    track_id: Optional[str] = None
    pending: bool = False


class SessionDocumentsResponse(BaseModel):
    session_id: str
    document_count: int
    document_ids: List[str]


class DeleteSessionResponse(BaseModel):
    status: str
    session_id: str
    deleted_count: int
    error_count: int


class SessionsListResponse(BaseModel):
    sessions: List[str]
    total: int


class HealthResponse(BaseModel):
    status: str
    lightrag_status: str
    active_sessions: int
