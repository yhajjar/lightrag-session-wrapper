from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable


def _matches_session(item: Dict[str, Any], doc_id_lookup: set[str]) -> bool:
    """Check whether a response item should be kept."""
    if not item:
        return False
    source_id = item.get("source_id")
    if source_id and source_id in doc_id_lookup:
        return True
    file_path = item.get("file_path", "")
    return any(doc_id in str(file_path) for doc_id in doc_id_lookup)


def filter_by_session(rag_data: Dict[str, Any], session_doc_ids: Iterable[str]) -> Dict[str, Any]:
    """
    Filter LightRAG response data to only include session-relevant information.
    """

    doc_id_lookup = {doc_id for doc_id in session_doc_ids if doc_id}
    if not rag_data or not doc_id_lookup:
        return rag_data or {}

    if "data" not in rag_data:
        return rag_data

    filtered = deepcopy(rag_data)
    data = filtered.get("data", {})

    for key in ("chunks", "entities", "relationships", "references"):
        if key in data and isinstance(data[key], list):
            data[key] = [item for item in data[key] if _matches_session(item, doc_id_lookup)]

    filtered["data"] = data
    return filtered
