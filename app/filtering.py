from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Dict, Iterable, Mapping

from .utils import normalize_file_name

logger = logging.getLogger("lightrag_session_wrapper")

DOC_ID_KEYS = {
    "doc_id",
    "doc_ids",
    "document_id",
    "document_ids",
    "source_doc_id",
    "source_doc_ids",
    "source_document_id",
    "source_document_ids",
    "documents",
    "source_documents",
}

FILE_NAME_KEYS = {
    "file_path",
    "file",
    "file_name",
    "original_file_name",
    "source_file",
    "source_path",
}


def _matches_identifier(value: str, doc_id_lookup: set[str]) -> bool:
    """Return True if identifier matches or contains a document id with safe heuristics."""
    if not value:
        return False
    for doc_id in doc_id_lookup:
        if value == doc_id:
            return True
        # Allow partial matches for generated ids (e.g. chunk-doc-<id>-001) while
        # enforcing a minimum length to reduce false positives (doc-1 vs doc-10).
        if len(doc_id) >= 6 and doc_id in value:
            return True
    return False


def _iter_string_values(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _iter_string_values(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_string_values(item)


def _collect_candidates(item: Mapping[str, Any], keys: Iterable[str]) -> Iterable[str]:
    for key in keys:
        if key in item:
            yield from _iter_string_values(item[key])


def _matches_session(
    item: Dict[str, Any],
    doc_id_lookup: set[str],
    file_name_lookup: set[str],
    reference_lookup: Dict[str, Dict[str, Any]],
) -> bool:
    """Check whether a response item should be kept."""
    if not item:
        return False
    source_id = item.get("source_id")
    if source_id and _matches_identifier(str(source_id), doc_id_lookup):
        return True
    file_path = item.get("file_path", "")
    if file_path:
        normalized = normalize_file_name(file_path)
        if normalized and normalized in file_name_lookup:
            return True
        if _matches_identifier(str(file_path), doc_id_lookup):
            return True
    metadata = item.get("metadata")
    if isinstance(metadata, dict):
        for doc_candidate in _collect_candidates(metadata, DOC_ID_KEYS):
            if _matches_identifier(str(doc_candidate), doc_id_lookup):
                return True
        for file_candidate in _collect_candidates(metadata, FILE_NAME_KEYS):
            normalized = normalize_file_name(file_candidate)
            if normalized and normalized in file_name_lookup:
                return True

    for doc_candidate in _collect_candidates(item, DOC_ID_KEYS):
        if _matches_identifier(str(doc_candidate), doc_id_lookup):
            return True

    for file_candidate in _collect_candidates(item, FILE_NAME_KEYS):
        normalized = normalize_file_name(file_candidate)
        if normalized and normalized in file_name_lookup:
            return True

    reference_id = item.get("reference_id")
    if reference_id is not None:
        ref = reference_lookup.get(str(reference_id))
        if ref:
            ref_file = normalize_file_name(ref.get("file_path"))
            if ref_file and ref_file in file_name_lookup:
                return True
            ref_source_id = ref.get("source_id") or ref.get("document_id")
            if ref_source_id and _matches_identifier(str(ref_source_id), doc_id_lookup):
                return True
            ref_metadata = ref.get("metadata")
            if isinstance(ref_metadata, dict):
                for doc_candidate in _collect_candidates(ref_metadata, DOC_ID_KEYS):
                    if _matches_identifier(str(doc_candidate), doc_id_lookup):
                        return True
                for file_candidate in _collect_candidates(ref_metadata, FILE_NAME_KEYS):
                    normalized = normalize_file_name(file_candidate)
                    if normalized and normalized in file_name_lookup:
                        return True

    return False


def filter_by_session(
    rag_data: Dict[str, Any],
    session_doc_ids: Iterable[str],
    session_file_names: Iterable[str] | None = None,
) -> Dict[str, Any]:
    """
    Filter LightRAG response data to only include session-relevant information.
    """

    doc_id_lookup = {doc_id for doc_id in session_doc_ids if doc_id}
    file_name_lookup = {
        normalize_file_name(name)
        for name in (session_file_names or [])
        if normalize_file_name(name)
    }
    if not rag_data:
        return {}

    raw_data = rag_data.get("data") or {}
    reference_lookup: Dict[str, Dict[str, Any]] = {}
    raw_references = raw_data.get("references")
    if isinstance(raw_references, list):
        for ref in raw_references:
            if isinstance(ref, dict):
                ref_id = ref.get("reference_id")
                if ref_id is not None:
                    reference_lookup[str(ref_id)] = ref

    if not doc_id_lookup and not file_name_lookup:
        return rag_data

    if "data" not in rag_data:
        return rag_data

    filtered = deepcopy(rag_data)
    data = filtered.get("data", {})

    doc_id_lookup = {value for value in doc_id_lookup if value}
    for key in ("chunks", "entities", "relationships", "references"):
        if key in data and isinstance(data[key], list):
            data[key] = [
                item
                for item in data[key]
                if _matches_session(
                    item,
                    doc_id_lookup,
                    file_name_lookup,
                    reference_lookup,
                )
            ]

    filtered["data"] = data

    if all(
        not data.get(key)
        for key in ("chunks", "entities", "relationships", "references")
    ):
        original_has_records = any(
            isinstance(raw_data.get(key), list) and raw_data.get(key)
            for key in ("chunks", "entities", "relationships", "references")
        )
        if original_has_records:
            logger.warning(
                "Filter produced empty result despite LightRAG returning context; falling back to original payload."
            )
            filtered["data"] = raw_data

    return filtered
