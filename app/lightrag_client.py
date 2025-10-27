from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import httpx


class LightRagClient:
    """HTTP client wrapper for LightRAG interactions."""

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        headers = {"Accept": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(
            base_url=self.base_url, headers=headers, timeout=timeout
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def upload_document(self, filename: str, content: bytes, content_type: str) -> Dict[str, Any]:
        files = {"file": (filename, content, content_type)}
        response = await self._client.post("/documents/upload", files=files)
        response.raise_for_status()
        return response.json()

    async def query_data(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = await self._client.post("/query/data", json=payload)
        response.raise_for_status()
        return response.json()

    async def query(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = await self._client.post("/query", json=payload)
        response.raise_for_status()
        return response.json()

    async def delete_document(self, document_id: str) -> bool:
        response = await self._client.delete(f"/documents/{document_id}")
        if response.status_code in (200, 202, 204):
            return True
        response.raise_for_status()
        return False

    async def health(self) -> Dict[str, Any]:
        response = await self._client.get("/health")
        response.raise_for_status()
        return response.json()
