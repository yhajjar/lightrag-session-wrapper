from app.filtering import filter_by_session


def test_filter_by_session_keeps_matching_items():
    rag_response = {
        "status": "success",
        "data": {
            "chunks": [
                {"content": "A", "file_path": "chunk-doc-xyz789-1.txt", "source_id": "doc-xyz789"},
                {"content": "B", "file_path": "chunk-doc-abc456-2.txt", "source_id": "doc-abc456"},
            ],
            "entities": [
                {"entity_name": "Entity A", "source_id": "doc-xyz789", "file_path": "doc_xyz789_summary.md"},
                {"entity_name": "Entity B", "source_id": "doc-zzz333", "file_path": "doc_zzz333_notes.md"},
            ],
            "relationships": [
                {"relation": "rel1", "source_id": "doc-abc456", "file_path": "doc_abc456_rel.json"}
            ],
            "references": [
                {"file_path": "doc_xyz789_summary.md"},
                {"file_path": "unmatched.pdf"},
            ],
        },
    }

    filtered = filter_by_session(rag_response, ["doc-xyz789"])

    chunks = filtered["data"]["chunks"]
    entities = filtered["data"]["entities"]
    relationships = filtered["data"]["relationships"]
    references = filtered["data"]["references"]

    assert len(chunks) == 1 and chunks[0]["source_id"] == "doc-xyz789"
    assert len(entities) == 1 and entities[0]["source_id"] == "doc-xyz789"
    assert len(relationships) == 0
    assert references == [{"file_path": "doc_xyz789_summary.md"}]


def test_filter_by_session_returns_original_when_no_data():
    result = filter_by_session({}, ["doc-1"])
    assert result == {}


def test_filter_allows_partial_source_match():
    rag_response = {
        "status": "success",
        "data": {
            "chunks": [
                {"content": "A", "file_path": "chunk-1.txt", "source_id": "chunk-doc-xyz789-001"}
            ]
        },
    }

    filtered = filter_by_session(rag_response, ["doc-xyz789"])
    assert filtered["data"]["chunks"][0]["source_id"] == "chunk-doc-xyz789-001"
