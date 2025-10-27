from app.filtering import filter_by_session


def test_filter_by_session_keeps_matching_items():
    rag_response = {
        "status": "success",
        "data": {
            "chunks": [
                {"content": "A", "file_path": "doc1.pdf", "source_id": "doc-1"},
                {"content": "B", "file_path": "doc2.pdf", "source_id": "doc-2"},
            ],
            "entities": [
                {"entity_name": "Entity A", "source_id": "doc-1", "file_path": "doc1.pdf"},
                {"entity_name": "Entity B", "source_id": "doc-3", "file_path": "doc3.pdf"},
            ],
            "relationships": [
                {"relation": "rel1", "source_id": "doc-2", "file_path": "doc2.pdf"}
            ],
            "references": [
                {"file_path": "doc1.pdf"},
                {"file_path": "other.pdf"},
            ],
        },
    }

    filtered = filter_by_session(rag_response, ["doc-1"])

    chunks = filtered["data"]["chunks"]
    entities = filtered["data"]["entities"]
    relationships = filtered["data"]["relationships"]
    references = filtered["data"]["references"]

    assert len(chunks) == 1 and chunks[0]["source_id"] == "doc-1"
    assert len(entities) == 1 and entities[0]["source_id"] == "doc-1"
    assert len(relationships) == 0
    assert references == [{"file_path": "doc1.pdf"}]


def test_filter_by_session_returns_original_when_no_data():
    result = filter_by_session({}, ["doc-1"])
    assert result == {}
