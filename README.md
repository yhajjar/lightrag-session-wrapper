# LightRAG Session Wrapper

Stateless FastAPI service that adds session-aware isolation on top of the LightRAG HTTP API. Each session tracks its own document identifiers, filters query responses, and supports lifecycle management (list, delete, auto-expire).

## Features
- Session-scoped document pools with metadata tracking
- Upload, query, list, and delete APIs mirroring the technical specification
- Background cleanup of expired sessions based on configurable TTL
- LightRAG query filtering to prevent cross-session leakage
- Docker image ready for docker-compose deployment

## Configuration
Set the following environment variables as needed:

```
LIGHTRAG_URL=http://lightrag:9621
LIGHTRAG_API_KEY=your_api_key
WRAPPER_PORT=8000
SESSION_MAX_AGE_HOURS=24
SESSION_CLEANUP_INTERVAL=3600
CORS_ORIGINS=*
```

## Local Development
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

> If `python3 -m venv` fails, install the `python3-venv` package (Ubuntu/Debian) or the equivalent for your distribution.

## Testing
```
pytest
```

The FastAPI application exposes an OpenAPI schema at `/docs` and `/openapi.json` when running locally.
