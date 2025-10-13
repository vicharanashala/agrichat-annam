# AgriChat Lightweight Backend API

This guide summarizes the HTTP and streaming endpoints exposed by the lightweight AgriChat backend (`agrichat-backend`). Share it with frontend engineers so they can integrate without reading the entire Python codebase.

---

## Base URL & Versioning

The service does not version its routes yet; all paths below are current as of October 2025.

Deployments typically expose FastAPI behind a reverse proxy. When tunneling through Serveo the public base is:

```
https://agrichat.serveo.net
```

Set the frontend `API_BASE` to `<BASE_URL>/api`. System endpoints (`/`, `/health`) live at the root.

---

## Authentication & Required Headers

* **Authentication:** No token-based auth. An optional CSV-backed login exists for admin tooling (`POST /api/auth/login`).
* **Device Id header:** Most session endpoints require a stable device identifier. Provide it in the request body and, for GET/DELETE routes, via header.
  * Header: `X-Device-Id: <uuid-or-random-string>`
  * Fallback query string: `?device_id=...` (mainly for CSV download/testing)
* **Content-Type:** JSON requests use `application/json`. File uploads (`/transcribe-audio`, `/session/{id}/rate`, `/test-database-toggle`) must be `multipart/form-data`.

Common error payloads:

| HTTP Code | JSON body | Meaning |
|-----------|-----------|---------|
| 400 | `{ "error": "Device ID is required" }` | Missing or empty `device_id` |
| 403 | `{ "error": "Device authorization failed" }` | Mismatched device for the session |
| 403 | `{ "error": "Session is archived, missing or unauthorized" }` | Attempted to continue a session that is archived/unknown |
| 404 | `{ "error": "Session not found" }` | No document matched the supplied `session_id` |
| 503 | `{ "error": "Session storage temporarily unavailable" }` | MongoDB is offline, only transient for in-flight requests |

---

## System Endpoints (no `/api` prefix)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health banner: `{ "message": "AgriChat backend is running." }` |
| `GET` | `/health` | Aggregated health for MongoDB, ChromaDB, and Ollama. |
| `OPTIONS` | `/{any}` | Manual CORS handler used by preflight requests. No need to call directly. |

### `GET /health` response
```json
{
  "status": "healthy",              // healthy | degraded | unhealthy
  "timestamp": "2025-10-13T10:46:21+05:30",
  "checks": {
    "mongo": { "status": "ok", "detail": "connected" },
    "chroma": { "status": "ok", "detail": "available", "path": "/app/chromaDb" },
    "ollama": { "status": "ok", "detail": "reachable", "endpoint": "http://localhost:11434" }
  }
}
```

---

## Auth Endpoint

| Method | Path | Body | Notes |
|--------|------|------|-------|
| `POST` | `/api/auth/login` | `{ "username": "...", "password": "..." }` | Validates against `agrichat-backend/users.csv`. Returns `AuthResponse`. |

Success response:
```json
{
  "authenticated": true,
  "username": "admin",
  "role": "maintainer",
  "full_name": "Agri Admin",
  "message": "Login successful"
}
```

Invalid credentials:
```json
{
  "authenticated": false,
  "username": null,
  "role": null,
  "full_name": null,
  "message": "Invalid username or password"
}
```

---

## Core Chat & Session Endpoints (`/api` prefix)

### 1. `POST /api/query`
Starts a brand-new chat session.

```json
{
  "question": "Which fertilizer suits paddy at tillering stage?",
  "device_id": "8c6e7f4f-1da5-49e8-9a59-b96c68c650ee",
  "state": "Tamil Nadu",                // optional
  "language": "English",               // optional (default English)
  "database_config": {                  // optional overrides
    "golden_enabled": true,
    "pops_enabled": true,
    "llm_enabled": true
  }
}
```

Successful 200 response:
```json
{
  "session": {
    "session_id": "3f709994-8dbe-49f5-9c18-4f51f078a81f",
    "timestamp": "2025-10-13T10:50:02+05:30",
    "messages": [
      {
        "question": "Which fertilizer suits paddy at tillering stage?",
        "thinking": "...optional hidden chain-of-thought...",
        "final_answer": "<p>Apply 45 kg N per hectare ...</p>",
        "answer": "<p>Apply 45 kg N per hectare ...</p>",
        "pipeline_metadata": { "database_config": { "golden_enabled": true } },
        "research_data": [
          {
            "source": "Golden Database",
            "content_preview": "Best practice for tillering stage ...",
            "metadata": { "state": "Tamil Nadu", "cosine": 0.81 }
          }
        ]
      }
    ],
    "crop": "unknown",
    "state": "Tamil Nadu",
    "status": "active",
    "language": "English",
    "has_unread": true,
    "device_id": "8c6e7f4f-1da5-49e8-9a59-b96c68c650ee",
    "recommendations": []
  }
}
```

### 2. `POST /api/session/{session_id}/query`
Continue an existing chat session.

* Request body mirrors `POST /api/query` but `language` is optional and defaults to the stored session language.
* Returns `{ "session": <updated-session-document> }` with the new message appended.
* Errors: `403` if the session belongs to a different device or is archived.

### 3. `POST /api/query/thinking-stream`
Server-Sent Events (SSE) stream for a single-shot answer with live "thinking" updates. Example event sequence:

```
event: message
data: {"type":"session_start","session_id":"..."}

event: message
data: {"type":"thinking_start"}

event: message
data: {"type":"thinking_complete","thinking":"Step-by-step reasoning..."}

event: message
data: {"type":"answer_start"}

event: message
data: {"type":"answer","answer":"<p>...</p>","source":"Golden Database","confidence":0.82}

event: message
data: {"type":"stream_end"}
```

Frontend hint: use the Fetch API with `EventSource` or `ReadableStream` to consume the SSE channel. The request body is the same JSON payload as `POST /api/query`.

### 4. Session management endpoints

| Method | Path | Purpose | Notes |
|--------|------|---------|-------|
| `GET` | `/api/sessions` | List last 20 sessions for the calling device. | Requires `X-Device-Id`; returns `{ "sessions": [...] }`. |
| `GET` | `/api/session/{session_id}` | Fetch a single session. Marks it as read (`has_unread=false`). |
| `DELETE` | `/api/delete-session/{session_id}` | Remove a session permanently. |
| `POST` | `/api/toggle-status/{session_id}/{status}` | Flip between `active` and `archived`. `status` is the current state; backend returns `{ "status": "archived" }` or `"active"`. |
| `POST` | `/api/session/{session_id}/rate` | Form-encoded rating for a specific answer. Fields: `question_index` (int), `rating` (`"up" | "down"`), `device_id`. |
| `GET` | `/api/export/csv/{session_id}` | Download the session as CSV (question/answer/rating/timestamp). |
| `POST` | `/api/update-language` | Bulk update `language` and `state` for every session owned by the device. Body: `{ "device_id": "...", "state": "AP", "language": "Telugu" }`. |

### 5. Utility endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/transcribe-audio` | `multipart/form-data` with `file` (audio bytes) and optional `language`. Returns `{ "transcript": "..." }` generated by Whisper. |
| `POST` | `/api/test-database-toggle` | Form-data quick check to see which database toggles are enabled. Returns `{ "question": "...", "overrides": { ... } }`. Mainly for internal tools. |

---

## Request/Response Tips

* **Database overrides:** `database_config` accepts any subset of `DatabaseToggleConfig` (see `app_core/models.py`). Unknown keys are ignored.
* **HTML vs text answers:** Each message stores `answer` / `final_answer` in HTML. To display plaintext, strip tags on the client or use the `answer_plain` field from the SSE payload when provided.
* **Thinking trace:** The backend keeps the full reasoning in `reasoning_trace` (array of steps) and `thinking` string. These may be hidden from the farmer UI but are useful for diagnostics.
* **Research data:** Each message may include `research_data` entries summarizing the top knowledge-base hits, including cosine similarity when confidence sharing is enabled.

---

## Environment Integrations

The backend honors the following env vars (helpful during integration/testing):

| Variable | Default | Purpose |
|----------|---------|---------|
| `MONGO_URI` | `mongodb://localhost:27017/agrichat` | Mongo connection for session persistence. |
| `OLLAMA_HOST` | `localhost:11434` | Host:port for the Ollama server. |
| `BACKEND_RELOAD` | `true` (docker compose) | Run `uvicorn` with hot reload for dev when `true`; production uses Gunicorn. |
| `USE_HTTPS` | `false` | Switches Gunicorn to `8443` with local self-signed certs when `true`. |

---

## Console payload demo

Use the helper script in `agrichat-backend/scripts/send_sample_query.py` to inspect the request payload and resulting response without writing custom tooling. The script prints the JSON sent to `/api/query`, makes the call, and pretty-prints the response body with status details.

```
python scripts/send_sample_query.py \
  --base-url https://agrichat.serveo.net \
  --question "What should I spray for cotton bollworms?" \
  --state "Tamil Nadu" \
  --language "English"
```

Arguments are optional; omit them to use the built-in defaults. Add `--device-id` to reuse a known device or leave it blank to auto-generate a UUID. Use `--no-golden`, `--no-pops`, or `--no-llm` to toggle knowledge sources for debugging.

---

## Minimal Integration Checklist

1. Configure the frontend `API_BASE` (and `STREAM_ENDPOINT`) to point at the chosen backend origin.
2. Generate or persist a per-device UUID; send it as `device_id` and `X-Device-Id` as required.
3. Use `POST /api/query` for the first message, store the returned `session_id`, then `POST /api/session/{id}/query` for follow-ups.
4. For live typing indicators or progressive responses, consume the SSE stream at `POST /api/query/thinking-stream`.
5. Surface session history using `GET /api/sessions` and `GET /api/session/{id}`.
6. Offer optional extras: CSV export, rating buttons, language preference update.

With this reference your frontend team can wire the lightweight backend without reverse-engineering responses. Reach out if you add new routes so the document stays current.