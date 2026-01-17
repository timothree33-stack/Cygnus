

# Copilot Instructions — Cygnus Pyramid (concise)

Purpose: immediate, repo-specific guidance to make small, safe, high-impact changes.

Quickstart — dev loop ✅
- Python: prefer `3.12`. Create venv + deps:
  - `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- Full stack (dev): `./launch.sh start` (use `START_LLAMAS=1` to also run local LLMs).
- Backend only: `MODEL_BACKEND_TYPE=openai OPENAI_API_KEY=... uvicorn backend.main:app --reload --port 8001`
- Local LLMs: `./start_servers.sh` → servers bind to ports **8081–8083**.
- Frontend dev: `cd frontend/frontend && npm install && npm run dev` (Vite — 5173).

Key files & patterns (read these first) 🔎
- `backend/main.py` — app entrypoint, `_StubAgent` examples, orchestration wiring.
- `backend/debate/orchestrator.py` — debate flow; add agents here or follow its event hooks.
- `backend/memory/`, `memory_stores/`, `backend/db/sqlite_store.py` — memory-first persistence (FTS + embeddings).
- `backend/api/*` — REST surface; see `backend/api/admin_routes.py` for admin-only endpoints.
- `llama.cpp/AGENTS.md` — contributor rules (protected area).

How to add an agent (checklist) ✳️
1. Minimal interface (required)
   - Implement a model-client **class** with an async `respond(**kwargs)` coroutine that returns a `str` or serializable `dict`.
   - Keep the method signature compatible with `_StubAgent.respond(topic: str, round: int, **kwargs)` in `backend/main.py`.

2. Behavior & persistence (recommended)
   - Return a concise `text` field; include optional metadata (`score`, `sources`) when available.
   - If the agent produces anchors, call `memory_store.upsert_embedding(agent_id, key, text, vector=None)` immediately after generation.

3. Tests (must-have)
   - Unit-test `respond()` for deterministic inputs (use `pytest.mark.asyncio`).
   - Add an integration-style smoke test that passes the agent into `DebateOrchestrator` or the `_ask_agent` helper if present.

4. Wiring & deployment
   - Register the agent as a drop-in by passing an instance to `DebateOrchestrator(...)` (see `backend/main.py`).
   - Expose any operator controls via `backend/api/*` only when the agent is stable; prefer feature flags for experimental agents.

5. Safety & review
   - If the agent performs OS/network actions or persists external data, add explicit tests and request maintainer review (see high-risk areas).
   - Do NOT add AI-authored changes to `llama.cpp/` — follow `llama.cpp/AGENTS.md`.

Quick scaffold (copy-paste) 🔧
- File: `backend/agents/sample_agent.py`
  - Implements `class SampleAgent` with `async def respond(self, **kwargs)` and a short unit test in `backend/tests/test_sample_agent.py`.

Environment & backends — concrete tips ⚙️
- Switch backends: set `MODEL_BACKEND_TYPE` (options: `openai|ollama|vllm|llama.cpp|onnx|torch`).
- Local llama gotchas: Ollama snap often occupies 8081–8083 — run `ss -ltnp|grep :8081` before starting.
- Admin routes require `ADMIN_API_KEY` (header `X-ADMIN-KEY`) — see `backend/api/admin_routes.py`.

Tests & CI — focused commands 🧪
- Unit: `pytest backend/` (or `pytest` at repo root).
- Frontend e2e: `npx playwright install && npm run test:e2e` from `frontend/frontend`.
- If adding infra-level changes, include a focused test and a README snippet that documents how to run it locally.

High-risk areas — do NOT auto-merge ⚠️
- `llama.cpp/` (AI-generated PRs prohibited)
- `agent_panel/native_control.py` and any code that performs OS/network/camera actions
- Crawlers, sandboxing, and any code that writes to host paths — require maintainer sign-off + tests.

Debug checklist (fast) 🔧
1. Check logs: `./logs/backend.log` and `./logs/frontend.log` (+ `.pid` files).
2. Health endpoints: `http://localhost:8081/health` (LLM), `http://localhost:8001/api/status` (backend).
3. Port conflicts: `sudo ss -ltnp | grep :8081` / `lsof -iTCP:8081 -sTCP:LISTEN`.
4. DB/migration: `POST /api/admin/import-memory` uses `./scripts/migrate_memory_to_db.py`.

PR checklist for maintainers ✅
- Small, focused change (one behavioral change or one new feature).
- Add/modify unit tests and include reproduction steps in PR description.
- Mention maintainers for sensitive areas and document required env to run locally.
- If touching `llama.cpp/` or `native_control.py`, include explicit reviewer request and tests.

Search shortcuts (save time) ⌘
- grep for: `DebateOrchestrator`, `MemorySystem`, `MODEL_BACKEND_TYPE`, `persist_agent_embedding_anchors`, `start_servers.sh`.

If something is missing or unclear
- Tell me which component or file you want expanded (e.g., `backend/debate/orchestrator.py`) and I'll add a 1‑page editing checklist or a focused code example.

— End of concise guide. Ask if you want this expanded into separate CONTRIBUTING / developer-FAQ snippets.
# Copilot Instructions — Cygnus Pyramid (2026)

Purpose: concise, repo-specific guidance for AI coding agents to be productive safely.

Quickstart
- Install deps: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt` (root). Optionally `pip install -r agent_panel/requirements.txt`.
- Start full stack: `./launch.sh` (LLM servers + backend + frontend).
- Run backend only (dev): `uvicorn backend.main:app --reload --host 0.0.0.0 --port 8001` or `python -m backend.main` (default port 8001).
- Start local LLM servers (if using `llama.cpp`): `./start_servers.sh` (ports **8081–8083**) — beware Ollama may conflict.
- Frontend dev: `cd frontend/frontend && npm install && npm run dev` (Vite, default **5173**).

How to run a local dev loop (compact) 🔁
1. Create venv & install deps:
   - `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
2. Start local LLM (if using `llama.cpp` backend):
   - `./start_servers.sh`
3. Start backend with auto-reload (example using OpenAI):
   - `MODEL_BACKEND_TYPE=openai OPENAI_API_KEY=... uvicorn backend.main:app --reload --port 8001`
   - Or run with local llama: `MODEL_BACKEND_TYPE=llama.cpp uvicorn backend.main:app --reload --port 8001`
4. Start frontend:
   - `cd frontend/frontend && npm run dev`
5. Quick test + reload cycle:
   - Make code change → refresh frontend or send HTTP request to backend (e.g., `curl http://localhost:8001/api/status`) → observe logs.

Notes
- Use `npx playwright install` before running frontend e2e tests (`npm run test:e2e`). See `.github/workflows/e2e-playwright.yml` for CI details.
- If ports 8081–8083 are in use (Ollama), stop/disable Ollama or choose a different model backend.

Health & logs
- LLM health: `http://localhost:8081/health`.
- Backend status: `http://localhost:8001/api/status`.
- Logs & PIDs: `./logs/*.log`, `./logs/*.pid`.

Model backends (practical)
- `MODEL_BACKEND_TYPE` chooses backend: `openai|ollama|vllm|llama.cpp|onnx|torch`.
- Common env vars: `OPENAI_API_KEY`, `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `LLAMA_CPP_MODEL_PATH`, `VLLM_MODEL_PATH`, `ONNX_MODEL_PATH`, `TORCH_MODEL_PATH`, `MAX_TOKENS`, `TEMPERATURE`.
- Implementation/fallbacks: see `agent_panel/app.py::initialize_model_backend()`.

Architecture (high level)
- `backend/`: orchestration, debate scheduler, LLM clients wired in `backend/main.py`.
- `agent_panel/`: control UI, agent defs (`agent_panel/agents.py`), native control (`agent_panel/native_control.py`), vision, and per-agent memory (`agent_panel_memory.db`).
- `llama.cpp/`: local model server runtime and tests (`llama.cpp/tools/server/tests`).
- Memory: backend uses `MemorySystem("./memory_stores")`; agent panel persists embedding anchors in `agent_panel/app.py::persist_agent_embedding_anchors()` and stores agent memory in `agent_panel_memory.db`.

Project-specific patterns
- Memory-first: most agent outputs are persisted (FTS + embeddings). Look at `backend/memory/` and `agent_panel/memory.py` for examples.
- Adding agents: follow `agent_panel/agents.py` — create `AgentDefinition`, provide `embedding_anchors`, and ensure anchors are persisted on startup (`agent_panel/app.py`).
- Deliberation flow: pipeline can yield `awaiting_approval` states; operator decisions are expected via UI and `/api/core/*` endpoints.
- Native control is high-risk: `agent_panel/native_control.py` exposes OS actions (mouse/keyboard/screenshot/OCR). Any changes require maintainer review and tests. Native actions are callable via `POST /api/native/action`.
- Autonomous crawlers/sandbox are disabled by default; domain configs in `agent_panel/app.py::DOMAIN_CONFIGS`.

Integration & external dependencies
- Local LLMs: `llama.cpp` servers (start with `./start_servers.sh`) or Ollama (set `OLLAMA_BASE_URL`/`OLLAMA_MODEL`).
  - **Note:** The Ollama snap can auto-launch and bind ports **8081–8083**, preventing local `llama-server` instances from starting. If you plan to use local servers, stop/remove Ollama first: `sudo snap stop ollama && sudo snap remove ollama` (or `sudo snap disable ollama` to keep it installed but inactive).
- Telemetry/metrics: agent panel exposes `/metrics` when `prometheus_client` is installed (see `agent_panel/app.py`).

Scripts & tests
- `./start_servers.sh`, `./launch.sh`, `./stop_servers.sh` for infra lifecycle.
- Run component tests with `pytest agent_panel/` or `pytest backend/`.

Safety & contributor rules (must follow)
- `llama.cpp/AGENTS.md` is authoritative: AI-generated PRs are forbidden in `llama.cpp` — AI may assist but must not author major changes there.
- Changes touching `agent_panel/native_control.py`, crawler/sandbox, or any host-operating code require maintainer approval and explicit tests.

Examples (explicit)
- Persist anchors: see `agent_panel/app.py::persist_agent_embedding_anchors()` — uses `memory_store.upsert_embedding(agent.id, key, text, vector=None)`.
- Start a debate: `POST /api/debate/start` (handler in `backend/main.py`).
- Execute native action: `POST /api/native/action` with payload `{ "type": "mouse_click", "x": 100, "y": 200 }` — test carefully.

What AI agents should do
- Prefer small, focused PRs that include tests and mention maintainers in PR description.
- For sensitive areas, propose patches but flag for human review; do not merge on behalf of maintainers.

FAQ — Port conflicts (8081–8083)  ⚠️
- Symptom: `start_servers.sh` fails with "couldn't bind HTTP server socket, port: 8081" — usually another LLM service is already listening (common culprits: Ollama snap, `bettercap`, `nessusd`).
- Diagnose:
  - `sudo ss -ltnp | grep :8081`
  - `sudo lsof -nP -iTCP:8081 -sTCP:LISTEN`
- If Ollama is running:
  - `sudo snap stop ollama` and then `sudo snap remove ollama` (or `sudo snap disable ollama` to keep it installed but inactive).
- If a system service is the culprit (e.g., `bettercap`, `nessusd`):
  - `sudo systemctl stop <service>`
  - `sudo systemctl disable <service>`
  - `sudo systemctl mask <service>`
- Quick force-free (use with caution): `sudo fuser -k 8081/tcp`
- Start servers from repo root: `cd /home/sophia/cygnus-pyramid && ./start_servers.sh`
- The `start_servers.sh` script now performs a preflight port check and supports `FORCE_KILL_PORTS=1 ./start_servers.sh` to override and kill port owners (use only after manual stop/disable attempts).

If something is unclear
- Ask component-specific questions and include the exact file(s) you plan to change.

``` 

