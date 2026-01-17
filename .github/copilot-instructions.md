

# Copilot Instructions — Cygnus Pyramid (2026)

Purpose: Short, actionable guidance so AI coding agents can be immediately productive in this repo.

Quickstart
- Python: Prefer **Python 3.12**. Create a venv and install deps:
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install -r requirements.txt`
- Launch infra:
  - Local LLMs: `./start_servers.sh` (starts local `llama-server` instances on **8081–8083**)
  - Full stack: `./launch.sh` (LLMs + backend + frontend)
  - Backend only: `python -m backend.main` (default port **8001**)
  - Agent UI: `python -m agent_panel.app` or `uvicorn agent_panel.app:app --reload --host 0.0.0.0 --port 8001`
  - Frontend dev: `cd frontend/frontend && npm install && npm run dev` (Vite, default **5173**)
- Health endpoints: LLM `http://localhost:8081/health`, Backend `http://localhost:8001/api/status`
- Logs: `./logs/*.log` and `./logs/*.pid`

Model backends & env
- `MODEL_BACKEND_TYPE` selects model backend (`openai|ollama|vllm|llama.cpp|onnx|torch`).
- Key env vars: `OPENAI_API_KEY`, `EMBEDDER_TYPE` ("stub"|"openai"), `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `LLAMA_CPP_MODEL_PATH`, `VLLM_MODEL_PATH`, `ONNX_MODEL_PATH`, `TORCH_MODEL_PATH`, `MAX_TOKENS`, `TEMPERATURE`.
- Per-agent URLs and fallbacks live in `backend/config.py` and `agent_panel/app.py::initialize_model_backend()`.

Architecture (concise)
- `backend/`: orchestration (debates, ResearchScheduler, memory, MCP server). See `backend/main.py` for LLM client wiring.
- `llama.cpp/`: local model server and low-level C/C++ tests (see `llama.cpp/AGENTS.md` for contributor rules).
- `backend/`: orchestration (debates, ResearchScheduler, memory, MCP adapters). See `backend/main.py`, `backend/debate/orchestrator.py`, and `backend/api/*` for wiring and APIs.
- Memory-first design: most outputs are persisted (FTS + embeddings). See `backend/memory/`, `memory_stores/` (episodic stores), `backend/db/sqlite_store.py` and `ARCHITECTURE.md` for intended Central Library components (Chroma/SQLite/JSON archives).

Project-specific patterns
- Agents: In this tree, agents are usually provided as model client objects with an async `respond(...)` coroutine (see `backend/main.py` `_StubAgent` and `backend/debate/orchestrator.py::_ask_agent`). Add new agents by providing objects following that pattern and wiring them into the orchestrator.
- Deliberation: Debates are scheduled/run by `DebateOrchestrator` (events: `debate_started`, `round_started`, `statement`, `snapshot_taken`, `scores_assigned`, `allcall_round`, `debate_finished`). Use `/api/debate/start`, `/api/debate/{id}/allcall`, `/api/debate/{id}/camera-capture` for common operations.
- Native control & cameras: Camera capture endpoints fall back to a harmless placeholder when OpenCV or hardware is unavailable—helpful for CI.
- Caution areas: Files that operate on host OS or external services (camera capture, crawlers, any code that touches the filesystem or network) require additional tests and maintainer sign-off before deployment.
- Note: Some higher-level components referenced in docs (e.g., `agent_panel/`) are not present in this workspace; treat those references as optional/legacy and prefer `frontend/` + `backend/api` for current UI/integration points.

Scripts, tests & infra notes
- Scripts: `./start_servers.sh`, `./launch.sh`, `./stop_servers.sh`.
- Tests: Run `pytest` at repo root or per-component (`pytest backend/`, `pytest agent_panel/`). Frontend e2e tests use Playwright and may need browser installs or extra env vars—see `.github/workflows/e2e-playwright.yml` and `frontend/tests/e2e` for examples. Some tests rely on optional dependencies or API keys; inspect failing tests or the test file for requirements and mock/stub external services when possible.
- `llama.cpp` contains its own test suite and contributor rules—**do not** author large AI-generated changes in that area (`llama.cpp/AGENTS.md`).

Admin & migration
- Admin endpoint protection: set `ADMIN_API_KEY` to require header `X-ADMIN-KEY` on admin endpoints (see `backend/api/admin_routes.py`).
- Import memory: POST `/api/admin/import-memory` triggers migration from `./memory_stores` into the SQLite DB. The migration utility is expected at `./scripts/migrate_memory_to_db.py`—the endpoint will return a graceful error if the script is missing.

Development tips
- Model backend: to run components with a specific model backend set `MODEL_BACKEND_TYPE` and the matching env vars (e.g., `MODEL_BACKEND_TYPE=openai OPENAI_API_KEY=... python -m backend.main`). For a local llama-based stack, start servers with `./start_servers.sh` before launching the full stack (`./launch.sh`).

Safety & contributor rules (required)
- For `llama.cpp/`, AI-generated PRs are prohibited—AI can assist but not author major changes; always disclose AI usage.
- Any edits touching `agent_panel/native_control.py`, crawler/sandbox, or other host-operating code must include tests and explicit maintainer review.
- Prefer small, focused PRs with clear descriptions and tests; do not merge on maintainer's behalf.

Practical examples
- Persisting embeddings: `agent_panel/app.py::persist_agent_embedding_anchors()` calls `memory_store.upsert_embedding(...)`.
- Start a debate: `POST /api/debate/start` (handler in `backend/main.py`).
- Execute a native action: `POST /api/native/action` with payload `{ "type": "mouse_click", "x": 100, "y": 200 }`.

Port conflicts & Ollama note ⚠️
- Ollama snap may auto-bind **8081–8083**, which prevents `./start_servers.sh` from starting local servers. Diagnose with `ss`/`lsof` and stop/disable Ollama (`sudo snap stop ollama && sudo snap disable ollama`) if needed.
- `start_servers.sh` includes a preflight check and supports `FORCE_KILL_PORTS=1` for emergency use—prefer stopping services manually first.

If anything is unclear
- Ask focused, component-specific questions and include the file paths you plan to change (e.g., `agent_panel/native_control.py`, `backend/main.py`).

(Kept concise and focused—ask if you want additional examples or deeper file pointers.)markdown
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

