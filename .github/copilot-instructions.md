

# Copilot Instructions — Cygnus Pyramid (2026)

## Purpose
A short, actionable guide to help AI coding agents be productive in Cygnus Pyramid. Focus on repository-specific workflows, sensitive areas, and integration patterns so suggestions are precise and safe.

---

## Quickstart & Core Workflows (practical)
- Install Python deps (root + components as needed):
  - `pip install -r requirements.txt`
  - In `agent_panel/` optionally: `pip install -r agent_panel/requirements.txt`
- Model infra (two options):
  - Ollama: `ollama pull <model>` and set `OLLAMA_BASE_URL`/`OLLAMA_MODEL` env vars
  - `llama.cpp` servers (recommended for local full-stack testing): run `./start_servers.sh` (builds/starts servers on ports 8081/8082/8083)
- One-step launch (starts LLMs, backend, frontend): `./launch.sh`
- Backend only: `python -m backend.main` (default port 8001)
- Agent panel (dev): `python -m agent_panel.app` or `uvicorn agent_panel.app:app --reload --host 0.0.0.0 --port 8001`
- Frontend (vite): `cd frontend && npm run dev` (default port 5173)
- Check health endpoints:
  - LLM servers: `http://localhost:8081/health`
  - Backend: `http://localhost:8001/api/status`
- Logs & PIDs: `./logs/*.log` and `./logs/*.pid` (created by start/launch scripts)

---

## Model backends & env vars
- `MODEL_BACKEND_TYPE` selects the backend (openai, ollama, vllm, llama.cpp, onnx, torch).
- Common env vars:
  - `OPENAI_API_KEY`, `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `LLAMA_CPP_MODEL_PATH`, `VLLM_MODEL_PATH`, `ONNX_MODEL_PATH`, `TORCH_MODEL_PATH`, `MAX_TOKENS`, `TEMPERATURE`.
- See `agent_panel/app.py::initialize_model_backend()` for exact behavior and fallbacks.

---

## Architecture (big picture)
- `backend/`: core orchestration (debates, ResearchScheduler, MCP server). LLM clients and broadcast plumbing are created in `backend/main.py` and passed to agents/orchestrator.
- `agent_panel/`: control UI, agents definitions, native control, vision, per-agent memory (SQLite `agent_panel_memory.db`). Avatars and emotion state managed here.
- `llama.cpp/`: local model server; contains strict contributor rules (`llama.cpp/AGENTS.md`).
- Memory: backend uses `MemorySystem("./memory_stores")`; agent panel uses `agent_panel_memory.db`. Embedding anchors are persisted at startup in `agent_panel.app.persist_agent_embedding_anchors()`.

---

## Project-specific patterns & examples
- **Memory-first:** Persist every agent output (FTS + embeddings). Files: `backend/memory/`, `agent_panel/memory.py`.
- **Add an agent:** Follow `agent_panel/agents.py` — create `AgentDefinition`, add `embedding_anchors`, and ensure startup code persists anchors and registers avatars (`agent_panel/app.py`).
- **Native control is high-risk:** `agent_panel/native_control.py` exposes mouse/keyboard/screenshot/ocr. Carefully test and ask maintainers before changing. Actions can be executed via `/api/native/action`.
- **Deliberation & approvals:** The deliberation pipeline can produce `awaiting_approval` states; UI (deliberation view) and endpoints expect operator decisions. Check `/api/core/*` routes.
- **Crawler & sandbox:** Autonomous crawlers and sandboxes exist but are disabled by default; domain configs are in `agent_panel/app.py::DOMAIN_CONFIGS`.

---

## Scripts & developer commands to reference
- `./start_servers.sh` — builds (if needed) and starts `llama-server` instances (ports 8081/8082/8083).
- `./launch.sh` — convenient launcher (starts LLM servers, backend, frontend and opens the browser).
- `python -m backend.main` — start backend (port 8001 default).
- `python -m agent_panel.app` or `uvicorn ...` — start agent panel.
- `cd frontend && npm run dev` — start Vite frontend.
- Component tests: run pytest inside component folders (e.g., `pytest agent_panel/`, `pytest backend/`). `llama.cpp` has its own C/C++/Python tests — see `llama.cpp/tools/server/tests`.

---

## Safety & contributor policy (must follow)
- `llama.cpp/AGENTS.md` is authoritative: **AI-generated PRs are forbidden in the `llama.cpp` area** — AI can assist but not author major changes; disclose AI usage when required.
- Changes to: `agent_panel/native_control.py`, crawler/sandbox code, or anything that operates on the host require maintainer review and clear tests.
- Playground is metadata-only. Admin actions are protected by `PLAYGROUND_ADMIN_KEY`.

---

```markdown
# Copilot Instructions — Cygnus Pyramid (2026)

Purpose: concise, repo-specific guidance for AI coding agents to be productive safely.

Quickstart
- Install deps: `pip install -r requirements.txt` (root). Optionally `pip install -r agent_panel/requirements.txt`.
- Start full stack: `./launch.sh` (LLM servers + backend + frontend).
- Start local LLM servers: `./start_servers.sh` (llama.cpp servers on 8081/8082/8083).
- Run backend only: `python -m backend.main` (default port 8001).
- Run agent panel dev: `python -m agent_panel.app` or `uvicorn agent_panel.app:app --reload --host 0.0.0.0 --port 8001`.
- Frontend dev: `cd frontend && npm run dev` (Vite, default 5173).

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

If something is unclear
- Ask component-specific questions and include the exact file(s) you plan to change.

``` 

