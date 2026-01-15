

# Copilot Instructions — Cygnus Pyramid (2026)

## Purpose
A short, actionable guide to help AI coding agents be productive in Cygnus Pyramid. Focus on repository-specific workflows, sensitive areas, and integration patterns so suggestions are precise and safe.

---

## Quickstart & Core Workflows (practical)
- **Python & venv:** Prefer Python 3.12. Create a venv and install deps:
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install -r requirements.txt`
- **Model infra (two options):**
  - **Ollama:** `ollama pull <model>`; set `OLLAMA_BASE_URL` and `OLLAMA_MODEL` env vars.
  - **llama.cpp (local GGUF):** `./start_servers.sh` — builds `llama.cpp` if missing and launches servers on **8081–8083**. Note: the script expects `models/Falcon-H1-1.5B-Deep-Instruct-Q5_K.gguf` by default (edit `start_servers.sh` if you use a different model).
- One-step launch (starts LLMs, backend, frontend): `./launch.sh`
- Backend only: `python -m backend.main` (default port 8001)
- Agent panel (dev): `python -m agent_panel.app` or `uvicorn agent_panel.app:app --reload --host 0.0.0.0 --port 8001`
- Frontend (vite): `cd frontend/frontend && npm install && npm run dev` (default port 5173). Note: this repo contains a nested canonical frontend at `frontend/frontend`; top-level frontend artifacts were archived to `frontend/_legacy/` to avoid conflicting copies.
- **Docker:** `docker-compose up` uses `docker-compose.yaml` to run `cygnus` + `chromadb`. The compose file sets `OLLAMA_HOST` and maps volumes for `memory_stores` and `config`.
- Check health endpoints:
  - LLM servers: `http://localhost:8081/health`
  - Backend: `http://localhost:8001/api/status`
- Logs & PIDs: `./logs/*.log` and `./logs/*.pid` (created by start/launch scripts)

---

## Model backends & env vars
- **Select backend:** `MODEL_BACKEND_TYPE` (options: `openai`, `ollama`, `vllm`, `llama.cpp`, `onnx`, `torch`). Defaults and required keys are implemented in `agent_panel/app.py::initialize_model_backend()`.
- **Common env vars:** `OPENAI_API_KEY`, `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `LLAMA_CPP_MODEL_PATH`, `VLLM_MODEL_PATH`, `ONNX_MODEL_PATH`, `TORCH_MODEL_PATH`, `MAX_TOKENS`, `TEMPERATURE`.
- **Per-agent model URLs (backend):** `backend/config.py` lists defaults for `katz`, `dogz`, `cygnus` (default: `http://localhost:8082`, `http://localhost:8083`, `http://localhost:8081`). Update there when using different host/ports.
- When using Ollama or remote visual models, check health endpoints (visual model URL in `backend/config.py` or `agent_panel/app.py`).

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
- `./start_servers.sh` — builds (if needed) and starts local `llama-server` instances (ports 8081/8082/8083). Requires `cmake` and system build tools; supports CUDA builds when available.
- `./launch.sh` — convenient launcher (may start LLM servers, backend, frontend and open the browser).
- `python -m backend.main` — start backend (port 8001 default). Use `PORT` env var to override.
- `python -m agent_panel.app` or `uvicorn ...` — start agent panel.
- `cd frontend && npm run dev` — start Vite frontend.
- **Tests:** `pytest` at repo root or per-component (`pytest backend/` / `pytest agent_panel/`). Some tests require optional dependencies or external API keys (see `backend/tests/*` and `backend/tests/conftest.py`).
- `llama.cpp` has separate C/C++ tests under `llama.cpp/tests/` — follow `llama.cpp/AGENTS.md` before making changes there.

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

