
from __future__ import annotations
from .memory import get_memory_store
from .avatars import get_avatar_manager
from .agents import get_all_agents, get_agent_by_name, get_team_by_name
"""Agent Control Panel - FastAPI Web Application.

Main entry point for the web UI with:
- Mini orchestrator with deliberation engine
- Two-round deliberation between AJ and Tesla
- Intent classification and approval gates
- Real-time avatar updates
- Voice integration
- WebSocket for live updates
"""

import asyncio
import json
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Header, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import time
import logging
from pathlib import Path
from .native_control import get_native_controller
from .vision import get_vision_module, VisionFrame

# NEW: Crawler Integration
from agent_panel.crawler import (
    AutonomousDomainExplorer,
    DomainExplorationConfig
)

# NEW: Core integration
from .integration import get_core_integration, get_core_router

# NEW: Model backend initialization
def initialize_model_backend():
    """Initialize the appropriate model backend based on MODEL_BACKEND_TYPE."""
    backend_type = os.getenv("MODEL_BACKEND_TYPE", "openai").lower()
    
    if backend_type == "openai":
        from agents.inference_backends import OpenAIBackend
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY required for OpenAI backend")
        return OpenAIBackend(api_key=api_key, model="gpt-4")
    
    elif backend_type == "ollama":
        from agents.inference_backends import OllamaBackend
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model = os.getenv("OLLAMA_MODEL", "llama2")
        return OllamaBackend(base_url=base_url, model=model)
    
    elif backend_type == "vllm":
        from agents.inference_backends import VLLMBackend
        model_path = os.getenv("VLLM_MODEL_PATH", "meta-llama/Llama-2-7b-hf")
        return VLLMBackend(model=model_path)
    
    elif backend_type == "llama.cpp":
        from agents.inference_backends import LlamaBackend
        model_path = os.getenv("LLAMA_CPP_MODEL_PATH", "./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
        return LlamaBackend(model_path=model_path)
    
    elif backend_type == "onnx":
        from agents.inference_backends import ONNXRuntimeBackend, InferenceConfig
        model_path = os.getenv("ONNX_MODEL_PATH", "meta-llama/Llama-2-7b")
        config = InferenceConfig(
            model_path=model_path,
            device="cuda" if os.getenv("CUDA_AVAILABLE", "true").lower() == "true" else "cpu",
            max_tokens=int(os.getenv("MAX_TOKENS", "512")),
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
        )
        return ONNXRuntimeBackend(config)
    
    elif backend_type == "torch":
        from agents.inference_backends import TorchBackend, InferenceConfig
        model_path = os.getenv("TORCH_MODEL_PATH", "meta-llama/Llama-2-7b-hf")
        config = InferenceConfig(
            model_path=model_path,
            device="cuda" if os.getenv("CUDA_AVAILABLE", "true").lower() == "true" else "cpu",
            max_tokens=int(os.getenv("MAX_TOKENS", "512")),
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
        )
        return TorchBackend(config)
    
    else:
        # Fallback to SimpleModelBackend for development
        from integration import SimpleModelBackend
        return SimpleModelBackend()

# On-call API nest
from .api.oncall import router as oncall_router
from .api.osint import router as osint_router
from .api.library import router as library_router
from .api.toolkit import router as toolkit_router
from .api.sandbox import router as sandbox_router
from .api.dashboard import router as dashboard_router
# from playground import api as playground_api  # TODO: playground module not found


# =============================================================================
# App Setup
# =============================================================================

app = FastAPI(
    title="Agent Control Panel",
    description="Multi-agent AI control interface with deliberation, voice, avatars, and native control",
    version="2.0.0",
)

# Module logger for broadcast/debugging
logger = logging.getLogger("agent_panel")
# Ensure a timestamped, structured-friendly formatter for the module logger.
if not logger.handlers:
    handler = logging.StreamHandler()
    # Always use JSON formatter for logs in this runtime.
    class JsonFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            try:
                payload = getattr(record, "payload", None)
                if payload is None:
                    payload = {
                        "ts": time.time(),
                        "level": record.levelname,
                        "name": record.name,
                        "message": record.getMessage(),
                        "pid": os.getpid(),
                    }
                return json.dumps(payload, default=str)
            except Exception:
                return super().format(record)

    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Simple in-memory telemetry counters (thread-safe)
_telemetry_lock = None
try:
    import threading
    _telemetry_lock = threading.Lock()
except Exception:
    _telemetry_lock = None

_telemetry_counters = {
    "messages_received": 0,
    "agent_responses_broadcast": 0,
    "broadcast_failures": 0,
}

# per-agent in-memory telemetry (simple counters)
_telemetry_per_agent = {}

# Optional Prometheus integration
_prometheus_available = False
try:
    from prometheus_client import Counter as PromCounter, Histogram as PromHistogram, generate_latest, CONTENT_TYPE_LATEST
    _prometheus_available = True
    messages_received_counter = PromCounter(
        "agent_panel_messages_received_total", "Total messages received by Agent Panel", ["agent_name"]
    )
    responses_broadcast_counter = PromCounter(
        "agent_panel_agent_responses_broadcast_total", "Total agent responses broadcast", ["agent_name"]
    )
    broadcast_failures_counter = PromCounter(
        "agent_panel_broadcast_failures_total", "Total broadcast failures", ["agent_name"]
    )
    # Latency histograms
    response_latency_histogram = PromHistogram(
        "agent_panel_response_latency_seconds", "Latency for generating an agent response (seconds)", ['agent_name'], buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
    )
    websocket_send_latency_histogram = PromHistogram(
        "agent_panel_ws_send_latency_seconds", "Latency for sending websocket messages (seconds)", ['agent_name'], buckets=(0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5)
    )
    # Internal per-step histogram: captures timing for model-selection/delegation/decoding/postprocess
    response_internal_histogram = PromHistogram(
        "agent_panel_response_internal_seconds",
        "Latency for internal response generation steps (seconds)",
        ["agent_name", "step"],
        buckets=(0.0005, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0),
    )
except Exception:
    _prometheus_available = False


def emit_event(event: str, level: str = "info", **fields: Any) -> None:
    """Emit a structured JSON log event alongside the normal logger output.

    Fields will include an ISO timestamp, event name, and any provided metadata.
    This keeps backward-compatible human logs and produces a JSON line for ingestion.
    """
    try:
        payload = {
            "ts": time.time(),
            "event": event,
            "level": level,
            "service": "agent_panel",
            "pid": os.getpid(),
        }
        # Merge fields, ensuring JSON serializable basic types
        for k, v in fields.items():
            try:
                json.dumps({k: v})
                payload[k] = v
            except Exception:
                payload[k] = str(v)

        # Log via `payload` so JSON formatter can emit clean JSON lines.
        if level == "debug":
            logger.debug("", extra={"payload": payload})
        elif level == "warning":
            logger.warning("", extra={"payload": payload})
        elif level == "error":
            logger.error("", extra={"payload": payload})
        else:
            logger.info("", extra={"payload": payload})
    except Exception:
        logger.exception("Failed to emit structured event: %s", event)

# Serve static files (JS/CSS/assets) from the local `static/` directory
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")

# CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
memory_store = get_memory_store(
    os.getenv("AGENT_MEMORY_DB", "agent_panel_memory.db")
)
avatar_manager = get_avatar_manager()


def _emotion_value(emotion_state: Any) -> str:
    # Normalize different possible return types from avatar manager
    try:
        if isinstance(emotion_state, str):
            return emotion_state
        if hasattr(emotion_state, "primary"):
            primary = getattr(emotion_state, "primary")
            if isinstance(primary, str):
                return primary
            if hasattr(primary, "value"):
                return getattr(primary, "value")
        return str(emotion_state)
    except Exception:
        return "neutral"

# Do not instantiate the native controller at import time. Creating the controller
# may import pyautogui or other GUI backends and should be opt-in via env var.
# Enable with: export ENABLE_NATIVE_CONTROL=1
native_controller = None
vision_module = get_vision_module()

# =========== CRAWLER INITIALIZATION ===========
# Initialize autonomous crawler explorer
logger = logging.getLogger("agent_panel")
crawler_explorer = None  # Temporarily disabled to avoid model downloads
logger.info("ℹ️  Crawler explorer disabled for startup speed")

# Domain configurations for crawler
DOMAIN_CONFIGS = {
    'quantum_computing': {
        'seed_urls': [
            'https://quantum.ibm.com',
            'https://www.rigetti.com',
            'https://www.d-wave.com',
        ],
        'keywords': [
            'quantum computing', 'quantum algorithm', 'qubit',
            'superposition', 'entanglement', 'quantum gate',
            'quantum circuit', 'quantum error correction'
        ],
        'max_crawls': 10,
        'max_iterations': 100,
        'exploration_strategy': 'HYBRID',
    },
    'machine_learning': {
        'seed_urls': [
            'https://pytorch.org',
            'https://tensorflow.org',
            'https://arxiv.org/list/cs.LG',
        ],
        'keywords': [
            'neural network', 'deep learning', 'training',
            'model', 'algorithm', 'data', 'optimization',
            'gradient descent', 'backpropagation'
        ],
        'max_crawls': 8,
        'max_iterations': 80,
        'exploration_strategy': 'TARGETED',
    },
    'cybersecurity': {
        'seed_urls': [
            'https://owasp.org',
            'https://cwe.mitre.org',
            'https://nvd.nist.gov',
        ],
        'keywords': [
            'vulnerability', 'exploit', 'attack', 'defense',
            'cryptography', 'authentication', 'authorization',
            'security', 'threat model'
        ],
        'max_crawls': 10,
        'max_iterations': 100,
        'exploration_strategy': 'HYBRID',
    },
    'web3': {
        'seed_urls': [
            'https://ethereum.org',
            'https://bitcoin.org',
            'https://cardano.org',
        ],
        'keywords': [
            'blockchain', 'cryptocurrency', 'smart contract',
            'decentralized', 'Web3', 'DeFi', 'NFT',
            'consensus', 'distributed ledger'
        ],
        'max_crawls': 8,
        'max_iterations': 80,
        'exploration_strategy': 'DEPTH_FIRST',
    },
}
logger.info(f"📚 Registered {len(DOMAIN_CONFIGS)} domains for crawler")

# Initialize core integration (Mini, Deliberation, etc.)
core_integration = get_core_integration()
app.include_router(get_core_router())

# Include on-call API nest
app.include_router(oncall_router)

# Include OSINT API services
app.include_router(osint_router)

# Include API Library (catalog for agents)
app.include_router(library_router)

# Include Cybersecurity Toolkit
app.include_router(toolkit_router)

# Include Sandbox Training Ground
app.include_router(sandbox_router)

# Include Playground simulated challenges and Lux agent
# app.include_router(playground_api.router, prefix="/playground")  # TODO: playground module not found

# Register agents with avatar manager
for name, agent in get_all_agents().items():
    avatar_manager.register_agent(
        agent.id,
        agent.personality.preferred_colors,
        agent.personality.avatar_style
    )


@app.on_event("startup")
async def persist_agent_embedding_anchors():
    """On startup, persist any agent-declared embedding anchors into memory.

    This is intentionally lightweight: anchors are stored with empty vectors
    as placeholders so later workers can compute and update real embeddings.
    """
    import logging
    logger = logging.getLogger("agent_panel")
    agents = list(get_all_agents().values())
    total = 0
    for a in agents:
        anchors = getattr(a, "embedding_anchors", []) or []
        for idx, text in enumerate(anchors):
            key = f"anchor:{idx}"
            try:
                memory_store.upsert_embedding(a.id, key, text, vector=None)
                total += 1
            except Exception as e:
                logger.exception(f"Failed to persist anchor for {a.name}: {e}")
    logger.info(f"Persisted {total} agent embedding anchors at startup")
    
# Active WebSocket connections per conversation
    mcp_url = os.getenv("MCP_URL")
    if mcp_url:
        try:
            await _push_anchors_to_mcp(agents, mcp_url)
        except Exception as e:
            logger.exception(f"Failed to push anchors to MCP at startup: {e}")


async def _push_anchors_to_mcp(agents: List[AgentDefinition], mcp_url: str) -> Dict[str, int]:
    """Push agent embedding anchors to external MCP `/ingest` endpoint.

    Returns counts per agent.
    """
    import httpx
    mcp_api_key = os.getenv("MCP_API_KEY")
    url_base = mcp_url.rstrip("/")
    headers = {}
    if mcp_api_key:
        headers["x-api-key"] = mcp_api_key

    counts: Dict[str, int] = {}
    async with httpx.AsyncClient(timeout=30.0) as client:
        for a in agents:
            anchors = getattr(a, "embedding_anchors", []) or []
            c = 0
            for idx, text in enumerate(anchors):
                payload = {
                    "id": f"{a.id}:anchor:{idx}",
                    "text": text,
                    "metadata": {"agent_name": a.name, "anchor_index": idx},
                }
                try:
                    resp = await client.post(f"{url_base}/ingest", json=payload, headers=headers)
                    if resp.status_code in (200, 201):
                        c += 1
                    else:
                        # 400 may mean already exists; ignore
                        if resp.status_code == 400:
                            c += 0
                        else:
                            # log but continue
                            print(f"MCP ingest failed for {payload['id']}: {resp.status_code} {resp.text}")
                except Exception as e:
                    print(f"Failed to POST anchor to MCP for {a.name}: {e}")
            counts[a.name] = c
    return counts


@app.post("/mcp/sync")
async def mcp_sync(x_api_key: Optional[str] = Header(None)):
    """On-demand push of agent anchors to configured MCP server.

    Protect this endpoint with `MCP_SYNC_KEY` env var if set (send as `x-api-key`).
    """
    sync_key = os.getenv("MCP_SYNC_KEY")
    if sync_key and x_api_key != sync_key:
        raise HTTPException(status_code=401, detail="Invalid sync API key")
    mcp_url = os.getenv("MCP_URL")
    if not mcp_url:
        raise HTTPException(status_code=400, detail="MCP_URL not configured")
    agents = list(get_all_agents().values())
    counts = await _push_anchors_to_mcp(agents, mcp_url)
    return {"status": "ok", "counts": counts}

# Active WebSocket connections per conversation
connections: Dict[str, List[WebSocket]] = {}


# =============================================================================
# Pydantic Models
# =============================================================================

class ChatMessage(BaseModel):
    content: str
    agent_id: Optional[str] = None
    conversation_id: Optional[str] = None


class AgentResponse(BaseModel):
    agent_id: str
    agent_name: str
    content: str
    emotion: str
    avatar: str
    timestamp: float


class NativeAction(BaseModel):
    type: str
    params: Dict[str, Any]


# =============================================================================
# REST Endpoints
# =============================================================================

@app.get("/")
async def root():
    """Serve the main UI (redirect to Cygnus dashboard)."""
    return RedirectResponse(url="/static/dashboard/index.html")


@app.get("/dashboard")
async def dashboard():
    """Serve the Cygnus dashboard (compatibility route)."""
    return RedirectResponse(url="/static/dashboard/index.html")


@app.get("/api/agents")
async def list_agents():
    """List all available agents."""
    # Simplified: expose only the primary agent (MiniOrca) by default.
    # Teams (AJ/Tesla) and their prompts are intentionally hidden to
    # keep the UI focused on MiniOrca. Individual specialists can still
    # be queried directly by name using the opinion flow.
    primary = get_agent_by_name("MiniOrca") or None
    return {
        "primary": primary.to_dict() if primary else {},
        "teams": {},
    }


@app.get("/api/perception/status")
async def agent_panel_perception_status():
    """Proxy perception status from Cygnus backend for the Agent Panel UI."""
    import httpx
    backend_url = os.getenv("CYGNUS_BACKEND_URL", "http://localhost:8001")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{backend_url}/api/perception/status")
            return JSONResponse(content=resp.json(), status_code=resp.status_code)
    except Exception as e:
        return JSONResponse(content={"available": False, "error": str(e)}, status_code=503)


@app.post("/api/perception/mock")
async def agent_panel_perception_mock(type: str = "vision"):
    """Proxy to create a mock perception frame on the backend for testing."""
    import httpx
    backend_url = os.getenv("CYGNUS_BACKEND_URL", "http://localhost:8001")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(f"{backend_url}/api/perception/mock?type={type}")
            return JSONResponse(content=resp.json(), status_code=resp.status_code)
    except Exception as e:
        return JSONResponse(content={"available": False, "error": str(e)}, status_code=503)


@app.get("/api/agents/{agent_name}")
async def get_agent(agent_name: str):
    """Get details about a specific agent."""
    agent = get_agent_by_name(agent_name)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent {agent_name} not found")
    
    return {
        "agent": agent.to_dict(),
        "avatar": avatar_manager.get_avatar_base64(agent.id),
        "system_prompt": agent.get_system_prompt(),
    }


@app.get("/api/agents/{agent_name}/avatar")
async def get_agent_avatar(agent_name: str, size: int = 128):
    """Get agent's current avatar SVG."""
    agent = get_agent_by_name(agent_name)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent {agent_name} not found")
    
    return HTMLResponse(
        content=avatar_manager.get_avatar(agent.id, size),
        media_type="image/svg+xml"
    )


@app.post("/api/agents/{agent_name}/emotion")
async def set_agent_emotion(agent_name: str, emotion: str, intensity: float = 0.7):
    """Manually set an agent's emotion."""
    agent = get_agent_by_name(agent_name)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent {agent_name} not found")
    
    try:
        emotion_enum = Emotion(emotion)
        avatar_manager.set_emotion(agent.id, emotion_enum, intensity)
        return {"success": True, "avatar": avatar_manager.get_avatar_base64(agent.id)}
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid emotion: {emotion}")


@app.get("/api/conversations")
async def list_conversations(agent_id: Optional[str] = None, limit: int = 50):
    """List conversations."""
    conversations = memory_store.list_conversations(agent_id, limit)
    return {"conversations": [c.to_dict() for c in conversations]}


@app.post("/api/conversations")
async def create_conversation(agent_name: str, title: Optional[str] = None):
    """Create a new conversation with an agent."""
    agent = get_agent_by_name(agent_name)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent {agent_name} not found")
    
    conv = memory_store.create_conversation(agent.id, title)
    return conv.to_dict()


@app.get("/api/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get a conversation with messages."""
    conv = memory_store.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {
        "conversation": conv.to_dict(),
        "messages": [m.to_dict() for m in conv.messages],
    }


@app.post("/api/conversations/{conversation_id}/messages")
async def add_message(conversation_id: str, message: ChatMessage, speak: bool = False):
    """Add a message to a conversation (for demo purposes)."""
    # Lower verbosity: detailed fields are debug-level; keep high-level events at info.
    logger.debug("add_message called: conv=%s speak=%s content=%s", conversation_id, speak, (message.content[:200] if message and message.content else ""))
    emit_event("add_message_called", level="debug", conversation_id=conversation_id, speak=bool(speak), content=(message.content[:200] if message and message.content else ""))
    try:
        if _telemetry_lock:
            with _telemetry_lock:
                _telemetry_counters["messages_received"] += 1
        else:
            _telemetry_counters["messages_received"] += 1
        # Prometheus per-agent increment will occur after we resolve the conversation's agent
    except Exception:
        logger.exception("Failed to increment telemetry counter")
    # In real implementation, this would call the actual AI model
    conv = memory_store.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Resolve agent name for telemetry/prometheus labeling
    agent_name_for_metrics = "unknown"
    try:
        for name, a in get_all_agents().items():
            if a.id == conv.agent_id:
                agent_name_for_metrics = a.name
                break
    except Exception:
        agent_name_for_metrics = "unknown"
    if _prometheus_available:
        try:
            messages_received_counter.labels(agent_name=agent_name_for_metrics).inc()
        except Exception:
            logger.exception("Failed to increment prometheus messages_received_counter with agent label")
    # also update per-agent in-memory telemetry for messages_received
    try:
        if _telemetry_lock:
            with _telemetry_lock:
                pa = _telemetry_per_agent.setdefault(agent_name_for_metrics, {"messages_received": 0, "agent_responses_broadcast": 0, "broadcast_failures": 0})
                pa["messages_received"] += 1
        else:
            pa = _telemetry_per_agent.setdefault(agent_name_for_metrics, {"messages_received": 0, "agent_responses_broadcast": 0, "broadcast_failures": 0})
            pa["messages_received"] += 1
    except Exception:
        logger.exception("Failed to update per-agent in-memory telemetry for messages_received")
    
    # Add user message
    user_msg = memory_store.add_message(
        conversation_id,
        conv.agent_id,
        "user",
        message.content
    )
    try:
        logger.debug("user_message added: id=%s conversation=%s role=%s content=%s", user_msg.id, user_msg.conversation_id, user_msg.role, (user_msg.content[:200] if user_msg and user_msg.content else ""))
        emit_event("user_message_added", level="debug", message_id=getattr(user_msg, 'id', None), conversation_id=getattr(user_msg, 'conversation_id', None), role=getattr(user_msg, 'role', None), content=(user_msg.content[:200] if user_msg and user_msg.content else ""))
    except Exception:
        logger.exception("Failed to log user_message details")
    
    # Simulate agent response (placeholder) only if speak=True
    if speak:
        agent = None
        for name, a in get_all_agents().items():
            if a.id == conv.agent_id:
                agent = a
                break

        # If the conversation's agent_id isn't found, fallback to primary MiniOrca
        if not agent:
            agent = get_agent_by_name("MiniOrca")

        if agent:
            logger.debug("Generating agent response for agent=%s (id=%s)", agent.name, agent.id)
            emit_event("agent_response_generation_start", level="debug", agent_name=agent.name, agent_id=agent.id, conversation_id=conversation_id)
            # Start response timer
            resp_start = time.time()
            # Instrument internal generation steps (model selection, deliberation, decoding, post-process)
            model_select_start = time.time()
            # (placeholder) model selection would occur here
            model_select_dur = time.time() - model_select_start
            if _prometheus_available:
                try:
                    response_internal_histogram.labels(agent_name=agent.name, step="model_selection").observe(model_select_dur)
                except Exception:
                    logger.exception("Failed to observe model_selection histogram")

            # Deliberation / emotion update step
            deliberation_start = time.time()
            emotion_state = avatar_manager.update_emotion(agent.id, message.content)
            deliberation_dur = time.time() - deliberation_start
            if _prometheus_available:
                try:
                    response_internal_histogram.labels(agent_name=agent.name, step="deliberation").observe(deliberation_dur)
                except Exception:
                    logger.exception("Failed to observe deliberation histogram")

            # Placeholder response (decoding/response generation)
            decoding_start = time.time()
            response_content = f"[{agent.name}] I received your message: '{message.content}'"
            decoding_dur = time.time() - decoding_start
            if _prometheus_available:
                try:
                    response_internal_histogram.labels(agent_name=agent.name, step="decoding").observe(decoding_dur)
                except Exception:
                    logger.exception("Failed to observe decoding histogram")

            # Persist agent message (post-process / storage)
            postproc_start = time.time()
            agent_msg = memory_store.add_message(
                conversation_id,
                agent.id,
                "agent",
                response_content
            )
            postproc_dur = time.time() - postproc_start
            if _prometheus_available:
                try:
                    response_internal_histogram.labels(agent_name=agent.name, step="post_process").observe(postproc_dur)
                except Exception:
                    logger.exception("Failed to observe post_process histogram")
            try:
                logger.debug("agent_message added: id=%s conversation=%s role=%s content=%s", agent_msg.id, agent_msg.conversation_id, agent_msg.role, (agent_msg.content[:200] if agent_msg and agent_msg.content else ""))
                emit_event("agent_message_added", level="debug", message_id=getattr(agent_msg, 'id', None), conversation_id=getattr(agent_msg, 'conversation_id', None), role=getattr(agent_msg, 'role', None), content=(agent_msg.content[:200] if agent_msg and agent_msg.content else ""))
            except Exception:
                logger.exception("Failed to log agent_message details")

            # Prepare agent response payload (for REST return and websocket broadcast)
            agent_response_payload = {
                "message": agent_msg.to_dict(),
                "emotion": _emotion_value(emotion_state),
                "avatar": avatar_manager.get_avatar_base64(agent.id),
            }

            # Broadcast to any active websocket connections for this conversation
            try:
                response_data = {
                    "type": "agent_response",
                    "agent_name": agent.name,
                    "content": agent_msg.content,
                    "emotion": _emotion_value(emotion_state),
                    "avatar": avatar_manager.get_avatar_base64(agent.id),
                    "message": agent_msg.to_dict(),
                }

                conns = connections.get(conversation_id, [])
                logger.info("Broadcasting agent_response (REST path) to %d connections for conv=%s", len(conns), conversation_id)
                logger.debug("response_data keys=%s", list(response_data.keys()))
                emit_event("broadcast_attempt", level="info", conversation_id=conversation_id, connections=len(conns), agent_name=agent.name)
                # send sequentially; failures are logged but do not break the REST response
                for idx, conn in enumerate(list(conns)):
                    try:
                        send_start = time.time()
                        await conn.send_text(json.dumps(response_data))
                        send_dur = time.time() - send_start
                        logger.info("Successfully sent agent_response to connection #%d for conv=%s", idx, conversation_id)
                        emit_event("broadcast_success", level="info", conversation_id=conversation_id, connection_index=idx, agent_name=agent.name, send_latency_seconds=send_dur)
                        try:
                            if _telemetry_lock:
                                with _telemetry_lock:
                                    _telemetry_counters["agent_responses_broadcast"] += 1
                                    # per-agent
                                    pa = _telemetry_per_agent.setdefault(agent.name, {"messages_received": 0, "agent_responses_broadcast": 0, "broadcast_failures": 0})
                                    pa["agent_responses_broadcast"] += 1
                            else:
                                _telemetry_counters["agent_responses_broadcast"] += 1
                                pa = _telemetry_per_agent.setdefault(agent.name, {"messages_received": 0, "agent_responses_broadcast": 0, "broadcast_failures": 0})
                                pa["agent_responses_broadcast"] += 1
                            if _prometheus_available:
                                try:
                                    responses_broadcast_counter.labels(agent_name=agent.name).inc()
                                    websocket_send_latency_histogram.labels(agent_name=agent.name).observe(send_dur)
                                except Exception:
                                    logger.exception("Failed to increment prometheus responses_broadcast_counter or histogram")
                        except Exception:
                            logger.exception("Failed to increment broadcast telemetry")
                    except Exception as e:
                        logger.exception("Failed to send websocket message to a connection (REST broadcast): %s", e)
                        emit_event("broadcast_failure", level="error", conversation_id=conversation_id, connection_index=idx, error=str(e), agent_name=getattr(agent, 'name', None))
                        try:
                            if _telemetry_lock:
                                with _telemetry_lock:
                                    _telemetry_counters["broadcast_failures"] += 1
                                    pa = _telemetry_per_agent.setdefault(agent.name, {"messages_received": 0, "agent_responses_broadcast": 0, "broadcast_failures": 0})
                                    pa["broadcast_failures"] += 1
                            else:
                                _telemetry_counters["broadcast_failures"] += 1
                                pa = _telemetry_per_agent.setdefault(agent.name, {"messages_received": 0, "agent_responses_broadcast": 0, "broadcast_failures": 0})
                                pa["broadcast_failures"] += 1
                            if _prometheus_available:
                                try:
                                    broadcast_failures_counter.labels(agent_name=agent.name).inc()
                                except Exception:
                                    logger.exception("Failed to increment prometheus broadcast_failures_counter")
                        except Exception:
                            logger.exception("Failed to increment broadcast failure telemetry")
            except Exception:
                logger.exception("Unexpected error while broadcasting agent response for conv=%s", conversation_id)

            # Observe total response generation latency (if prometheus available)
            try:
                resp_dur = time.time() - resp_start
                emit_event("agent_response_generated", level="info", conversation_id=conversation_id, agent_name=agent.name, response_latency_seconds=resp_dur)
                if _prometheus_available:
                    try:
                        response_latency_histogram.labels(agent_name=agent.name).observe(resp_dur)
                    except Exception:
                        logger.exception("Failed to observe response latency histogram")
            except Exception:
                logger.exception("Failed to record response latency")

            return {
                "user_message": user_msg.to_dict(),
                "agent_response": agent_response_payload,
            }

    return {"user_message": user_msg.to_dict()}


@app.post("/api/native/action")
async def execute_native_action(action: NativeAction, request: Request):
    """Execute a native control action (MiniOrca only).

    Security: Native control is disabled by default. To enable, set
    `ENABLE_NATIVE_CONTROL=1` and set a strong `NATIVE_CONTROL_AUTH_KEY`.
    Clients must include `X-NATIVE-AUTH: <key>` header with the key.
    """
    controller = get_native_controller()
    if controller is None:
        raise HTTPException(status_code=403, detail="Native control is disabled. Set ENABLE_NATIVE_CONTROL=1 to enable.")

    expected_key = os.getenv("NATIVE_CONTROL_AUTH_KEY")
    if not expected_key:
        raise HTTPException(status_code=403, detail="Native control requires NATIVE_CONTROL_AUTH_KEY to be set in the environment.")

    provided = request.headers.get("X-NATIVE-AUTH")
    if provided != expected_key:
        raise HTTPException(status_code=403, detail="Unauthorized to execute native actions.")

    caller_meta = {"ip": request.client.host if request.client else None, "headers_present": bool(provided)}

    result = await controller.execute_action({
        "type": action.type,
        **action.params
    }, caller_meta=caller_meta)
    return result


@app.get("/api/native/screen")
async def describe_screen(request: Request):
    """Get current screen state."""
    controller = get_native_controller()
    if controller is None:
        raise HTTPException(status_code=403, detail="Native control is disabled. Set ENABLE_NATIVE_CONTROL=1 to enable.")

    expected_key = os.getenv("NATIVE_CONTROL_AUTH_KEY")
    if not expected_key:
        raise HTTPException(status_code=403, detail="Native control requires NATIVE_CONTROL_AUTH_KEY to be set in the environment.")

    provided = request.headers.get("X-NATIVE-AUTH")
    if provided != expected_key:
        raise HTTPException(status_code=403, detail="Unauthorized to access native control.")

    return await controller.describe_screen()


@app.post("/api/native/screenshot")
async def take_screenshot(request: Request):
    """Take a screenshot of the current screen."""
    controller = get_native_controller()
    if controller is None:
        raise HTTPException(status_code=403, detail="Native control is disabled. Set ENABLE_NATIVE_CONTROL=1 to enable.")

    expected_key = os.getenv("NATIVE_CONTROL_AUTH_KEY")
    if not expected_key:
        raise HTTPException(status_code=403, detail="Native control requires NATIVE_CONTROL_AUTH_KEY to be set in the environment.")

    provided = request.headers.get("X-NATIVE-AUTH")
    if provided != expected_key:
        raise HTTPException(status_code=403, detail="Unauthorized to access native control.")

    img = await controller.screen.screenshot_base64()
    return {"image": img}


# =============================================================================
# Vision Endpoints - Giving the AI "Eyes"
# =============================================================================

@app.get("/api/vision/camera")
async def capture_camera():
    """Capture an image from the camera.
    
    This gives the AI the ability to "see" the physical world.
    Like how MAI-UI uses screenshots, the camera provides visual context.
    """
    frame = await vision_module.see_camera()
    if not frame:
        return {"error": "Camera not available. Install a camera to give the AI eyes!", "available": False}
    
    return {
        "available": True,
        "image": frame.to_base64(),
        "width": frame.width,
        "height": frame.height,
        "source": "camera",
        "timestamp": frame.timestamp,
    }


@app.get("/api/vision/screen")
async def capture_screen():
    """Capture the current screen.
    
    Following MAI-UI's approach of using screenshots as visual input.
    """
    frame = await vision_module.see_screen()
    if not frame:
        return {"error": "Screen capture failed", "available": False}
    
    return {
        "available": True,
        "image": frame.to_base64(),
        "width": frame.width,
        "height": frame.height,
        "source": "screen",
        "timestamp": frame.timestamp,
    }


@app.get("/api/vision/both")
async def capture_both():
    """Capture both camera and screen - full visual context.
    
    This provides the AI with complete awareness:
    - Camera: what's happening in the physical world
    - Screen: what's on the computer display
    """
    camera_frame = await vision_module.see_camera()
    screen_frame = await vision_module.see_screen()
    
    return {
        "camera": {
            "available": camera_frame is not None,
            "image": camera_frame.to_base64() if camera_frame else None,
            "width": camera_frame.width if camera_frame else None,
            "height": camera_frame.height if camera_frame else None,
        } if camera_frame else {"available": False},
        "screen": {
            "available": screen_frame is not None,
            "image": screen_frame.to_base64() if screen_frame else None,
            "width": screen_frame.width if screen_frame else None,
            "height": screen_frame.height if screen_frame else None,
        } if screen_frame else {"available": False},
    }


@app.get("/api/vision/status")
async def vision_status():
    """Check what vision capabilities are available.
    
    Philosophy: An AI needs eyes to effectively communicate about
    the visual world. This endpoint reports what 'senses' are available.
    """
    camera_frame = await vision_module.see_camera()
    has_camera = camera_frame is not None
    
    return {
        "camera_available": has_camera,
        "screen_capture_available": True,  # Always available on desktop
        "message": "Full vision capability!" if has_camera else "No camera detected. Screen capture only.",
        "recommendation": None if has_camera else "Install a webcam to give the AI eyes for the physical world.",
    }


@app.get("/api/telemetry")
async def get_telemetry():
    """Return simple in-memory telemetry counters for quick debugging/metrics."""
    try:
        if _telemetry_lock:
            with _telemetry_lock:
                data = dict(_telemetry_counters)
        else:
            data = dict(_telemetry_counters)
    except Exception:
        logger.exception("Failed to read telemetry counters")
        data = {k: None for k in _telemetry_counters.keys()}
    # Add a timestamp for freshness
    return {"timestamp": float(time.time()), "counters": data}


@app.get("/metrics")
async def metrics_endpoint():
    """Prometheus metrics endpoint (if prometheus_client is installed)."""
    if not _prometheus_available:
        raise HTTPException(status_code=501, detail="Prometheus client not available")
    return JSONResponse(content=generate_latest().decode(), media_type=CONTENT_TYPE_LATEST)


@app.get("/api/metrics_summary")
async def metrics_summary():
    """Return a JSON summary of telemetry and selected Prometheus metrics per-agent.

    This is a lightweight helper for UIs that want to display per-agent counters
    and basic histogram summaries (count/sum) without parsing Prometheus text.
    """
    summary = {
        "timestamp": time.time(),
        "telemetry": {
            "counters": dict(_telemetry_counters),
            "per_agent": dict(_telemetry_per_agent),
        },
        "prometheus": {},
    }
    if not _prometheus_available:
        return summary

    try:
        from prometheus_client.core import REGISTRY

        # initialize containers
        summary["prometheus"]["response_latency"] = {}
        summary["prometheus"]["ws_send_latency"] = {}
        summary["prometheus"]["response_internal"] = {}

        for metric in REGISTRY.collect():
            # histograms expose samples with suffixes _count and _sum and bucket samples
            if metric.name == "agent_panel_response_latency_seconds":
                for s in metric.samples:
                    # sample.name may be like agent_panel_response_latency_seconds_count
                    if s.name.endswith("_count"):
                        labels = dict(s.labels)
                        agent = labels.get("agent_name", "unknown")
                        summary["prometheus"]["response_latency"].setdefault(agent, {})["count"] = s.value
                    elif s.name.endswith("_sum"):
                        labels = dict(s.labels)
                        agent = labels.get("agent_name", "unknown")
                        summary["prometheus"]["response_latency"].setdefault(agent, {})["sum"] = s.value
            if metric.name == "agent_panel_ws_send_latency_seconds":
                for s in metric.samples:
                    if s.name.endswith("_count"):
                        labels = dict(s.labels)
                        agent = labels.get("agent_name", "unknown")
                        summary["prometheus"]["ws_send_latency"].setdefault(agent, {})["count"] = s.value
                    elif s.name.endswith("_sum"):
                        labels = dict(s.labels)
                        agent = labels.get("agent_name", "unknown")
                        summary["prometheus"]["ws_send_latency"].setdefault(agent, {})["sum"] = s.value
            if metric.name == "agent_panel_response_internal_seconds":
                # internal histogram has labels agent_name and step
                for s in metric.samples:
                    if s.name.endswith("_count") or s.name.endswith("_sum"):
                        labels = dict(s.labels)
                        agent = labels.get("agent_name", "unknown")
                        step = labels.get("step", "unknown")
                        container = summary["prometheus"]["response_internal"].setdefault(agent, {})
                        container.setdefault(step, {})
                        if s.name.endswith("_count"):
                            container[step]["count"] = s.value
                        else:
                            container[step]["sum"] = s.value
    except Exception:
        logger.exception("Failed to build prometheus summary")

    return summary


@app.get("/api/voice/test/{agent_name}")
async def test_voice(agent_name: str, text: str = "Hello, I am ready to assist you."):
    """Test an agent's voice (returns audio URL)."""
    agent = get_agent_by_name(agent_name)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent {agent_name} not found")
    
    voice = get_voice_for_agent(agent_name)
    try:
        audio_bytes = await voice.speak_bytes(text)
        import base64
        audio_b64 = base64.b64encode(audio_bytes).decode()
        return {"audio": f"data:audio/wav;base64,{audio_b64}"}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/memory/stats")
async def memory_stats(agent_id: Optional[str] = None):
    """Get memory statistics."""
    return memory_store.get_stats(agent_id)


@app.get("/api/memory/{agent_name}")
async def get_agent_memory(agent_name: str):
    """Get an agent's learned memory."""
    agent = get_agent_by_name(agent_name)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent {agent_name} not found")
    
    memory = memory_store.get_agent_memory(agent.id)
    return memory.to_dict()


# =============================================================================
# Crawler REST Endpoints (Part 2: WebSocket Integration)
# =============================================================================

@app.get("/api/crawler/domains")
async def list_available_domains():
    """Get list of available domains for crawler"""
    return {
        'domains': list(DOMAIN_CONFIGS.keys()),
        'count': len(DOMAIN_CONFIGS),
        'domains_config': {
            domain: {
                'seed_urls_count': len(config['seed_urls']),
                'keywords_count': len(config['keywords']),
            }
            for domain, config in DOMAIN_CONFIGS.items()
        }
    }


@app.post("/api/crawler/session/{domain}")
async def create_crawler_session(domain: str, query: str = None):
    """Create a new crawler session for a domain"""
    
    if domain not in DOMAIN_CONFIGS:
        raise HTTPException(status_code=404, detail=f"Domain '{domain}' not found")
    
    session_id = await start_crawler_session(domain, query or "")
    
    if not session_id:
        raise HTTPException(status_code=500, detail="Failed to start crawler session")
    
    return {
        'session_id': session_id,
        'domain': domain,
        'status': 'started',
        'timestamp': time.time(),
    }


@app.get("/api/crawler/session/{session_id}/status")
async def get_crawler_session_status(session_id: str):
    """Get status of a crawler session"""
    
    status = await get_crawler_status(session_id)
    
    if 'error' in status:
        raise HTTPException(status_code=404, detail=status['error'])
    
    return status


@app.get("/api/crawler/session/{session_id}/knowledge")
async def query_crawler_knowledge(session_id: str, query: str):
    """Query knowledge graph from a crawler session"""
    
    try:
        if not crawler_explorer:
            raise HTTPException(status_code=503, detail="Crawler not initialized")
        
        # Get domain from session
        status = await get_crawler_status(session_id)
        domain = status.get('domain')
        
        if not domain:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Query knowledge
        knowledge = await crawler_explorer.query_agent(
            query=query,
            domain=domain,
            max_results=5,
        )
        
        return knowledge
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/crawler/session/{session_id}")
async def stop_crawler_session(session_id: str):
    """Stop a crawler session"""
    
    try:
        if crawler_explorer:
            await crawler_explorer.stop_exploration(session_id)
            return {'status': 'stopped', 'session_id': session_id}
        raise HTTPException(status_code=503, detail="Crawler not initialized")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/{agent_name}")
async def chat_with_agent(agent_name: str, message: ChatMessage, opinion: Optional[str] = None, opinion_agents: Optional[str] = None):
    """Chat with an agent - returns response with emotion and optional voice.

    Query params:
    - `opinion=all` requests opinions from all agents.
    - `opinion_agents` is a comma-separated list of agent names to request opinions from.
    """
    agent = get_agent_by_name(agent_name)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent {agent_name} not found")

    content = message.content.strip()

    # Detect emotion from user input to influence response
    emotion_state = avatar_manager.update_emotion(agent.id, content)
    import re

    # Primary responder is MiniOrca by design
    primary_agent = get_agent_by_name("MiniOrca") or agent

    main_query = content
    opinion_targets = []

    # Query-param driven opinion requests
    if opinion == "all":
        opinion_targets = list(get_all_agents().values())
    elif opinion_agents:
        for n in opinion_agents.split(','):
            n = n.strip()
            a = get_agent_by_name(n)
            if a:
                opinion_targets.append(a)
    else:
        # Text-detection fallback: try to find opinion phrases and a mentioned agent
        lc = content.lower()
        opinion_phrase_found = any(p in lc for p in ("what do you think", "what does", "opinion of", "what is the opinion of"))
        if opinion_phrase_found:
            for agent_name in get_all_agents().keys():
                if re.search(rf"\b{re.escape(agent_name)}\b", content, re.IGNORECASE):
                    a = get_agent_by_name(agent_name)
                    if a:
                        opinion_targets.append(a)
                        # remove the agent mention and common opinion phrase to form the main query
                        try:
                            main_query = re.sub(rf"(?i)what do you think[\s,]*{re.escape(agent_name)}\??", "", content).strip()
                            main_query = re.sub(rf"(?i){re.escape(agent_name)}", "", main_query).strip()
                        except Exception:
                            main_query = re.sub(re.escape(agent_name), "", content, flags=re.IGNORECASE).strip()
                        break

            if not main_query:
                # try to pull last user message from conversation history if available
                if message.conversation_id:
                    conv = memory_store.get_conversation(message.conversation_id)
                    if conv:
                        for m in reversed(conv.messages):
                            if m.role == "user" and m.content.strip().lower() != content.lower():
                                main_query = m.content.strip()
                                break

    # Generate primary response (MiniOrca)
    response = await generate_agent_response(primary_agent, main_query)

    # Update emotion based on response
    response_emotion = avatar_manager.update_emotion(primary_agent.id, response)
    emotion = _emotion_value(response_emotion)
    
    # Generate voice audio
    audio_data = None
    try:
        voice = get_voice_for_agent(primary_agent.name)
        audio_bytes = await voice.speak_bytes(response, emotion)
        if audio_bytes:
            import base64
            audio_data = "data:audio/mp3;base64," + base64.b64encode(audio_bytes).decode()
    except Exception as e:
        print(f"Voice generation failed: {e}")
    
    result = {
        "primary": {
            "agent": primary_agent.name,
            "response": response,
            "emotion": emotion,
            "avatar": avatar_manager.get_avatar_base64(primary_agent.id),
            "audio": audio_data,
        }
    }

    # If opinions were requested, generate secondary opinion responses
    if opinion_targets:
        result["opinions"] = []
        for oa in opinion_targets:
            try:
                opinion_resp = await generate_agent_response(oa, main_query or content)
            except Exception:
                opinion_resp = "(opinion unavailable)"
            entry = {
                "agent": oa.name,
                "response": opinion_resp,
                "avatar": avatar_manager.get_avatar_base64(oa.id),
            }
            result["opinions"].append(entry)
        # Backwards-compatible single-opinion field when only one target requested
        if len(result["opinions"]) == 1:
            result["opinion"] = result["opinions"][0]

    return result


async def generate_agent_response(agent: AgentDefinition, user_input: str) -> str:
    """Async response generator with basic guardrails and optional delegation to Mini.

    Improvements:
    - Use word-boundary greeting detection to avoid substring matches.
    - Track recent greetings per-agent to avoid repeating the same long greeting.
    - For non-trivial inputs, delegate to Core/Mini pipeline for deliberation.
    """
    import re
    import time

    # Simple per-agent cooldown for verbose greetings
    if not hasattr(generate_agent_response, "_last_greeted"):
        generate_agent_response._last_greeted = {}

    user_strip = (user_input or "").strip()
    user_lower = user_strip.lower()
    name = agent.name

    # Empty input -> prompt for clarification
    if not user_strip:
        return "I didn't catch that — could you say that again?"

    # Greeting detection (word-boundary to avoid matching substrings)
    greet_pattern = re.compile(r"\b(hello|hi|hey|greetings)\b", re.IGNORECASE)
    if greet_pattern.search(user_lower):
        last = generate_agent_response._last_greeted.get(agent.id)
        now = time.time()
        # If greeted recently (5 minutes), give a short ack instead
        if last and (now - last) < 300:
            return f"{name} here — ready when you are. How can I help further?"
        generate_agent_response._last_greeted[agent.id] = now

        greetings = {
            "MiniOrca": f"Hello! I'm MiniOrca, your primary AI assistant. I can control your computer, browse the web, and help with any task. What would you like me to do?",
            "Axiom": f"Greetings! I'm Axiom, the code and quantum computing grandmaster. I know every programming language in existence. What shall we build today?",
            "Cipher": f"Hello. I'm Cipher, security and cryptography expert. I ensure systems are secure and code is bulletproof. How can I help protect your project?",
            "Vector": f"Hi there! I'm Vector, your data science and ML specialist. Need help with analysis, machine learning, or visualization? Let's dive into the data!",
            "Nexus": f"Hello! I'm Nexus, systems integration expert. I connect APIs, build microservices, and manage infrastructure. What systems do you need connected?",
            "Echo": f"Hi! I'm Echo, communication and NLP specialist. I help craft clear documentation and build language tools. How can I help you communicate better?",
            "Flux": f"Hello! I'm Flux, hardware and IoT specialist. From embedded systems to robotics, I bridge the physical and digital. What device are we working with?",
            "Prism": f"Hi! I'm Prism, visualization and UI/UX expert. I create beautiful, intuitive interfaces. Ready to make something stunning?",
            "Helix": f"Hello! I'm Helix, research synthesis specialist. I analyze papers, discover patterns, and connect insights across domains. What topic shall we explore?",
            "Volt": f"Hey! I'm Volt, performance optimization expert. I make systems fast and efficient. Got a bottleneck that needs eliminating?",
            "Spark": f"Hi there! I'm Spark, creative AI specialist. From image generation to creative coding, I bring imagination to life. What shall we create?",
        }
        return greetings.get(name, f"Hello! I'm {name}. How can I assist you today?")

    # Short-help keywords handled locally
    if any(k in user_lower for k in ["help", "what can you do", "capabilities"]):
        capabilities = {
            "MiniOrca": "I can: 🖱️ Control your mouse and keyboard, 📸 Take screenshots and read the screen, 🌐 Browse the web, 📁 Manage files, 📧 Handle emails, and execute any tool the team creates for me!",
            "Axiom": "I specialize in: 💻 ALL programming languages (Python, Rust, C++, JavaScript, Haskell, Assembly, and hundreds more), 🔬 Quantum computing (Qiskit, Cirq, Q#), 🧮 Algorithms and data structures, 🔧 Creating tools for MiniOrca!",
            "Cipher": "I handle: 🔐 Cryptography (AES, RSA, ECC, post-quantum), 🛡️ Security audits and penetration testing, 🔒 Secure coding practices, 🕵️ Threat analysis and incident response!",
            "Vector": "I work with: 📊 Data analysis and visualization, 🤖 Machine learning (PyTorch, TensorFlow, scikit-learn), 📈 Statistical modeling, 🔄 ML pipelines and data engineering!",
            "Nexus": "I build: 🔌 API integrations (REST, GraphQL, gRPC), ☁️ Cloud infrastructure (AWS, GCP, Azure), 🐳 Container orchestration (Docker, Kubernetes), 📬 Message systems (Kafka, RabbitMQ)!",
            "Echo": "I specialize in: 📝 Technical writing and documentation, 🗣️ NLP and language models, 💬 Conversation design, 🌍 Multilingual support and translation!",
        }
        return capabilities.get(name, f"I'm {name}, part of the agent team. I specialize in {agent.specialization}. Ask me anything in my domain!")

    # For anything beyond short help or greetings, delegate to Mini for deliberation
    try:
        core = get_core_integration()
        # Use Mini's pipeline for better intent classification and response
        mini_resp = await core.mini.process(user_input)
        if isinstance(mini_resp, dict):
            return mini_resp.get("message", str(mini_resp))
        # MiniResponse has .message
        return getattr(mini_resp, "message", str(mini_resp))
    except Exception:
        # Fallback to a polite contextual reply
        warmth = getattr(agent.personality, "warmth", 0.5)
        curiosity = getattr(agent.personality, "curiosity", 0.5)

        if warmth > 0.7:
            opener = "I'd love to help with that! "
        elif warmth > 0.4:
            opener = "Let me help you with that. "
        else:
            opener = "Understood. "

        followup = "This is a fascinating topic! " if curiosity > 0.7 else ""

        return f"{opener}{followup}Regarding '{user_input[:60]}{'...' if len(user_input) > 60 else ''}' - as {name}, I'll use my expertise in {agent.specialization.split(',')[0].strip()} to assist you. What specific aspect would you like me to focus on?"


# =============================================================================
# Crawler Helper Functions (Part 2: WebSocket Integration)
# =============================================================================

async def start_crawler_session(domain: str, user_query: str) -> Optional[str]:
    """
    Initialize autonomous crawler for a specific domain.
    
    Args:
        domain: Domain name (e.g., 'quantum_computing')
        user_query: User's query to guide crawler prioritization
        
    Returns:
        Session ID if successful, None otherwise
    """
    
    if not crawler_explorer or domain not in DOMAIN_CONFIGS:
        logger.warning(f"⚠️  Crawler not available or domain '{domain}' not configured")
        return None
    
    domain_config = DOMAIN_CONFIGS[domain]
    
    # Create exploration config
    try:
        config = DomainExplorationConfig(
            domain=domain,
            seed_urls=domain_config['seed_urls'],
            keywords=domain_config['keywords'],
            max_crawls_per_iteration=domain_config['max_crawls'],
            exploration_strategy=domain_config['exploration_strategy'],
            relevance_threshold=0.45,
        )
        
        # Initialize exploration
        session_id = await crawler_explorer.initialize_domain_exploration(config);
        
        # Start exploration in background (don't block the request)
        asyncio.create_task(
            crawler_explorer.continuous_exploration_loop(
                session_id,
                max_iterations=domain_config['max_iterations'],
                iteration_delay=1.0,
            )
        );
        
        logger.info(f"✅ Crawler session started: {session_id} for domain '{domain}'")
        return session_id
    except Exception as e:
        logger.error(f"❌ Failed to start crawler session: {e}")
        return None


async def get_crawler_status(session_id: str) -> Dict:
    """Get status of a crawler session"""
    try:
        if crawler_explorer:
            return crawler_explorer.get_session_status(session_id)
        return {'error': 'Crawler not initialized'}
    except Exception as e:
        return {'error': str(e)}


# =============================================================================
# WebSocket for Real-time Updates
# =============================================================================

@app.websocket("/ws/{conversation_id}")
async def websocket_endpoint(websocket: WebSocket, conversation_id: str):
    """WebSocket for real-time conversation updates."""
    await websocket.accept()
    
    if conversation_id not in connections:
        connections[conversation_id] = []
    connections[conversation_id].append(websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "chat":
                # Process chat message
                content = message.get("content", "")
                agent_id = message.get("agent_id")
                
                # Get agent
                agent = None
                for name, a in get_all_agents().items():
                    if a.id == agent_id:
                        agent = a
                        break
                
                if agent:
                    # Add to memory
                    memory_store.add_message(conversation_id, agent_id, "user", content)
                    
                    # Update emotion
                    emotion_state = avatar_manager.update_emotion(agent.id, content)
                    
                    # Placeholder response (would be real AI in production)
                    response = f"[{agent.name}] Processing: {content[:50]}..."
                    memory_store.add_message(conversation_id, agent_id, "agent", response)
                    
                    # Broadcast to all connections
                    response_data = {
                        "type": "agent_response",
                        "agent_name": agent.name,
                        "content": response,
                        "emotion": _emotion_value(emotion_state),
                        "avatar": avatar_manager.get_avatar_base64(agent.id),
                    }
                    
                    conns = connections.get(conversation_id, [])
                    logger.info("Broadcasting agent_response to %d connections for conv=%s", len(conns), conversation_id)
                    for conn in conns:
                        try:
                            await conn.send_text(json.dumps(response_data))
                        except Exception as e:
                            logger.exception("Failed to send websocket message to a connection: %s", e)
            
            elif message.get("type") == "emotion_update":
                # Manual emotion update
                agent_id = message.get("agent_id")
                emotion = message.get("emotion")
                
                if agent_id and emotion:
                    try:
                        avatar_manager.set_emotion(agent_id, Emotion(emotion))
                        
                        response_data = {
                            "type": "avatar_update",
                            "agent_id": agent_id,
                            "avatar": avatar_manager.get_avatar_base64(agent_id),
                        }
                        
                        conns = connections.get(conversation_id, [])
                        logger.info("Broadcasting avatar_update to %d connections for conv=%s", len(conns), conversation_id)
                        for conn in conns:
                            try:
                                await conn.send_text(json.dumps(response_data))
                            except Exception as e:
                                logger.exception("Failed to send websocket avatar update: %s", e)
                    except:
                        pass
                        
    except WebSocketDisconnect:
        if conversation_id in connections:
            connections[conversation_id].remove(websocket)


# =============================================================================
# Core Integration WebSocket
# =============================================================================

@app.websocket("/ws/core")
async def core_websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time core updates (deliberation, approvals, etc.)."""
    await get_core_integration().get_websocket_route()(websocket)


# =============================================================================
# Teams View - Shows all teams and agents
# =============================================================================

@app.get("/teams")
async def teams_view():
    """Serve the teams management UI."""
    return HTMLResponse(content=get_teams_html())


def get_teams_html() -> str:
    """Generate HTML for teams view."""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Agent Teams - Control Panel</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #21262d 100%);
            min-height: 100vh;
            color: #c9d1d9;
        }
        
        .container {
            max-width: 1600px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            text-align: center;
            margin-bottom: 30px;
        }
        
        header h1 {
            font-size: 2.5em;
            background: linear-gradient(135deg, #58a6ff, #f778ba, #ff7b72);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        
        header p {
            color: #8b949e;
            font-size: 1.1em;
        }
        
        nav {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-bottom: 30px;
        }
        
        nav a {
            color: #58a6ff;
            text-decoration: none;
            padding: 10px 20px;
            border: 1px solid #30363d;
            border-radius: 8px;
            transition: all 0.3s;
        }
        
        nav a:hover {
            background: #21262d;
            border-color: #58a6ff;
        }
        
        .primary-agent {
            background: linear-gradient(135deg, #1a3a4a 0%, #0d1117 100%);
            border: 2px solid #00CED1;
            border-radius: 16px;
            padding: 30px;
            margin-bottom: 30px;
            text-align: center;
        }
        
        .primary-agent h2 {
            color: #00CED1;
            font-size: 2em;
            margin-bottom: 10px;
        }
        
        .primary-agent .role {
            color: #48D1CC;
            font-size: 1.2em;
            margin-bottom: 15px;
        }
        
        .primary-agent .description {
            color: #8b949e;
            max-width: 600px;
            margin: 0 auto;
        }
        
        .teams-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 30px;
            margin-bottom: 30px;
        }
        
        @media (max-width: 1200px) {
            .teams-grid {
                grid-template-columns: 1fr;
            }
        }
        
        .team-card {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 16px;
            overflow: hidden;
        }
        
        .team-header {
            padding: 20px;
            border-bottom: 1px solid #30363d;
        }
        
        .team-header.aj {
            background: linear-gradient(135deg, #1e3a5f 0%, #161b22 100%);
            border-left: 4px solid #4169E1;
        }
        
        .team-header.tesla {
            background: linear-gradient(135deg, #3d3a1e 0%, #161b22 100%);
            border-left: 4px solid #FFD700;
        }
        
        .team-header h3 {
            font-size: 1.8em;
            margin-bottom: 5px;
        }
        
        .team-header.aj h3 { color: #4169E1; }
        .team-header.tesla h3 { color: #FFD700; }
        
        .team-header p {
            color: #8b949e;
        }
        
        .team-lead {
            padding: 20px;
            background: #0d1117;
            border-bottom: 1px solid #30363d;
            display: flex;
            align-items: center;
            gap: 15px;
        }
        
        .lead-avatar {
            width: 60px;
            height: 60px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5em;
            font-weight: bold;
            color: white;
        }
        
        .lead-avatar.aj { background: linear-gradient(135deg, #4169E1, #1E90FF); }
        .lead-avatar.tesla { background: linear-gradient(135deg, #FFD700, #FFA500); }
        
        .lead-info h4 {
            font-size: 1.2em;
            color: #f0f6fc;
        }
        
        .lead-info .role {
            color: #8b949e;
            font-size: 0.9em;
        }
        
        .executives {
            padding: 20px;
        }
        
        .executives h4 {
            color: #8b949e;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 15px;
        }
        
        .executive-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        
        .executive-card {
            background: #21262d;
            border: 1px solid #30363d;
            border-radius: 12px;
            padding: 15px;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .executive-card:hover {
            transform: translateY(-3px);
            border-color: #58a6ff;
            box-shadow: 0 5px 20px rgba(88, 166, 255, 0.2);
        }
        
        .exec-header {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 10px;
        }
        
        .exec-avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            color: white;
            font-size: 0.9em;
        }
        
        .exec-name {
            font-weight: 600;
            color: #f0f6fc;
        }
        
        .exec-spec {
            color: #8b949e;
            font-size: 0.85em;
            line-height: 1.4;
        }
        
        .task-section {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 16px;
            padding: 20px;
            margin-top: 30px;
        }
        
        .task-section h3 {
            color: #f0f6fc;
            margin-bottom: 20px;
        }
        
        .task-form {
            display: grid;
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .task-form input, .task-form textarea, .task-form select {
            background: #0d1117;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 12px;
            color: #c9d1d9;
            font-size: 1em;
        }
        
        .task-form textarea {
            min-height: 100px;
            resize: vertical;
        }
        
        .task-form input:focus, .task-form textarea:focus {
            outline: none;
            border-color: #58a6ff;
        }
        
        .btn {
            background: linear-gradient(135deg, #238636 0%, #2ea043 100%);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            font-size: 1em;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(46, 160, 67, 0.4);
        }
        
        .btn-secondary {
            background: #21262d;
            border: 1px solid #30363d;
        }
        
        .btn-secondary:hover {
            background: #30363d;
            box-shadow: none;
        }
        
        .routing-result {
            background: #0d1117;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 15px;
            margin-top: 15px;
            display: none;
        }
        
        .routing-result.show {
            display: block;
        }
        
        .routing-result .team {
            color: #58a6ff;
            font-size: 1.2em;
            font-weight: bold;
        }
        
        .routing-result .agent {
            color: #f778ba;
            margin-top: 5px;
        }
        
        .tasks-list {
            margin-top: 20px;
        }
        
        .task-item {
            background: #21262d;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .task-item .title {
            font-weight: 600;
            color: #f0f6fc;
        }
        
        .task-item .meta {
            color: #8b949e;
            font-size: 0.9em;
        }
        
        .task-status {
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: 600;
        }
        
        .task-status.pending { background: #b08800; color: #000; }
        .task-status.assigned { background: #1f6feb; color: #fff; }
        .task-status.in_progress { background: #8957e5; color: #fff; }
        .task-status.complete { background: #238636; color: #fff; }
        .task-status.failed { background: #da3633; color: #fff; }
        
        /* Agent Detail Modal */
        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.8);
            z-index: 1000;
            justify-content: center;
            align-items: center;
        }
        
        .modal.show {
            display: flex;
        }
        
        .modal-content {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 16px;
            max-width: 600px;
            width: 90%;
            max-height: 80vh;
            overflow-y: auto;
        }
        
        .modal-header {
            padding: 20px;
            border-bottom: 1px solid #30363d;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .modal-header h3 {
            color: #f0f6fc;
        }
        
        .modal-close {
            background: none;
            border: none;
            color: #8b949e;
            font-size: 1.5em;
            cursor: pointer;
        }
        
        .modal-body {
            padding: 20px;
        }
        
        .agent-detail-section {
            margin-bottom: 20px;
        }
        
        .agent-detail-section h4 {
            color: #8b949e;
            font-size: 0.9em;
            text-transform: uppercase;
            margin-bottom: 10px;
        }
        
        .capabilities-list {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }
        
        .capability-tag {
            background: #21262d;
            border: 1px solid #30363d;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.85em;
            color: #58a6ff;
        }
        
        .personality-bars {
            display: grid;
            gap: 10px;
        }
        
        .personality-bar {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .personality-bar label {
            width: 100px;
            color: #8b949e;
            font-size: 0.9em;
        }
        
        .bar-container {
            flex: 1;
            height: 8px;
            background: #21262d;
            border-radius: 4px;
            overflow: hidden;
        }
        
        .bar-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.5s;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🤖 Agent Teams</h1>
            <p>Coordinate AI agents across specialized teams</p>
        </header>
        
        <nav>
            <a href="/">Home</a>
            <a href="/deliberation">Deliberation</a>
            <a href="/teams">Teams</a>
        </nav>
        
        <!-- MiniOrca - Primary Agent -->
        <div class="primary-agent" id="primary-agent">
            <h2>🐋 MiniOrca</h2>
            <div class="role">Primary RL Agent</div>
            <div class="description">
                Action execution, tool usage, native control (mouse, keyboard, display).
                MiniOrca coordinates between teams and executes tasks in the real world.
            </div>
        </div>
        
        <!-- Teams Grid -->
        <div class="teams-grid" id="teams-grid">
            <!-- Teams will be loaded here -->
        </div>
        
        <!-- Task Assignment Section -->
        <div class="task-section">
            <h3>📋 Assign Task</h3>
            <div class="task-form">
                <input type="text" id="task-title" placeholder="Task title...">
                <textarea id="task-description" placeholder="Describe what you need done..."></textarea>
                <div style="display: flex; gap: 10px;">
                    <button class="btn btn-secondary" onclick="routeTask()">🔀 Auto-Route</button>
                    <button class="btn" onclick="createTask()">➕ Create Task</button>
                </div>
            </div>
            <div class="routing-result" id="routing-result">
                <div class="team">Team: <span id="route-team">-</span></div>
                <div class="agent">Assigned to: <span id="route-agent">-</span></div>
            </div>
            
            <div class="tasks-list" id="tasks-list">
                <!-- Tasks will be loaded here -->
            </div>
        </div>
    </div>
    
    <!-- Agent Detail Modal -->
    <div class="modal" id="agent-modal">
        <div class="modal-content">
            <div class="modal-header">
                <h3 id="modal-agent-name">Agent Details</h3>
                <button class="modal-close" onclick="closeModal()">&times;</button>
            </div>
            <div class="modal-body" id="modal-body">
                <!-- Agent details will be loaded here -->
            </div>
        </div>
    </div>
    
    <script>
        // Agent color map
        const agentColors = {
            'Axiom': '#9400D3',
            'Cipher': '#2F4F4F',
            'Vector': '#228B22',
            'Nexus': '#FF8C00',
            'Echo': '#FF69B4',
            'Flux': '#00CED1',
            'Prism': '#FF1493',
            'Helix': '#4B0082',
            'Volt': '#FF4500',
            'Spark': '#FF69B4',
        };
        
        // Load teams data
        async function loadTeams() {
            try {
                const response = await fetch('/api/core/teams');
                const data = await response.json();
                renderTeams(data.teams);
            } catch (error) {
                console.error('Error loading teams:', error);
            }
        }
        
        // Render teams
        function renderTeams(teams) {
            const grid = document.getElementById('teams-grid');
            grid.innerHTML = teams.map(team => `
                <div class="team-card">
                    <div class="team-header ${team.name.toLowerCase()}">
                        <h3>${team.name === 'AJ' ? '💼' : '🔬'} ${team.name}'s Team</h3>
                        <p>${team.description}</p>
                    </div>
                    <div class="team-lead">
                        <div class="lead-avatar ${team.name.toLowerCase()}">${team.lead[0]}</div>
                        <div class="lead-info">
                            <h4>${team.lead}</h4>
                            <div class="role">Team Lead</div>
                        </div>
                    </div>
                    <div class="executives">
                        <h4>Executives</h4>
                        <div class="executive-grid">
                            ${team.members.map(member => `
                                <div class="executive-card" onclick="showAgentDetail('${member}')">
                                    <div class="exec-header">
                                        <div class="exec-avatar" style="background: ${agentColors[member] || '#58a6ff'}">${member[0]}</div>
                                        <div class="exec-name">${member}</div>
                                    </div>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                </div>
            `).join('');
        }
        
        // Load agents data for detail view
        async function showAgentDetail(agentName) {
            try {
                const response = await fetch(`/api/core/agents/${agentName}`);
                const agent = await response.json();
                renderAgentDetail(agent);
                document.getElementById('agent-modal').classList.add('show');
            } catch (error) {
                console.error('Error loading agent:', error);
            }
        }
        
        // Render agent detail
        function renderAgentDetail(agent) {
            document.getElementById('modal-agent-name').textContent = agent.name;
            
            const body = document.getElementById('modal-body');
            body.innerHTML = `
                <div class="agent-detail-section">
                    <h4>Role & Specialization</h4>
                    <p style="color: #f0f6fc">${agent.role} | ${agent.team || 'Independent'}</p>
                    <p style="color: #8b949e; margin-top: 5px">${agent.specialization}</p>
                </div>
                
                <div class="agent-detail-section">
                    <h4>Description</h4>
                    <p style="color: #c9d1d9">${agent.description}</p>
                </div>
                
                <div class="agent-detail-section">
                    <h4>Capabilities</h4>
                    <div class="capabilities-list">
                        ${agent.capabilities.map(cap => `
                            <span class="capability-tag">${cap}</span>
                        `).join('')}
                    </div>
                </div>
                
                <div class="agent-detail-section">
                    <h4>Personality</h4>
                    <div class="personality-bars">
                        ${renderPersonalityBar('Warmth', agent.personality.warmth, '#FF69B4')}
                        ${renderPersonalityBar('Creativity', agent.personality.creativity, '#FFD700')}
                    </div>
                </div>
                
                <div class="agent-detail-section">
                    <h4>Current Workload</h4>
                    <p style="color: #c9d1d9">
                        Active: ${agent.workload?.active_tasks || 0} | 
                        Completed: ${agent.workload?.completed_tasks || 0}
                    </p>
                </div>
            `;
        }
        
        function renderPersonalityBar(label, value, color) {
            return `
                <div class="personality-bar">
                    <label>${label}</label>
                    <div class="bar-container">
                        <div class="bar-fill" style="width: ${value * 100}%; background: ${color}"></div>
                    </div>
                    <span style="color: #8b949e; font-size: 0.9em">${(value * 100).toFixed(0)}%</span>
                </div>
            `;
        }
        
        function closeModal() {
            document.getElementById('agent-modal').classList.remove('show');
        }
        
        // Route task
        async function routeTask() {
            const description = document.getElementById('task-description').value;
            if (!description) {
                alert('Please enter a task description');
                return;
            }
            
            try {
                const response = await fetch('/api/core/route?description=' + encodeURIComponent(description), {
                    method: 'POST'
                });
                const data = await response.json();
                
                document.getElementById('route-team').textContent = data.team || 'No specific team';
                document.getElementById('route-agent').textContent = data.agent || 'Team lead';
                document.getElementById('routing-result').classList.add('show');
            } catch (error) {
                console.error('Error routing task:', error);
            }
        }
        
        // Create task
        async function createTask() {
            const title = document.getElementById('task-title').value;
            const description = document.getElementById('task-description').value;
            
            if (!title || !description) {
                alert('Please enter both title and description');
                return;
            }
            
            try {
                const response = await fetch(`/api/core/tasks?title=${encodeURIComponent(title)}&description=${encodeURIComponent(description)}`, {
                    method: 'POST'
                });
                const task = await response.json();
                
                // Clear form
                document.getElementById('task-title').value = '';
                document.getElementById('task-description').value = '';
                document.getElementById('routing-result').classList.remove('show');
                
                // Reload tasks
                loadTasks();
                
                alert(`Task created! Assigned to: ${task.assigned_to || task.team || 'Pending'}`);
            } catch (error) {
                console.error('Error creating task:', error);
            }
        }
        
        // Load tasks
        async function loadTasks() {
            try {
                const response = await fetch('/api/core/tasks');
                const data = await response.json();
                renderTasks(data.tasks);
            } catch (error) {
                console.error('Error loading tasks:', error);
            }
        }
        
        // Render tasks
        function renderTasks(tasks) {
            const list = document.getElementById('tasks-list');
            
            if (!tasks || tasks.length === 0) {
                list.innerHTML = '<p style="color: #8b949e; text-align: center; padding: 20px;">No tasks yet. Create one above!</p>';
                return;
            }
            
            list.innerHTML = tasks.map(task => `
                <div class="task-item">
                    <div>
                        <div class="title">${task.title}</div>
                        <div class="meta">${task.team || 'No team'} → ${task.assigned_to || 'Unassigned'}</div>
                    </div>
                    <span class="task-status ${task.status}">${task.status}</span>
                </div>
            `).join('');
        }
        
        // Close modal on outside click
        document.getElementById('agent-modal').addEventListener('click', (e) => {
            if (e.target.id === 'agent-modal') {
                closeModal();
            }
        });
        
        // Initial load
        loadTeams();
        loadTasks();
    </script>
</body>
</html>'''


# =============================================================================
# Deliberation View - Shows the two-round deliberation process
# =============================================================================

@app.get("/deliberation")
async def deliberation_view():
    """Serve the deliberation-enhanced UI."""
    return HTMLResponse(content=get_deliberation_html())


def get_deliberation_html() -> str:
    """Generate HTML for deliberation view."""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mini - Deliberation View</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #21262d 100%);
            min-height: 100vh;
            color: #c9d1d9;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            text-align: center;
            padding: 20px 0;
            margin-bottom: 30px;
        }
        
        header h1 {
            font-size: 2rem;
            background: linear-gradient(90deg, #58a6ff, #a371f7, #f778ba);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        header p {
            color: #8b949e;
            margin-top: 8px;
        }
        
        /* Main Chat Area */
        .chat-section {
            background: rgba(22, 27, 34, 0.8);
            border-radius: 16px;
            border: 1px solid #30363d;
            padding: 20px;
            margin-bottom: 20px;
        }
        
        .chat-messages {
            min-height: 200px;
            max-height: 300px;
            overflow-y: auto;
            padding: 10px;
            margin-bottom: 15px;
        }
        
        .message {
            margin-bottom: 15px;
            animation: fadeIn 0.3s ease;
        }
        
        .message.user {
            text-align: right;
        }
        
        .message.user .bubble {
            background: #238636;
            display: inline-block;
            padding: 10px 15px;
            border-radius: 18px 18px 4px 18px;
        }
        
        .message.mini .bubble {
            background: #1f6feb;
            display: inline-block;
            padding: 10px 15px;
            border-radius: 18px 18px 18px 4px;
        }
        
        .chat-input {
            display: flex;
            gap: 10px;
        }
        
        .chat-input input {
            flex: 1;
            padding: 15px 20px;
            border-radius: 25px;
            border: 1px solid #30363d;
            background: #0d1117;
            color: #c9d1d9;
            font-size: 1rem;
        }
        
        .chat-input input:focus {
            outline: none;
            border-color: #58a6ff;
        }
        
        .chat-input button {
            padding: 15px 25px;
            border-radius: 25px;
            border: none;
            background: linear-gradient(90deg, #238636, #2ea043);
            color: white;
            font-weight: bold;
            cursor: pointer;
        }
        
        /* Deliberation Panel */
        .deliberation-section {
            background: rgba(22, 27, 34, 0.8);
            border-radius: 16px;
            border: 1px solid #30363d;
            padding: 20px;
        }
        
        .deliberation-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 1px solid #30363d;
        }
        
        .deliberation-header h2 {
            font-size: 1.2rem;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .phase-indicator {
            display: flex;
            gap: 10px;
        }
        
        .phase {
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            background: #21262d;
            color: #8b949e;
        }
        
        .phase.active {
            background: #1f6feb;
            color: white;
        }
        
        .phase.complete {
            background: #238636;
            color: white;
        }
        
        /* Agent Cards */
        .agents-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .agent-card {
            background: #0d1117;
            border-radius: 12px;
            border: 1px solid #30363d;
            padding: 15px;
        }
        
        .agent-card.aj {
            border-color: #58a6ff;
        }
        
        .agent-card.tesla {
            border-color: #a371f7;
        }
        
        .agent-card-header {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 12px;
        }
        
        .agent-avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            background: #21262d;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 0.9em;
            color: white;
        }
        
        .agent-card.aj .agent-avatar { background: rgba(88, 166, 255, 0.2); }
        .agent-card.tesla .agent-avatar { background: rgba(163, 113, 247, 0.2); }
        
        .agent-name {
            font-weight: bold;
        }
        
        .agent-role {
            font-size: 0.8rem;
            color: #8b949e;
        }
        
        .round-content {
            background: #161b22;
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 10px;
        }
        
        .round-label {
            font-size: 0.75rem;
            color: #8b949e;
            margin-bottom: 8px;
        }
        
        .round-text {
            font-size: 0.9rem;
            line-height: 1.5;
        }
        
        .prediction {
            font-style: italic;
            color: #8b949e;
            font-size: 0.85rem;
            margin-top: 8px;
            padding-top: 8px;
            border-top: 1px dashed #303d3d;
        }
        
        /* Synthesis Section */
        .synthesis {
            background: #0d1117;
            border-radius: 12px;
            border: 2px solid #238636;
            padding: 20px;
        }
        
        .synthesis-header {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 12px;
        }
        
        .synthesis-header h3 {
            color: #3fb950;
        }
        
        .synthesis-content {
            line-height: 1.6;
        }
        
        /* Intent Badge */
        .intent-badge {
            display: inline-flex;
            align-items: center;
            gap: 5px;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: bold;
        }
        
        .intent-badge.query { background: #1f6feb; color: white; }
        .intent-badge.action { background: #238636; color: white; }
        .intent-badge.build { background: #a371f7; color: white; }
        .intent-badge.clarify { background: #f0883e; color: white; }
        
        /* Approval Section */
        .approval-section {
            background: #0d1117;
            border-radius: 12px;
            border: 2px solid #f0883e;
            padding: 20px;
            margin-top: 20px;
        }
        
        .approval-actions {
            display: flex;
            gap: 10px;
            margin-top: 15px;
        }
        
        .btn-approve {
            padding: 10px 20px;
            border-radius: 8px;
            border: none;
            background: #238636;
            color: white;
            cursor: pointer;
        }
        
        .btn-reject {
            padding: 10px 20px;
            border-radius: 8px;
            border: 1px solid #f85149;
            background: transparent;
            color: #f85149;
            cursor: pointer;
        }
        
        .collapsible {
            cursor: pointer;
        }
        
        .collapsible::after {
            content: ' ▼';
            font-size: 0.8em;
        }
        
        .collapsible.collapsed::after {
            content: ' ▶';
        }
        
        .collapse-content {
            display: block;
        }
        
        .collapse-content.hidden {
            display: none;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .loading {
            display: flex;
            gap: 5px;
            padding: 10px;
        }
        
        .loading span {
            width: 8px;
            height: 8px;
            background: #58a6ff;
            border-radius: 50%;
            animation: bounce 1.4s infinite ease-in-out both;
        }
        
        .loading span:nth-child(1) { animation-delay: -0.32s; }
        .loading span:nth-child(2) { animation-delay: -0.16s; }
        
        @keyframes bounce {
            0%, 80%, 100% { transform: scale(0); }
            40% { transform: scale(1); }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🧠 Mini - Deliberation Mode</h1>
            <p>Two-round deliberation between AJ and Tesla, synthesized by Mini</p>
        </header>
        
        <!-- Main Chat -->
        <div class="chat-section">
            <div class="chat-messages" id="chat-messages">
                <div class="message mini">
                    <div class="bubble">Hello! I use two-round deliberation to answer your questions. Ask me anything!</div>
                </div>
            </div>
            
            <div class="chat-input">
                <input type="text" id="user-input" placeholder="Ask a question or give a command..." onkeypress="if(event.key==='Enter')sendMessage()">
                <button onclick="sendMessage()">Send</button>
            </div>
        </div>
        
        <!-- Deliberation View -->
        <div class="deliberation-section" id="deliberation-section" style="display:none;">
            <div class="deliberation-header">
                <h2 class="collapsible" onclick="toggleDeliberation()">
                    🔄 Deliberation Process
                    <span class="intent-badge query" id="intent-badge">QUERY</span>
                </h2>
                <div class="phase-indicator">
                    <span class="phase" id="phase-round1">Round 1</span>
                    <span class="phase" id="phase-round2">Round 2</span>
                    <span class="phase" id="phase-synthesis">Synthesis</span>
                </div>
            </div>
            
            <div class="collapse-content" id="deliberation-content">
                <div class="agents-row">
                    <!-- AJ Card -->
                    <div class="agent-card aj">
                        <div class="agent-card-header">
                            <div class="agent-avatar">💼</div>
                            <div>
                                <div class="agent-name">AJ</div>
                                <div class="agent-role">Systems & Security</div>
                            </div>
                        </div>
                        
                        <div class="round-content" id="aj-round1">
                            <div class="round-label">Round 1</div>
                            <div class="round-text">Waiting...</div>
                        </div>
                        
                        <div class="round-content" id="aj-round2">
                            <div class="round-label">Round 2 (Refined)</div>
                            <div class="round-text">Waiting...</div>
                        </div>
                    </div>
                    
                    <!-- Tesla Card -->
                    <div class="agent-card tesla">
                        <div class="agent-card-header">
                            <div class="agent-avatar">⚡</div>
                            <div>
                                <div class="agent-name">Tesla</div>
                                <div class="agent-role">Philosophy & Purpose</div>
                            </div>
                        </div>
                        
                        <div class="round-content" id="tesla-round1">
                            <div class="round-label">Round 1</div>
                            <div class="round-text">Waiting...</div>
                        </div>
                        
                        <div class="round-content" id="tesla-round2">
                            <div class="round-label">Round 2 (Refined)</div>
                            <div class="round-text">Waiting...</div>
                        </div>
                    </div>
                </div>
                
                <!-- Synthesis -->
                <div class="synthesis" id="synthesis">
                    <div class="synthesis-header">
                        <span>✨</span>
                        <h3>Mini's Synthesis</h3>
                    </div>
                    <div class="synthesis-content" id="synthesis-content">
                        Waiting for deliberation to complete...
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Approval Section (shown when needed) -->
        <div class="approval-section" id="approval-section" style="display:none;">
            <h3>⚠️ Approval Required</h3>
            <p id="approval-message">This action requires your approval before proceeding.</p>
            <div id="approval-assessment"></div>
            <div class="approval-actions">
                <button class="btn-approve" onclick="approve(true)">✓ Approve</button>
                <button class="btn-reject" onclick="approve(false)">✕ Reject</button>
            </div>
        </div>
    </div>
    
    <script>
        let ws = null;
        let currentApprovalId = null;
        
        // Connect WebSocket
        function connectWS() {
            ws = new WebSocket(`ws://${window.location.host}/ws/core`);
            
            ws.onmessage = (event) => {
                const msg = JSON.parse(event.data);
                handleWSMessage(msg);
            };
            
            ws.onclose = () => {
                setTimeout(connectWS, 3000);
            };
        }
        
        function handleWSMessage(msg) {
            if (msg.type === 'core_update') {
                const data = msg.data;
                
                if (data.type === 'state_change') {
                    updatePhaseIndicator(data.state);
                }
                
                if (data.type === 'approval_update') {
                    showApproval(data.request);
                }
            }
            
            if (msg.type === 'response') {
                displayResponse(msg.data);
            }
        }
        
        async function sendMessage() {
            const input = document.getElementById('user-input');
            const message = input.value.trim();
            if (!message) return;
            
            // Add user message
            addMessage('user', message);
            input.value = '';
            
            // Show deliberation section
            document.getElementById('deliberation-section').style.display = 'block';
            resetDeliberation();
            
            // Show loading
            addMessage('mini', '<div class="loading"><span></span><span></span><span></span></div>');
            
            try {
                const resp = await fetch('/api/core/process', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message })
                });
                
                const data = await resp.json();
                displayResponse(data);
                
            } catch (e) {
                removeLoading();
                addMessage('mini', 'Error processing request: ' + e.message);
            }
        }
        
        function displayResponse(data) {
            removeLoading();
            
            // Update intent badge
            if (data.intent) {
                const badge = document.getElementById('intent-badge');
                badge.textContent = data.intent.intent_type.toUpperCase();
                badge.className = 'intent-badge ' + data.intent.intent_type;
            }
            
            // Update deliberation if present
            if (data.deliberation) {
                const d = data.deliberation;
                
                if (d.aj_round1) {
                    updateAgentRound('aj', 1, d.aj_round1);
                }
                if (d.tesla_round1) {
                    updateAgentRound('tesla', 1, d.tesla_round1);
                }
                if (d.aj_round2) {
                    updateAgentRound('aj', 2, d.aj_round2);
                }
                if (d.tesla_round2) {
                    updateAgentRound('tesla', 2, d.tesla_round2);
                }
                if (d.synthesis) {
                    document.getElementById('synthesis-content').textContent = d.synthesis;
                }
                
                updatePhaseIndicator('complete');
            }
            
            // Add final message
            addMessage('mini', data.message);
            
            // Check for approval
            if (data.state === 'awaiting_approval') {
                // Will be handled by webhook
            }
        }
        
        async function sendMessage() {
            const input = document.getElementById('user-input');
            const message = input.value.trim();
            if (!message) return;
            
            // Add user message
            addMessage('user', message);
            input.value = '';
            
            // Show deliberation section
            document.getElementById('deliberation-section').style.display = 'block';
            resetDeliberation();
            
            // Show loading
            addMessage('mini', '<div class="loading"><span></span><span></span><span></span></div>');
            
            try {
                const resp = await fetch('/api/core/process', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message })
                });
                
                const data = await resp.json();
                displayResponse(data);
                
            } catch (e) {
                removeLoading();
                addMessage('mini', 'Error processing request: ' + e.message);
            }
        }
        
        function displayResponse(data) {
            removeLoading();
            
            // Update intent badge
            if (data.intent) {
                const badge = document.getElementById('intent-badge');
                badge.textContent = data.intent.intent_type.toUpperCase();
                badge.className = 'intent-badge ' + data.intent.intent_type;
            }
            
            // Update deliberation if present
            if (data.deliberation) {
                const d = data.deliberation;
                
                if (d.aj_round1) {
                    updateAgentRound('aj', 1, d.aj_round1);
                }
                if (d.tesla_round1) {
                    updateAgentRound('tesla', 1, d.tesla_round1);
                }
                if (d.aj_round2) {
                    updateAgentRound('aj', 2, d.aj_round2);
                }
                if (d.tesla_round2) {
                    updateAgentRound('tesla', 2, d.tesla_round2);
                }
                if (d.synthesis) {
                    document.getElementById('synthesis-content').textContent = d.synthesis;
                }
                
                updatePhaseIndicator('complete');
            }
            
            // Add final message
            addMessage('mini', data.message);
            
            // Check for approval
            if (data.state === 'awaiting_approval') {
                // Will be handled by webhook
            }
        }
        
        async function sendMessage() {
            const input = document.getElementById('user-input');
            const message = input.value.trim();
            if (!message) return;
            
            // Add user message
            addMessage('user', message);
            input.value = '';
            
            // Show deliberation section
            document.getElementById('deliberation-section').style.display = 'block';
            resetDeliberation();
            
            // Show loading
            addMessage('mini', '<div class="loading"><span></span><span></span><span></span></div>');
            
            try {
                const resp = await fetch('/api/core/process', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message })
                });
                
                const data = await resp.json();
                displayResponse(data);
                
            } catch (e) {
                removeLoading();
                addMessage('mini', 'Error processing request: ' + e.message);
            }
        }
        
        function displayResponse(data) {
            removeLoading();
            
            // Update intent badge
            if (data.intent) {
                const badge = document.getElementById('intent-badge');
                badge.textContent = data.intent.intent_type.toUpperCase();
                badge.className = 'intent-badge ' + data.intent.intent_type;
            }
            
            // Update deliberation if present
            if (data.deliberation) {
                const d = data.deliberation;
                
                if (d.aj_round1) {
                    updateAgentRound('aj', 1, d.aj_round1);
                }
                if (d.tesla_round1) {
                    updateAgentRound('tesla', 1, d.tesla_round1);
                }
                if (d.aj_round2) {
                    updateAgentRound('aj', 2, d.aj_round2);
                }
                if (d.tesla_round2) {
                    updateAgentRound('tesla', 2, d.tesla_round2);
                }
                if (d.synthesis) {
                    document.getElementById('synthesis-content').textContent = d.synthesis;
                }
                
                updatePhaseIndicator('complete');
            }
            
            // Add final message
            addMessage('mini', data.message);
            
            // Check for approval
            if (data.state === 'awaiting_approval') {
                // Will be handled by webhook
            }
        }
        
        async function sendMessage() {
            const input = document.getElementById('user-input');
            const message = input.value.trim();
            if (!message) return;
            
            // Add user message
            addMessage('user', message);
            input.value = '';
            
            // Show deliberation section
            document.getElementById('deliberation-section').style.display = 'block';
            resetDeliberation();
            
            // Show loading
            addMessage('mini', '<div class="loading"><span></span><span></span><span></span></div>');
            
            try {
                const resp = await fetch('/api/core/process', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message })
                });
                
                const data = await resp.json();
                displayResponse(data);
                
            } catch (e) {
                removeLoading();
                addMessage('mini', 'Error processing request: ' + e.message);
            }
        }
        
        function displayResponse(data) {
            removeLoading();
            
            // Update intent badge
            if (data.intent) {
                const badge = document.getElementById('intent-badge');
                badge.textContent = data.intent.intent_type.toUpperCase();
                badge.className = 'intent-badge ' + data.intent.intent_type;
            }
            
            // Update deliberation if present
            if (data.deliberation) {
                const d = data.deliberation;
                
                if (d.aj_round1) {
                    updateAgentRound('aj', 1, d.aj_round1);
                }
                if (d.tesla_round1) {
                    updateAgentRound('tesla', 1, d.tesla_round1);
                }
                if (d.aj_round2) {
                    updateAgentRound('aj', 2, d.aj_round2);
                }
                if (d.tesla_round2) {
                    updateAgentRound('tesla', 2, d.tesla_round2);
                }
                if (d.synthesis) {
                    document.getElementById('synthesis-content').textContent = d.synthesis;
                }
                
                updatePhaseIndicator('complete');
            }
            
            // Add final message
            addMessage('mini', data.message);
            
            // Check for approval
            if (data.state === 'awaiting_approval') {
                // Will be handled by webhook
            }
        }
        
        async function sendMessage() {
            const input = document.getElementById('user-input');
            const message = input.value.trim();
            if (!message) return;
            
            // Add user message
            addMessage('user', message);
            input.value = '';
            
            // Show deliberation section
            document.getElementById('deliberation-section').style.display = 'block';
            resetDeliberation();
            
            // Show loading
            addMessage('mini', '<div class="loading"><span></span><span></span><span></span></div>');
            
            try {
                const resp = await fetch('/api/core/process', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message })
                });
                
                const data = await resp.json();
                displayResponse(data);
                
            } catch (e) {
                removeLoading();
                addMessage('mini', 'Error processing request: ' + e.message);
            }
        }
        
        function displayResponse(data) {
            removeLoading();
            
            // Update intent badge
            if (data.intent) {
                const badge = document.getElementById('intent-badge');
                badge.textContent = data.intent.intent_type.toUpperCase();
                badge.className = 'intent-badge ' + data.intent.intent_type;
            }
            
            // Update deliberation if present
            if (data.deliberation) {
                const d = data.deliberation;
                
                if (d.aj_round1) {
                    updateAgentRound('aj', 1, d.aj_round1);
                }
                if (d.tesla_round1) {
                    updateAgentRound('tesla', 1, d.tesla_round1);
                }
                if (d.aj_round2) {
                    updateAgentRound('aj', 2, d.aj_round2);
                }
                if (d.tesla_round2) {
                    updateAgentRound('tesla', 2, d.tesla_round2);
                }
                if (d.synthesis) {
                    document.getElementById('synthesis-content').textContent = d.synthesis;
                }
                
                updatePhaseIndicator('complete');
            }
            
            // Add final message
            addMessage('mini', data.message);
            
            // Check for approval
            if (data.state === 'awaiting_approval') {
                // Will be handled by webhook
            }
        }
        
        async function sendMessage() {
            const input = document.getElementById('user-input');
            const message = input.value.trim();
            if (!message) return;
            
            // Add user message
            addMessage('user', message);
            input.value = '';
            
            // Show deliberation section
            document.getElementById('deliberation-section').style.display = 'block';
            resetDeliberation();
            
            // Show loading
            addMessage('mini', '<div class="loading"><span></span><span></span><span></span></div>');
            
            try {
                const resp = await fetch('/api/core/process', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message })
                });
                
                const data = await resp.json();
                displayResponse(data);
                
            } catch (e) {
                removeLoading();
                addMessage('mini', 'Error processing request: ' + e.message);
            }
        }
        
        function displayResponse(data) {
            removeLoading();
            
            // Update intent badge
            if (data.intent) {
                const badge = document.getElementById('intent-badge');
                badge.textContent = data.intent.intent_type.toUpperCase();
                badge.className = 'intent-badge ' + data.intent.intent_type;
            }
            
            // Update deliberation if present
            if (data.deliberation) {
                const d = data.deliberation;
                
                if (d.aj_round1) {
                    updateAgentRound('aj', 1, d.aj_round1);
                }
                if (d.tesla_round1) {
                    updateAgentRound('tesla', 1, d.tesla_round1);
                }
                if (d.aj_round2) {
                    updateAgentRound('aj', 2, d.aj_round2);
                }
                if (d.tesla_round2) {
                    updateAgentRound('tesla', 2, d.tesla_round2);
                }
                if (d.synthesis) {
                    document.getElementById('synthesis-content').textContent = d.synthesis;
                }
                
                updatePhaseIndicator('complete');
            }
            
            // Add final message
            addMessage('mini', data.message);
            
            // Check for approval
            if (data.state === 'awaiting_approval') {
                // Will be handled by webhook
            }
        }
        
        function toggleDeliberation() {
            const content = document.getElementById('deliberation-content');
            const header = document.querySelector('.deliberation-header h2');
            content.classList.toggle('hidden');
            header.classList.toggle('collapsed');
        }
        
        function showApproval(request) {
            currentApprovalId = request.id;
            document.getElementById('approval-section').style.display = 'block';
            document.getElementById('approval-message').textContent = 
                `Action: ${request.intent}\\nRisk Level: ${request.risk_level}`;
            
            if (request.security_assessment) {
                document.getElementById('approval-assessment').innerHTML = 
                    `<p><strong>Security Assessment:</strong> ${request.security_assessment.reasoning}</p>`;
            }
        }
        
        async function approve(approved) {
            if (!currentApprovalId) return;
            
            await fetch(`/api/core/approvals/${currentApprovalId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ approved, notes: '' })
            });
            
            document.getElementById('approval-section').style.display = 'none';
            currentApprovalId = null;
            
            addMessage('mini', approved ? 'Action approved! Proceeding...' : 'Action rejected.');
        }
        
        // Initialize
        connectWS();
    </script>
</body>
</html>'''


# =============================================================================
# Main HTML UI
# =============================================================================

def get_main_html() -> str:
    """Generate the main HTML UI - Speech Balloon Design."""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Agent Control Panel</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            min-height: 100vh;
            color: #e4e4e4;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            text-align: center;
            padding: 20px 0;
            margin-bottom: 30px;
        }
        
        header h1 {
            font-size: 2.5rem;
            background: linear-gradient(90deg, #00CED1, #4A90D9, #9400D3);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        
        header p {
            color: #888;
        }
        
        .panels {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        
        @media (max-width: 1200px) {
            .panels {
                grid-template-columns: 1fr;
            }
        }
        
        .panel {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 20px;
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .panel-header {
            display: flex;
            align-items: center;
            gap: 15px;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .avatar-container {
            width: 80px;
            height: 80px;
            border-radius: 50%;
            overflow: hidden;
            background: rgba(255, 255, 255, 0.1);
            flex-shrink: 0;
        }
        
        .avatar-container img {
            width: 100%;
            height: 100%;
        }
        
        .agent-info h2 {
            font-size: 1.3rem;
            margin-bottom: 5px;
        }
        
        .agent-info .role {
            font-size: 0.8rem;
            color: #888;
        }
        
        .team-dropdown {
            width: 100%;
            padding: 10px;
            border-radius: 10px;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: #fff;
            font-size: 0.9rem;
            margin-bottom: 15px;
            cursor: pointer;
        }
        
        .team-dropdown option {
            background: #1a1a2e;
            color: #fff;
        }
        
        /* Speech Balloon Styles - Cartoon style! */
        .speech-area {
            min-height: 300px;
            max-height: 400px;
            overflow-y: auto;
            padding: 15px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 15px;
        }
        
        .speech-balloon {
            position: relative;
            background: linear-gradient(135deg, rgba(255,255,255,0.15), rgba(255,255,255,0.05));
            border-radius: 20px;
            padding: 15px 20px;
            margin-bottom: 15px;
            border: 1px solid rgba(255,255,255,0.1);
            animation: popIn 0.3s ease-out;
        }
        
        .speech-balloon::before {
            content: '';
            position: absolute;
            bottom: -10px;
            left: 30px;
            border-width: 10px 10px 0;
            border-style: solid;
            border-color: rgba(255,255,255,0.1) transparent transparent;
        }
        
        .speech-balloon.miniorca {
            background: linear-gradient(135deg, rgba(0, 206, 209, 0.2), rgba(32, 178, 170, 0.1));
            border-color: rgba(0, 206, 209, 0.3);
        }
        
        .speech-balloon.miniorca::before {
            border-color: rgba(0, 206, 209, 0.3) transparent transparent;
        }
        
        .speech-balloon.aj-team {
            background: linear-gradient(135deg, rgba(74, 144, 217, 0.2), rgba(74, 144, 217, 0.1));
            border-color: rgba(74, 144, 217, 0.3);
        }
        
        .speech-balloon.aj-team::before {
            border-color: rgba(74, 144, 217, 0.3) transparent transparent;
        }
        
        .speech-balloon.tesla-team {
            background: linear-gradient(135deg, rgba(148, 0, 211, 0.2), rgba(148, 0, 211, 0.1));
            border-color: rgba(148, 0, 211, 0.3);
        }
        
        .speech-balloon.tesla-team::before {
            border-color: rgba(148, 0, 211, 0.3) transparent transparent;
        }
        
        .speech-balloon .agent-name {
            font-weight: bold;
            font-size: 0.85rem;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .speech-balloon .agent-name .emotion {
            font-size: 0.7rem;
            padding: 2px 8px;
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
        }
        
        .speech-balloon .text {
            font-size: 1rem;
            line-height: 1.5;
        }
        
        .speech-balloon.thinking {
            opacity: 0.7;
        }
        
        .speech-balloon.thinking .text {
            font-style: italic;
            color: #aaa;
        }
        
        @keyframes popIn {
            0% { transform: scale(0.8); opacity: 0; }
            100% { transform: scale(1); opacity: 1; }
        }
        
        .emotion-indicator {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 10px;
            font-size: 0.75rem;
            background: rgba(255, 255, 255, 0.1);
            margin-left: 10px;
        }
        
        .primary-panel {
            border: 2px solid rgba(0, 206, 209, 0.3);
        }
        
        .primary-panel .panel-header {
            background: linear-gradient(90deg, rgba(0, 206, 209, 0.1), rgba(32, 178, 170, 0.1));
            margin: -20px -20px 20px -20px;
            padding: 20px;
            border-radius: 18px 18px 0 0;
        }
        
        /* Main Input Area - All input goes here */
        .main-input-area {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 20px;
            padding: 20px;
            border: 2px solid rgba(0, 206, 209, 0.3);
        }
        
        .input-header {
            display: flex;
            align-items: center;
            gap: 15px;
            margin-bottom: 15px;
        }
        
        .input-header h3 {
            font-size: 1.1rem;
        }
        
        .recipient-select {
            padding: 8px 15px;
            border-radius: 20px;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: #fff;
            font-size: 0.9rem;
            cursor: pointer;
        }
        
        .recipient-select option {
            background: #1a1a2e;
            color: #fff;
        }
        
        .input-row {
            display: flex;
            gap: 10px;
        }
        
        .input-row input {
            flex: 1;
            padding: 15px 20px;
            border-radius: 30px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            background: rgba(255, 255, 255, 0.1);
            color: #fff;
            font-size: 1rem;
        }
        
        .input-row input::placeholder {
            color: #888;
        }
        
        .input-row input:focus {
            outline: none;
            border-color: #00CED1;
            box-shadow: 0 0 20px rgba(0, 206, 209, 0.2);
        }
        
        .btn {
            padding: 15px 25px;
            border-radius: 30px;
            border: none;
            background: linear-gradient(90deg, #00CED1, #4A90D9);
            color: #fff;
            font-size: 1rem;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(0, 206, 209, 0.3);
        }
        
        .btn-voice {
            background: linear-gradient(90deg, #9400D3, #4A90D9);
            padding: 15px 18px;
        }
        
        .btn-voice.active {
            background: linear-gradient(90deg, #00CED1, #20B2AA);
            box-shadow: 0 0 15px rgba(0, 206, 209, 0.5);
        }
        
        .controls-row {
            display: flex;
            gap: 10px;
            margin-top: 15px;
            flex-wrap: wrap;
        }
        
        .control-btn {
            padding: 10px 15px;
            border-radius: 15px;
            border: 1px solid #30363d;
            background: rgba(255, 255, 255, 0.05);
            color: #8b949e;
            cursor: pointer;
            font-size: 0.85em;
            transition: all 0.2s;
        }
        
        .control-btn:hover {
            background: #21262d;
            color: #f0f6fc;
        }
        
        .status-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 0;
            margin-top: 10px;
            font-size: 0.8rem;
            color: #888;
        }
        
        .status-indicator {
            display: flex;
            align-items: center;
            gap: 5px;
        }
        
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #4CAF50;
        }
        
        .vision-preview {
            position: fixed;
            bottom: 20px;
            right: 20px;
            width: 320px;
            background: rgba(26, 26, 46, 0.95);
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            overflow: hidden;
            display: none;
            z-index: 1000;
        }
        
        .vision-preview.active {
            display: block;
        }
        
        .vision-preview-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 15px;
            background: rgba(0, 206, 209, 0.2);
        }
        
        .vision-preview-header h4 {
            margin: 0;
            font-size: 0.9rem;
        }
        
        .vision-preview img {
            width: 100%;
            height: auto;
        }
        
        /* Mobile Notes Banner */
        .mobile-note {
            background: linear-gradient(90deg, rgba(255, 193, 7, 0.2), rgba(255, 152, 0, 0.2));
            border: 1px solid rgba(255, 193, 7, 0.3);
            border-radius: 10px;
            padding: 10px 15px;
            margin-top: 20px;
            font-size: 0.85rem;
            color: #ffc107;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🤖 Agent Control Panel</h1>
            <p>Multi-Agent AI System • All agents respond at their own pace</p>
        </header>
        
        <!-- Quick Links Dropdown -->
        <div style="display:flex;gap:8px;align-items:center;margin-bottom:18px;">
            <div style="color:#bbb;font-size:0.95rem;">Quick Links:</div>
            <select id="quick-links" class="team-dropdown" onchange="openQuickLink()">
                <option value="/">Main UI (this page)</option>
                <option value="/static/playground/">Playground UI (static)</option>
                <option value="/playground/challenges">Playground API: /playground/challenges</option>
                <option value="/api/agents">API: /api/agents</option>
                <option value="/api/conversations">API: /api/conversations</option>
                <option value="/api/conversations/{conversation_id}">API: conversation details (needs id)</option>
                <option value="/api/conversations/{conversation_id}/messages?speak=true">API: send message (POST) (needs id)</option>
                <option value="ws://{host}/ws/{conversation_id}">WebSocket (example) (needs host & id)</option>
                <option value="/api/telemetry">API: /api/telemetry</option>
                <option value="/api/metrics_summary">API: /api/metrics_summary</option>
                <option value="/metrics">Prometheus: /metrics</option>
            </select>
            <input id="quick-conv" placeholder="conversation id (optional)" style="padding:8px;border-radius:8px;border:1px solid rgba(255,255,255,0.08);background:rgba(0,0,0,0.15);color:#fff;" />
            <input id="quick-host" placeholder="host (optional)" style="padding:8px;border-radius:8px;border:1px solid rgba(255,255,255,0.08);background:rgba(0,0,0,0.15);color:#fff;width:160px;" value="localhost:8001" />
            <button class="control-btn" onclick="openQuickLink(true)">Open</button>
        </div>
        
        <div class="panels">
            <!-- MiniOrca Panel (Primary) -->
            <div class="panel primary-panel">
                <div class="panel-header">
                    <div class="avatar-container" id="miniorca-avatar">
                        <img src="/api/agents/MiniOrca/avatar" alt="MiniOrca">
                    </div>
                    <div class="agent-info">
                        <h2>MiniOrca</h2>
                        <div class="role">Primary RL Agent • Native Control • 👁️ Vision</div>
                    </div>
                </div>
                
                <div class="speech-area" id="miniorca-speech">
                    <div class="speech-balloon miniorca">
                        <div class="agent-name">🐋 MiniOrca <span class="emotion">neutral</span></div>
                        <div class="text">Hello! I'm MiniOrca, your primary AI assistant. I can see, hear, and control your computer. Send me a message below!</div>
                    </div>
                </div>
                
                <div class="status-bar">
                    <div class="status-indicator">
                        <span class="status-dot"></span>
                        <span id="camera-status">Checking vision...</span>
                    </div>
                </div>
            </div>
            
            <!-- (AJ and Tesla panels removed to keep UI focused on MiniOrca) -->
        </div>

            <!-- Playground Panel -->
            <div class="panel">
                <div class="panel-header">
                    <div class="avatar-container" id="playground-avatar">
                        <img src="/static/img/shield.png" alt="Playground">
                    </div>
                    <div class="agent-info">
                        <h2>Playground • Lux</h2>
                        <div class="role">Daily CVE Challenges • Simulated</div>
                    </div>
                </div>
                <div class="speech-area" id="playground-area">
                    <div id="playground-root"></div>
                    <div style="margin-top:8px;">
                        <input id="playground-admin-key" type="password" placeholder="Admin key (for NVD refresh)" style="padding:6px;border-radius:6px;border:1px solid rgba(255,255,255,0.1);background:rgba(0,0,0,0.2);color:#fff;width:60%" />
                        <button class="control-btn" onclick="refreshNVD()">Refresh NVD</button>
                    </div>
                    <script type="module" src="/static/playground/index.js"></script>
                </div>
            </div>

        <!-- Main Input Area - Single point of entry -->
        <div class="main-input-area">
            <div class="input-header">
                <h3>💬 Send Message To:</h3>
                <select class="recipient-select" id="recipient-select">
                    <option value="all">All Agents (Parallel)</option>
                    <option value="MiniOrca">🐋 MiniOrca Only</option>
                    <optgroup label="AJ's Team">
                        <option value="AJ">💼 AJ</option>
                        <option value="Axiom">💻 Axiom</option>
                        <option value="Cipher">🔐 Cipher</option>
                        <option value="Vector">📊 Vector</option>
                        <option value="Nexus">🔌 Nexus</option>
                        <option value="Echo">💬 Echo</option>
                    </optgroup>
                    <optgroup label="Tesla's Team">
                        <option value="Tesla">⚡ Tesla</option>
                        <option value="Flux">🔧 Flux</option>
                        <option value="Prism">🎨 Prism</option>
                        <option value="Helix">🧬 Helix</option>
                        <option value="Volt">⚡ Volt</option>
                        <option value="Spark">✨ Spark</option>
                    </optgroup>
                </select>
            </div>
            <div style="margin:10px 0 8px 0;padding:10px;border-radius:10px;background:rgba(0,0,0,0.12);border:1px solid rgba(255,255,255,0.03);color:#ffd;">
                <strong>Intro Note:</strong> Please introduce yourself the first time you message an agent. Your introduction will be stored in memory and associated with the selected agent to help personalize future conversations.
            </div>
            
            <div class="input-row">
                <input type="text" id="main-input" placeholder="Introduce yourself (this will be stored) or type your message...">
                <button class="btn-voice" id="voice-btn" onclick="toggleVoiceInput()">🎤</button>
                <button class="btn" onclick="sendToAgents()">Send</button>
            </div>
            
            <div class="controls-row">
                <button class="control-btn" onclick="takeScreenshot()">📸 Screenshot</button>
                <button class="control-btn" onclick="captureCamera()">📷 Camera</button>
                <button class="control-btn" onclick="toggleAllVoices()">🔊 Toggle Voices</button>
                <button class="control-btn" onclick="clearAllChats()">🗑️ Clear</button>
            </div>
        </div>
        
        <div class="mobile-note">
            📱 <strong>Mobile App Notes:</strong> Phone version will require: Screen Access, Camera, Microphone, and App Permissions for full agent capabilities.
        </div>
        
        <!-- Vision Preview -->
        <div class="vision-preview" id="vision-preview">
            <div class="vision-preview-header">
                <h4>👁️ Vision Feed</h4>
                <button onclick="closeVisionPreview()" style="background:none;border:none;color:#fff;cursor:pointer;">✕</button>
            </div>
            <img id="vision-image" src="" alt="Vision Feed">
            <div style="padding:10px;font-size:0.8rem;color:#888;">
                <span id="vision-info">Camera feed provides visual context</span>
            </div>
        </div>
    </div>
    
    <script>
        // Agent icons for speech balloons
        const agentIcons = {
            'MiniOrca': '🐋',
            'AJ': '💼', 'Axiom': '💻', 'Cipher': '🔐', 'Vector': '📊', 'Nexus': '🔌', 'Echo': '💬',
            'Tesla': '⚡', 'Flux': '🔧', 'Prism': '🎨', 'Helix': '🧬', 'Volt': '⚡', 'Spark': '✨'
        };
        
        // Agent team mapping
        const agentTeams = {
            'MiniOrca': 'miniorca',
            'AJ': 'aj', 'Axiom': 'aj', 'Cipher': 'aj', 'Vector': 'aj', 'Nexus': 'aj', 'Echo': 'aj',
            'Tesla': 'tesla', 'Flux': 'tesla', 'Prism': 'tesla', 'Helix': 'tesla', 'Volt': 'tesla', 'Spark': 'tesla'
        };
        
        // State
        const state = {
            voicesEnabled: true,
            selectedTeamAgents: {
                aj: 'AJ',
                tesla: 'Tesla'
            }
        };
        
        // Audio players for each panel (can play simultaneously)
        const audioPlayers = {
            miniorca: new Audio(),
            aj: new Audio(),
            tesla: new Audio()
        };
        
        // Initialize
        document.addEventListener('DOMContentLoaded', () => {
            checkVisionStatus();
            
            document.getElementById('main-input').addEventListener('keypress', (e) => {
                if (e.key === 'Enter') sendToAgents();
            });
        });
        
        function selectTeamAgent(team) {
            const select = document.getElementById(`${team}-team-select`);
            const agentName = select.value;
            state.selectedTeamAgents[team] = agentName;
            
            // Update avatar
            const avatarImg = document.querySelector(`#${team}-team-avatar img`);
            avatarImg.src = `/api/agents/${agentName}/avatar?t=${Date.now()}`;
        }
        
        async function sendToAgents() {
            const input = document.getElementById('main-input');
            const content = input.value.trim();
            if (!content) return;
            
            const recipient = document.getElementById('recipient-select').value;
            input.value = '';
            
            // Determine which agents to send to
            let agents = [];
            if (recipient === 'all') {
                agents = ['MiniOrca', state.selectedTeamAgents.aj, state.selectedTeamAgents.tesla];
            } else {
                agents = [recipient];
            }
            
            // Show user message in MiniOrca panel (as she's the hub)
            addSpeechBalloon('miniorca-speech', 'You', content, 'user', '👤');
            
            // Send to all selected agents in PARALLEL - they respond at their own pace!
            agents.forEach(agentName => {
                const team = agentTeams[agentName];
                const speechAreaId = `${team}-speech`;
                
                // Show thinking balloon
                const thinkingId = `thinking-${agentName}-${Date.now()}`;
                addThinkingBalloon(speechAreaId, agentName, thinkingId);
                
                // Make async request - each agent responds independently
                fetchAgentResponse(agentName, content, speechAreaId, thinkingId);
            });
        }
        
        async function fetchAgentResponse(agentName, content, speechAreaId, thinkingId) {
            try {
                const resp = await fetch(`/api/chat/${agentName}`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ content: content })
                });
                
                const data = await resp.json();
                
                // Remove thinking balloon
                document.getElementById(thinkingId)?.remove();
                
                // Handle the nested response structure
                const primaryData = data.primary || data;
                const responseText = primaryData.response || "No response received";
                const emotion = primaryData.emotion || "neutral";
                
                // Add response balloon
                const team = agentTeams[agentName];
                const teamClass = team === 'miniorca' ? 'miniorca' : `${team}-team`;
                addSpeechBalloon(speechAreaId, agentName, responseText, teamClass, agentIcons[agentName], emotion);
                
                // Play voice if enabled
                if (state.voicesEnabled && primaryData.audio) {
                    const player = audioPlayers[team];
                    player.src = primaryData.audio;
                    player.play().catch(e => console.log('Audio play failed:', e));
                }
                
            } catch (e) {
                console.error(`Error from ${agentName}:`, e);
                document.getElementById(thinkingId)?.remove();
                addSpeechBalloon(speechAreaId, agentName, "Sorry, I encountered an error.", agentTeams[agentName] === 'miniorca' ? 'miniorca' : `${agentTeams[agentName]}-team`, agentIcons[agentName], 'confused');
            }
        }
        
        function addSpeechBalloon(areaId, agentName, text, teamClass, icon, emotion = 'neutral') {
            const area = document.getElementById(areaId);
            const balloon = document.createElement('div');
            balloon.className = `speech-balloon ${teamClass}`;
            balloon.innerHTML = `
                <div class="agent-name">${icon} ${agentName} <span class="emotion">${emotion}</span></div>
                <div class="text">${escapeHtml(text)}</div>
            `;
            area.appendChild(balloon);
            area.scrollTop = area.scrollHeight;
        }
        
        function addThinkingBalloon(areaId, agentName, id) {
            const area = document.getElementById(areaId);
            const team = agentTeams[agentName];
            const teamClass = team === 'miniorca' ? 'miniorca' : `${team}-team`;
            const balloon = document.createElement('div');
            balloon.id = id;
            balloon.className = `speech-balloon ${teamClass} thinking`;
            balloon.innerHTML = `
                <div class="agent-name">${agentIcons[agentName]} ${agentName} <span class="emotion">thinking</span></div>
                <div class="text">💭 Thinking...</div>
            `;
            area.appendChild(balloon);
            area.scrollTop = area.scrollHeight;
        }
        
        function toggleVoiceInput() {
            const btn = document.getElementById('voice-btn');
            btn.classList.toggle('active');
            // TODO: Implement speech-to-text
            alert('Voice input coming soon! For now, type your message.');
        }
        
        function toggleAllVoices() {
            state.voicesEnabled = !state.voicesEnabled;
            const status = state.voicesEnabled ? 'ON' : 'OFF';
            addSpeechBalloon('miniorca-speech', 'System', `Voice output ${status}`, 'miniorca', '🔊');
        }
        
        function clearAllChats() {
            ['miniorca-speech', 'aj-speech', 'tesla-speech'].forEach(id => {
                document.getElementById(id).innerHTML = '';
            });
        }
        
        async function checkVisionStatus() {
            try {
                const resp = await fetch('/api/vision/status');
                const data = await resp.json();
                
                const statusEl = document.getElementById('camera-status');
                if (data.camera_available) {
                    statusEl.innerHTML = '🟢 Vision Active';
                } else {
                    statusEl.innerHTML = '📺 Screen Only';
                }
            } catch (e) {
                console.error('Vision status check failed:', e);
            }
        }
        
        async function takeScreenshot() {
            try {
                const resp = await fetch('/api/native/screenshot', { method: 'POST' });
                const data = await resp.json();
                if (data.image) {
                    document.getElementById('vision-image').src = data.image;
                    document.getElementById('vision-info').textContent = 'Screenshot captured';
                    document.getElementById('vision-preview').classList.add('active');
                }
            } catch (e) {
                console.error('Screenshot failed:', e);
            }
        }
        
        async function captureCamera() {
            try {
                const resp = await fetch('/api/vision/camera');
                const data = await resp.json();
                
                if (data.available && data.image) {
                    document.getElementById('vision-image').src = data.image;
                    document.getElementById('vision-info').textContent = 'Camera capture';
                    document.getElementById('vision-preview').classList.add('active');
                    
                    addSpeechBalloon('miniorca-speech', 'MiniOrca', 
                        "I can see through the camera now! This gives me visual context.", 
                        'miniorca', '🐋', 'happy');
                } else {
                    addSpeechBalloon('miniorca-speech', 'MiniOrca',
                        "No camera detected. Install one to give me eyes for the physical world!",
                        'miniorca', '🐋', 'thinking');
                }
            } catch (e) {
                console.error('Camera capture failed:', e);
            }
        }
        
        function closeVisionPreview() {
            document.getElementById('vision-preview').classList.remove('active');
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        // Playground functions removed — handled by React bundle at /static/playground/index.js
        document.addEventListener('DOMContentLoaded', () => {
            // React bundle mounts into #playground-root
        });

        function openQuickLink(forceOpen=false) {
            const sel = document.getElementById('quick-links');
            if (!sel) return;
            let url = sel.value;
            const conv = (document.getElementById('quick-conv') || {}).value || '';
            const host = (document.getElementById('quick-host') || {}).value || window.location.host;

            // Replace placeholders
            if (url.includes('{conversation_id}')) {
                url = url.replace('{conversation_id}', conv || '');
            }
            if (url.includes('{host}')) {
                url = url.replace('{host}', host);
            }

            // If the option is a ws:// template, ensure host is used
            if (url.startsWith('ws://') || url.startsWith('wss://')) {
                // open in new window/tab
                window.open(url, '_blank');
                return;
            }

            // If absolute path, prepend origin when host looks like host:port
            if (url.startsWith('/')) {
                const full = `${window.location.protocol}//${host}${url}`;
                window.open(full, '_blank');
                return;
            }

            // fallback open raw value
            if (forceOpen) window.open(url, '_blank');
        }
    </script>
</body>
</html>'''


# =============================================================================
# Sandbox Observation View - Watch Agents Train with Tools
# =============================================================================

@app.get("/sandbox")
async def sandbox_view():
    """Serve the sandbox observation UI."""
    return HTMLResponse(content=get_sandbox_html())


def get_sandbox_html() -> str:
    """Generate HTML for sandbox observation view."""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sandbox Training Ground - Agent Observation</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #21262d 100%);
            min-height: 100vh;
            color: #c9d1d9;
        }
        
        .container {
            max-width: 1800px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            text-align: center;
            margin-bottom: 30px;
        }
        
        header h1 {
            font-size: 2.5em;
            background: linear-gradient(135deg, #00CED1, #48D1CC, #40E0D0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        
        header p {
            color: #8b949e;
            font-size: 1.1em;
        }
        
        nav {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-bottom: 30px;
        }
        
        nav a {
            color: #58a6ff;
            text-decoration: none;
            padding: 10px 20px;
            border: 1px solid #30363d;
            border-radius: 8px;
            transition: all 0.3s;
        }
        
        nav a:hover {
            background: #21262d;
            border-color: #58a6ff;
        }
        
        /* Stats Banner */
        .stats-banner {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 15px;
            margin-bottom: 30px;
        }
        
        .stat-card {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
        }
        
        .stat-card.active { border-color: #238636; }
        .stat-card.pending { border-color: #f0883e; }
        .stat-card.attention { border-color: #da3633; }
        
        .stat-value {
            font-size: 2.5em;
            font-weight: bold;
            color: #f0f6fc;
        }
        
        .stat-label {
            color: #8b949e;
            font-size: 0.9em;
            margin-top: 5px;
        }
        
        /* Main Layout */
        .main-grid {
            display: grid;
            grid-template-columns: 300px 1fr 350px;
            gap: 20px;
        }
        
        @media (max-width: 1400px) {
            .main-grid { grid-template-columns: 1fr; }
        }
        
        /* Pending Approvals */
        .pending-section {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 16px;
            padding: 20px;
        }
        
        .section-title {
            font-size: 1.2em;
            color: #f0f6fc;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .pending-card {
            background: #21262d;
            border: 1px solid #30363d;
            border-radius: 12px;
            padding: 15px;
            margin-bottom: 10px;
        }
        
        .pending-card .agent-name {
            font-weight: 600;
            color: #58a6ff;
            margin-bottom: 5px;
        }
        
        .pending-card .request-info {
            color: #8b949e;
            font-size: 0.85em;
            margin-bottom: 10px;
        }
        
        .pending-actions {
            display: flex;
            gap: 10px;
        }
        
        .btn {
            padding: 8px 16px;
            border-radius: 6px;
            border: none;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.2s;
        }
        
        .btn-approve {
            background: #238636;
            color: white;
        }
        
        .btn-approve:hover { background: #2ea043; }
        
        .btn-deny {
            background: #da3633;
            color: white;
        }
        
        .btn-deny:hover { background: #f85149; }
        
        /* Active Sandboxes Grid */
        .observation-center {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 16px;
            padding: 20px;
        }
        
        .sandbox-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 15px;
        }
        
        .sandbox-card {
            background: #0d1117;
            border: 2px solid #30363d;
            border-radius: 12px;
            overflow: hidden;
            transition: all 0.3s;
        }
        
        .sandbox-card.running { border-color: #238636; }
        .sandbox-card.paused { border-color: #f0883e; }
        .sandbox-card.attention { border-color: #da3633; animation: pulse 2s infinite; }
        
        @keyframes pulse {
            0%, 100% { box-shadow: 0 0 0 0 rgba(218, 54, 51, 0.4); }
            50% { box-shadow: 0 0 0 10px rgba(218, 54, 51, 0); }
        }
        
        .sandbox-header {
            padding: 15px;
            background: #161b22;
            border-bottom: 1px solid #30363d;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .sandbox-agent {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .agent-avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 0.9em;
            color: white;
        }
        
        .agent-info .name {
            font-weight: 600;
            color: #f0f6fc;
        }
        
        .agent-info .mode {
            font-size: 0.8em;
            color: #8b949e;
        }
        
        .sandbox-status {
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.75em;
            font-weight: 600;
            text-transform: uppercase;
        }
        
        .status-running { background: #238636; color: white; }
        .status-paused { background: #f0883e; color: white; }
        
        .sandbox-body {
            padding: 15px;
        }
        
        .score-display {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-bottom: 15px;
        }
        
        .score-item {
            background: #161b22;
            padding: 10px;
            border-radius: 8px;
            text-align: center;
        }
        
        .score-item .value {
            font-size: 1.5em;
            font-weight: bold;
        }
        
        .score-item .value.grade-S { color: #a371f7; }
        .score-item .value.grade-A { color: #3fb950; }
        .score-item .value.grade-B { color: #58a6ff; }
        .score-item .value.grade-C { color: #f0883e; }
        
        .score-item .label {
            color: #8b949e;
            font-size: 0.75em;
        }
        
        .current-activity {
            background: #161b22;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 15px;
        }
        
        .current-activity .label {
            color: #8b949e;
            font-size: 0.75em;
            margin-bottom: 5px;
        }
        
        /* No Active Sandboxes */
        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: #8b949e;
        }
        
        .empty-state .icon {
            font-size: 4em;
            margin-bottom: 20px;
        }
        
        /* Modal for detailed view */
        .modal-overlay {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0,0,0,0.8);
            z-index: 1000;
            justify-content: center;
            align-items: center;
        }
        
        .modal-content {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 16px;
            max-width: 800px;
            width: 90%;
            max-height: 80vh;
            overflow-y: auto;
            padding: 30px;
        }
        
        .modal-close {
            float: right;
            font-size: 1.5em;
            cursor: pointer;
            color: #8b949e;
        }
        
        .modal-close:hover { color: #f0f6fc; }
        
        /* Connection status */
        .connection-status {
            position: fixed;
            bottom: 20px;
            right: 20px;
            padding: 10px 20px;
            border-radius: 8px;
            font-size: 0.9em;
        }
        
        .connection-status.connected {
            background: #238636;
            color: white;
        }
        
        .connection-status.disconnected {
            background: #da3633;
            color: white;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🏖️ Sandbox Training Ground</h1>
            <p>Observe agents learning security tools through autonomous practice</p>
        </header>
        
        <nav>
            <a href="/">🏠 Home</a>
            <a href="/teams">👥 Teams</a>
            <a href="/deliberation">🧠 Deliberation</a>
            <a href="/sandbox" style="background:#21262d; border-color:#00CED1;">🏖️ Sandbox</a>
        </nav>
        
        <!-- Stats Banner -->
        <div class="stats-banner" id="statsBanner">
            <div class="stat-card active">
                <div class="stat-value" id="activeSandboxes">-</div>
                <div class="stat-label">Active Sandboxes</div>
            </div>
            <div class="stat-card pending">
                <div class="stat-value" id="pendingApprovals">-</div>
                <div class="stat-label">Pending Approval</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="totalActions">-</div>
                <div class="stat-label">Actions Today</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="avgSuccessRate">-</div>
                <div class="stat-label">Avg Success Rate</div>
            </div>
            <div class="stat-card attention">
                <div class="stat-value" id="needsAttention">-</div>
                <div class="stat-label">Need Attention</div>
            </div>
        </div>
        
        <!-- Main Grid -->
        <div class="main-grid">
            <!-- Pending Approvals -->
            <div class="pending-section">
                <h3 class="section-title">🔔 Pending Approvals</h3>
                <div id="pendingList">
                    <div class="empty-state" style="padding:30px">
                        <div style="font-size:2em">✅</div>
                        <p style="margin-top:10px">No pending requests</p>
                    </div>
                </div>
                
                <h3 class="section-title" style="margin-top:20px">📬 Recent Check-ins</h3>
                <div id="checkinList">
                    <p style="color:#8b949e;font-size:0.9em">No check-ins yet</p>
                </div>
            </div>
            
            <!-- Active Sandboxes -->
            <div class="observation-center">
                <h3 class="section-title">🖥️ Active Training Sessions</h3>
                <div class="sandbox-grid" id="sandboxGrid">
                    <div class="empty-state">
                        <div class="icon">🏖️</div>
                        <h3>No Active Sandboxes</h3>
                        <p style="margin-top:10px">Agents can request sandbox access to practice with security tools</p>
                    </div>
                </div>
            </div>
            
            <!-- Activity Feed -->
            <div class="activity-section">
                <h3 class="section-title">📡 Live Activity Feed</h3>
                <div class="activity-feed" id="activityFeed">
                    <p style="color:#8b949e;text-align:center;padding:20px">Waiting for activity...</p>
                </div>
            </div>
        </div>
        
        <!-- Scoreboard -->
        <div class="scoreboard-section">
            <h3 class="section-title">🏆 Training Scoreboard</h3>
            <table class="scoreboard-table" id="scoreboard">
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Agent</th>
                        <th>Sessions</th>
                        <th>Total Score</th>
                        <th>Best Grade</th>
                        <th>Tools Learned</th>
                    </tr>
                </thead>
                <tbody id="scoreboardBody">
                    <tr>
                        <td colspan="6" style="text-align:center;color:#8b949e;padding:30px">
                            No training data yet
                        </td>
                    </tr>
                </tbody>
            </table>
        </div>
    </div>
    
    <!-- Connection Status -->
    <div class="connection-status disconnected" id="connectionStatus">
        🔴 Disconnected
    </div>
    
    <!-- Detail Modal -->
    <div class="modal-overlay" id="detailModal">
        <div class="modal-content">
            <span class="modal-close" onclick="closeModal()">&times;</span>
            <div id="modalContent"></div>
        </div>
    </div>
    
    <script>
        // Agent color map
        const AGENT_COLORS = {
            'AJ': '#4169E1',
            'Axiom': '#9370DB',
            'Cipher': '#DC143C',
            'Vector': '#32CD32',
            'Nexus': '#FF8C00',
            'Echo': '#87CEEB',
            'Tesla': '#FFD700',
            'Flux': '#FF69B4',
            'Prism': '#E6E6FA',
            'Helix': '#00CED1',
            'Volt': '#FFFF00',
            'Spark': '#FFA500'
        };
        
        const MOOD_EMOJIS = {
            'focused': '🎯',
            'excited': '🤩',
            'confused': '😕',
            'frustrated': '😤',
            'accomplished': '😊',
            'curious': '🤔',
            'determined': '💪'
        };
        
        let ws = null;
        let reconnectAttempts = 0;
        
        // Connect WebSocket for live updates
        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/api/sandbox/ws/observe`);
            
            ws.onopen = () => {
                document.getElementById('connectionStatus').className = 'connection-status connected';
                document.getElementById('connectionStatus').textContent = '🟢 Live';
                reconnectAttempts = 0;
            };
            
            ws.onclose = () => {
                document.getElementById('connectionStatus').className = 'connection-status disconnected';
                document.getElementById('connectionStatus').textContent = '🔴 Disconnected';
                
                // Reconnect with backoff
                setTimeout(() => {
                    reconnectAttempts++;
                    if (reconnectAttempts < 10) connectWebSocket();
                }, Math.min(1000 * Math.pow(2, reconnectAttempts), 30000));
            };
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                handleWebSocketMessage(data);
            };
        }
        
        function handleWebSocketMessage(data) {
            if (data.event === 'connected') {
                // Initial data load
                if (data.recent_activity) {
                    data.recent_activity.forEach(addActivityItem);
                }
            } else if (data.event === 'heartbeat') {
                // Ignore heartbeats
            } else {
                // Add to activity feed
                addActivityItem(data);
                
                // Refresh dashboard for important events
                if (['sandbox_requested', 'sandbox_approved', 'sandbox_running', 
                     'sandbox_terminated', 'checkin', 'action'].includes(data.event)) {
                    refreshDashboard();
                }
            }
        }
        
        async function sendMessage() {
            const input = document.getElementById('user-input');
            const message = input.value.trim();
            if (!message) return;
            
            // Add user message
            addMessage('user', message);
            input.value = '';
            
            // Show deliberation section
            document.getElementById('deliberation-section').style.display = 'block';
            resetDeliberation();
            
            // Show loading
            addMessage('mini', '<div class="loading"><span></span><span></span><span></span></div>');
            
            try {
                const resp = await fetch('/api/core/process', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message })
                });
                
                const data = await resp.json();
                displayResponse(data);
                
            } catch (e) {
                removeLoading();
                addMessage('mini', 'Error processing request: ' + e.message);
            }
        }
        
        function displayResponse(data) {
            removeLoading();
            
            // Update intent badge
            if (data.intent) {
                const badge = document.getElementById('intent-badge');
                badge.textContent = data.intent.intent_type.toUpperCase();
                badge.className = 'intent-badge ' + data.intent.intent_type;
            }
            
            // Update deliberation if present
            if (data.deliberation) {
                const d = data.deliberation;
                
                if (d.aj_round1) {
                    updateAgentRound('aj', 1, d.aj_round1);
                }
                if (d.tesla_round1) {
                    updateAgentRound('tesla', 1, d.tesla_round1);
                }
                if (d.aj_round2) {
                    updateAgentRound('aj', 2, d.aj_round2);
                }
                if (d.tesla_round2) {
                    updateAgentRound('tesla', 2, d.tesla_round2);
                }
                if (d.synthesis) {
                    document.getElementById('synthesis-content').textContent = d.synthesis;
                }
                
                updatePhaseIndicator('complete');
            }
            
            // Add final message
            addMessage('mini', data.message);
            
            // Check for approval
            if (data.state === 'awaiting_approval') {
                // Will be handled by webhook
            }
        }
        
        function updateAgentRound(agent, round, response) {
            const el = document.getElementById(`${agent}-round${round}`);
            if (el && response) {
                let html = `<div class="round-label">Round ${round}</div>`;
                html += `<div class="round-text">${response.content || 'Processing...'}</div>`;
                
                if (response.prediction_of_other && round === 1) {
                    html += `<div class="prediction">Predicted other would focus on: ${response.prediction_of_other}</div>`;
                }
                
                el.innerHTML = html;
            }
        }
        
        function updatePhaseIndicator(phase) {
            const phases = ['round_1', 'round_2', 'synthesis', 'complete'];
            const phaseMap = {
                'round_1': 'phase-round1',
                'round_2': 'phase-round2',
                'synthesis': 'phase-synthesis',
                'complete': 'phase-synthesis'
            };
            
            // Reset all
            document.querySelectorAll('.phase').forEach(p => p.className = 'phase');
            
            // Mark complete phases
            let reachedCurrent = false;
            for (const p of phases) {
                const elId = phaseMap[p];
                const el = document.getElementById(elId);
                if (el) {
                    if (p === phase) {
                        el.classList.add('active');
                        reachedCurrent = true;
                    } else if (!reachedCurrent) {
                        el.classList.add('complete');
                    }
                }
            }
        }
        
        function resetDeliberation() {
            document.querySelectorAll('.phase').forEach(p => p.className = 'phase');
            ['aj-round1', 'aj-round2', 'tesla-round1', 'tesla-round2'].forEach(id => {
                const el = document.getElementById(id);
                if (el) {
                    el.innerHTML = '<div class="round-label">Waiting...</div><div class="round-text">...</div>';
                }
            });
            document.getElementById('synthesis-content').textContent = 'Waiting for deliberation...';
        }
        
        function addMessage(type, content) {
            const container = document.getElementById('chat-messages');
            const div = document.createElement('div');
            div.className = 'message ' + type;
            div.innerHTML = `<div class="bubble">${content}</div>`;
            container.appendChild(div);
            container.scrollTop = container.scrollHeight;
        }
        
        function removeLoading() {
            const loading = document.querySelector('.message .loading');
            if (loading) {
                loading.closest('.message').remove();
            }
        }
        
        function toggleDeliberation() {
            const content = document.getElementById('deliberation-content');
            const header = document.querySelector('.deliberation-header h2');
            content.classList.toggle('hidden');
            header.classList.toggle('collapsed');
        }
        
        function showApproval(request) {
            currentApprovalId = request.id;
            document.getElementById('approval-section').style.display = 'block';
            document.getElementById('approval-message').textContent = 
                `Action: ${request.intent}\\nRisk Level: ${request.risk_level}`;
            
            if (request.security_assessment) {
                document.getElementById('approval-assessment').innerHTML = 
                    `<p><strong>Security Assessment:</strong> ${request.security_assessment.reasoning}</p>`;
            }
        }
        
        async function approve(approved) {
            if (!currentApprovalId) return;
            
            await fetch(`/api/core/approvals/${currentApprovalId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ approved, notes: '' })
            });
            
            document.getElementById('approval-section').style.display = 'none';
            currentApprovalId = null;
            
            addMessage('mini', approved ? 'Action approved! Proceeding...' : 'Action rejected.');
        }
        
        // Initialize
        connectWS();
    </script>
</body>
</html>'''


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8001"))  # Allow override via PORT env var (changed default to 8001 to avoid conflicts)
    uvicorn.run(app, host="0.0.0.0", port=port)

