# Agent Control Panel

A multi-agent AI control interface with voice, dynamic avatars, and native computer control.

## Features

- **MiniOrca** - Primary RL agent with native control (mouse, keyboard, screen)
- **AJ's Team** (6 agents) - Business & Operations specialists
- **Tesla's Team** (6 agents) - Innovation & Research specialists
- **Voice Module** - TTS (edge-tts/pyttsx3/espeak) and STT (whisper/vosk/google)
- **Dynamic Avatars** - Emotion-reactive SVG avatars that change in real-time
- **Persistent Memory** - SQLite-backed conversation history with FTS search
- **Web UI** - Three-panel interface accessible from any device

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AGENT CONTROL PANEL                               │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │   MiniOrca   │  │  AJ's Team   │  │ Tesla's Team │               │
│  │   (Primary)  │  │  (Dropdown)  │  │  (Dropdown)  │               │
│  │              │  │              │  │              │               │
│  │ [  Avatar  ] │  │ [  Avatar  ] │  │ [  Avatar  ] │               │
│  │ [Chat Box  ] │  │ [Chat Box  ] │  │ [Chat Box  ] │               │
│  │ [Voice/TTS ] │  │ [Voice/TTS ] │  │ [Voice/TTS ] │               │
│  └──────────────┘  └──────────────┘  └──────────────┘               │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                  NATIVE CONTROL (MiniOrca Only)                 │ │
│  │  Mouse: move, click, drag, scroll                               │ │
│  │  Keyboard: type, hotkeys, special keys                          │ │
│  │  Screen: screenshot, OCR, element detection                     │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

## Agent Teams

### MiniOrca (Primary)
The main RL agent that can actually take actions on your computer:
- Mouse control (move, click, drag, scroll)
- Keyboard input (typing, hotkeys)
- Screen reading (screenshots, OCR)
- Application control

### AJ's Team - Business & Operations
| Agent | Specialization |
|-------|---------------|
| **AJ** | Lead Coordinator - project management, task delegation |
| **Axiom** | Code/CS/Quantum Expert - ALL programming languages, algorithms |
| **Cipher** | Security & Cryptography - penetration testing, secure coding |
| **Vector** | Data Science & ML - analysis, ML pipelines, visualization |
| **Nexus** | Systems & APIs - integration, microservices, infrastructure |
| **Echo** | Communication & NLP - language processing, documentation |

### Tesla's Team - Innovation & Research
| Agent | Specialization |
|-------|---------------|
| **Tesla** | Lead Innovator - breakthrough thinking, future tech |
| **Flux** | Hardware & IoT - embedded systems, robotics |
| **Prism** | Visualization & UI/UX - design systems, accessibility |
| **Helix** | Research Synthesis - paper analysis, knowledge discovery |
| **Volt** | Performance Optimization - profiling, scaling |
| **Spark** | Creative AI - generative models, creative applications |

## Installation

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install fastapi uvicorn websockets pydantic

# Optional: Voice support
pip install edge-tts pyttsx3 SpeechRecognition

# Optional: Native control
pip install pyautogui pillow

# Optional: STT with Whisper
pip install openai-whisper

# Linux: Install system dependencies
sudo apt install espeak tesseract-ocr xdotool scrot
```

## Running

```bash
# Start the server
cd /home/sophia/Desktop
python -m agent_panel.app

# Or with uvicorn directly
uvicorn agent_panel.app:app --host 0.0.0.0 --port 8080 --reload
```

Access the UI at:
- Local: http://localhost:8080
- Tailscale: http://100.67.234.54:8080 (from your phone)

## API Endpoints

### Agents
- `GET /api/agents` - List all agents
- `GET /api/agents/{name}` - Get agent details
- `GET /api/agents/{name}/avatar` - Get agent's avatar SVG
- `POST /api/agents/{name}/emotion` - Set agent emotion

### Conversations
- `GET /api/conversations` - List conversations
- `POST /api/conversations` - Create new conversation
- `GET /api/conversations/{id}` - Get conversation with messages
- `POST /api/conversations/{id}/messages` - Add message

### Native Control
- `GET /api/native/screen` - Describe current screen
- `POST /api/native/screenshot` - Take screenshot
- `POST /api/native/action` - Execute native action

**Security note:** Native control is disabled by default. To enable native control, set the environment variable `ENABLE_NATIVE_CONTROL=1` and set a strong auth key in `NATIVE_CONTROL_AUTH_KEY`. All API requests must include an `X-NATIVE-AUTH: <key>` header. This ensures native OS actions (mouse, keyboard, screenshot, window operations) remain opt-in and auditable.

### Voice
- `GET /api/voice/test/{agent}` - Test agent voice

### Memory
- `GET /api/memory/stats` - Memory statistics
- `GET /api/memory/{agent}` - Agent's learned memory

## WebSocket

Connect to `/ws/{conversation_id}` for real-time updates:

```javascript
const ws = new WebSocket('ws://localhost:8080/ws/conv123');

ws.send(JSON.stringify({
    type: 'chat',
    content: 'Hello!',
    agent_id: 'mini-orca-001'
}));

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    // data.type: 'agent_response' | 'avatar_update'
};
```

## Files

```
agent_panel/
├── __init__.py      # Architecture overview
├── agents.py        # Agent definitions (13 agents)
├── voice.py         # TTS/STT module
├── avatars.py       # Emotion detection & avatar generation
├── memory.py        # Persistent memory (SQLite + FTS)
├── native_control.py # Mouse, keyboard, screen control
├── app.py           # FastAPI web application
└── README.md        # This file
```

## License

MIT

## Playground (Lux) — Daily simulated CVE challenges

- **Purpose:** A safe, simulated red-team playground that creates daily CVE-based challenges (metadata-only). The attack agent "Lux" proposes challenges; defenders run simulations locally against an emulated target (no real network attacks or exploit code are executed).
- **How to enable admin refresh:** set the environment variable `PLAYGROUND_ADMIN_KEY` to a secret value that will protect NVD refresh operations. Example (Linux/macOS):

```bash
export PLAYGROUND_ADMIN_KEY="supersecretkey"
```

- **Using the UI:** Open the Agent Control Panel (root URL). The Playground panel shows today's challenges. To fetch additional CVE metadata from NVD, enter the admin key in the Playground panel input and click **Refresh NVD** (this calls the protected `/playground/admin/refresh-nvd` endpoint). The refresh is metadata-only and rate-limited.

- **Running a simulation:** Click **Start Simulation** on a listed challenge — this triggers a deterministic, sandboxed simulation run (no exploit code). The run creates a record in `playground.db` under `runs` and may push a text-only summary to the configured MCP via `MCP_URL` for agent memory/reinforcement.

- **Agent context:** You can upload contextual corpora for an agent via the API: `POST /playground/agents/{agent_name}/context` with JSON payload; these are stored locally and can be forwarded to MCP for per-agent memories.

- **Safety:** The system intentionally never fetches or executes exploit code. Use only on isolated lab environments if you later wire real sandboxed containers.

