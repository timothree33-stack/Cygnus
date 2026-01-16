# CYGNUS MCP CORE — Specification (draft)

Purpose: provide a compact, unambiguous specification for the MCP manager (H2) and adjacent components to enable reproducible, local-only deployments and a clear implementation path for the `mcp/` package.

Principles
- Local-first models: artifacts must be downloaded, checksummed, verifiable, and launched in local runtimes (vLLM/llama.cpp/ONNX/Torch). No remote-hosted model instances are used in production workflows.
- Memory-first: all debate outputs, snapshots, and embeddings are persisted with provenance and versioning.
- Safety-first: any native OS interactions (camera, keyboard, mouse) must be opt-in, sandboxed and require explicit approval.

Key components
- Model Manifest (see `mcp/model_manifest.py`): canonical metadata describing a model artifact (name, version, format, source_url, checksum, tokenizer_spec, required_hw).
- Manager (H2 / `mcp/manager.py`): lifecycle for model artifacts (download -> verify -> convert -> launch -> health). Exposes an async model client interface and orchestration hooks (agent registry, debate orchestration entrypoints).
- Model Runtime Adapters: per-runtime adapters implementing a small `ModelBackend` interface (async `respond(...)`, stream, health). Provide `vllm_adapter`, `llama_cpp_adapter`, `onnx_adapter`.
- ZProxy: routing, security and message policy enforcement. Routes requests between clients, manager and tool adapters.
- Episodic Memory (`mcp/episodic.py`): thin API over the existing sqlite store that exposes timeline operations (commit, query, export).
- Visual Language Adapter (`mcp/visual_language.py`): camera integration and frame chunk pipeline; includes CI-safe camera fallback.

Model Manifest
- Fields: `name, version, source_url, checksum, format, quantization, tokenizer, required_hw, license`.
- Manager MUST verify checksum before accepting an artifact for launch.

APIs (high level)
- Manager
  - `initialize(config)`
  - `download_model(manifest: ModelManifest) -> path`
  - `verify_model(manifest) -> bool`
  - `launch_model(manifest) -> ModelBackend` (returns an instance implementing ModelBackend)
  - `list_models()`, `get_model_status(model_name)`
- ModelBackend
  - `async respond(prompt, stream=False) -> response` (stream emits partial tokens)
  - `health()`

Operational notes
- K8s: Manager should emit manifests / helm values for model deployment; prefer containers with pinned runtime images and model cache volumes for reproducibility.
- Metrics & Logs: Manager and per-model runtime must export Prometheus metrics and structured JSON logs.

Security
- Enforce signature verification for production artifacts.
- ZProxy must support allow/deny rules for network egress.

Next steps
1. Implement `mcp/model_manifest.py` dataclass and `mcp/manager.py` skeleton.
2. Add unit tests for manifest parsing and manager lifecycle stubs.
3. Iterate by adding adapters and integration tests that run a tiny local model in CI.

(End of spec - draft)
