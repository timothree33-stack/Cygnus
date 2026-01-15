// Use relative base by default so Vite dev proxy (vite.config.ts) can forward /api to the backend.
export const API_BASE = import.meta.env.VITE_API_BASE ?? '';
// WebSocket path (relative); proxied by Vite in development.
export const WS_PATH = import.meta.env.VITE_WS_PATH ?? '/ws';
