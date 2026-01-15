// Central place for API base URL used by frontend
// Defaults to empty string so that relative paths hit the Vite dev proxy (/api -> backend)
export const API_BASE = import.meta.env.VITE_API_BASE ?? '';
