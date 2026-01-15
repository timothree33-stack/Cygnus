# Frontend layout

Canonical frontend directory: `frontend/frontend` (this is the app used for development, CI, and local dev server).

The top-level `frontend/` previously contained duplicate files for the same project. To avoid confusion, legacy artifacts were moved to `frontend/_legacy/`.

Files to note:
- `frontend/frontend/` — canonical app (contains `package.json`, `src/`, `vite.config.ts`, `public/`)
- `frontend/_legacy/` — archived files (node_modules, package.json, build artifacts)

To run development locally:
- `cd frontend/frontend && npm install && npm run dev`

The project `launch.sh` has been updated to prefer `frontend/frontend` when present.

Archived builds
- The previous build artifacts `frontend/frontend/dist` have been moved into `_legacy` as `frontend/_legacy/dist-frontend-20260115154818` (timestamped). This is archived and intentionally untracked to avoid confusing duplicate copies.