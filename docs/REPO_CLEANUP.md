# Repo cleanup & archive plan — Cygnus Pyramid

Purpose
- Reduce noise, flaky CI, and long-running dependency installs by archiving large unrelated subprojects and scoping test discovery to Cygnus-specific code only.

Why
- The repository contains multiple large subprojects (e.g., TensorRT-LLM, llama.cpp, various examples) with heavy test dependencies (PyTorch, NumPy, Triton) that are not required for active Cygnus development.
- Running pytest at the repository root attempts to collect and run all tests, which causes frequent collection/import failures and slows developer iteration.

Goals
1. Make local and CI test runs fast and reliable by default.
2. Preserve history and keep archived projects available (snapshot branch or `archive/` folder) so we can restore them later if needed.
3. Keep the cleanup reversible and transparent to maintainers.

Immediate actions (safe & reversible)
- Add `pytest.ini` at the repo root to limit test discovery to `backend/tests` (and frontend tests run with their own tooling). (Done in PR) ✅
- Add a short `docs/REPO_CLEANUP.md` describing the plan, rationale, and checklist. (This document.) ✅
- Create a branch `archive/2026-01-16` and move large unrelated folders there (optional; will be done after review). ⚠️

Recommended follow-up (after review)
- Create `archive/2026-01-16` branch containing the archived folders and an index listing what's archived.
- Update CI to run the frontend test job separately (Vitest/Playwright) and keep backend pytest job scoped (already updated). ✅
- Add a GitHub Issue titled: "Repo cleanup: archive non‑Cygnus subprojects" and link to this doc. 📝

Notes & risks
- Archiving avoids loss of history; don't delete until stakeholders sign off.
- If the repo integrates external tooling that depends on archived files, update docs or scripts to reference the archive snapshot.

If you approve, I'll open a PR with `pytest.ini` + this doc and a short change to `.github/workflows/run-tests.yml` so CI only runs `pytest backend/tests` (keeps it tight). After review/approval we can proceed to snapshot/archive large directories on a separate branch.
