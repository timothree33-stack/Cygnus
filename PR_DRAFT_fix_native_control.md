PR Draft: security(native_control): gate controller behind ENABLE_NATIVE_CONTROL, require auth key, add audit logging and tests

Summary
-------
This draft PR hardens the native control surface to reduce the risk of accidental or unauthorized OS-level actions:

- Do not instantiate the native controller on import/startup. Use env var `ENABLE_NATIVE_CONTROL=1` to opt-in.
- Require `NATIVE_CONTROL_AUTH_KEY` to be set when native control is enabled and require clients to present `X-NATIVE-AUTH: <key>` header.
- Add structured audit logging (`native_action_requested`, `native_action_executed`) to `NativeController.execute_action()` including `action_id`, timestamps, duration, action payload and caller metadata.
- Add unit tests that assert: disabled-by-default behavior, auth key requirement, and authorized request flow using a stub controller.
- Update `agent_panel/README.md` with a small security note documenting enable/auth steps.

Files changed
-------------
- agent_panel/native_control.py (add time/uuid imports; add audit logging and caller metadata support; gate get_native_controller)
- agent_panel/app.py (do not instantiate controller at import; require auth + env key on native endpoints; check controller enabled)
- agent_panel/tests/test_native_control_security.py (new tests)
- agent_panel/README.md (security note)

Review notes / discussion points
-------------------------------
- Current approach requires `NATIVE_CONTROL_AUTH_KEY` and `ENABLE_NATIVE_CONTROL=1`. Alternative: allow localhost access without key (decided here to require a key to be explicit and safer).
- Consider adding an app-level audit store (append-only file / DB) rather than relying on structured logs if high-assurance audits are needed.
- Additional improvements: rate-limits, integration with approval gate for high-risk actions, and an admin API to rotate/inspect auth keys.

Testing
-------
- New tests added under `agent_panel/tests/`; run `pytest agent_panel/tests/test_native_control_security.py`.

If this looks good, I can push the branch `fix/native-control-safety` to the remote as a draft PR for final review and CI run.  Let me know to proceed.