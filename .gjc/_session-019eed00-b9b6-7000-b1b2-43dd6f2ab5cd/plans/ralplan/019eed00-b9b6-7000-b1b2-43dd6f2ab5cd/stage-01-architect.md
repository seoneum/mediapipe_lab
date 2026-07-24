## Summary
G001 covers the requested dossier-first local MVP flows: local dossier JSON storage, dossier update commands, approved session summaries, baseline recommendation drafting and approval capture, and human-readable handoff brief generation. However, the ON DAMM CLI is transitively coupled to mediapipe, which violates the explicit no sensor dependency constraint for this story and blocks approval until the dossier shell can run without the sensor stack.

## Analysis
The local-first data model is clear and scoped: Dossier stores only local continuity fields plus approved session summaries, approved plan history, and audit records in JSON at app/ondamm_models.py lines 140-149 and 237-246, and persistence stays on local disk under data/ondamm/dossiers and outputs/ondamm at app/ondamm_store.py lines 25-53 and README.md lines 93-96. The CLI exposes the required product actions for dossier creation, viewing, editing, approved session capture, baseline recommendation generation, and handoff brief export at app/ondamm_cli.py lines 228-282.

The product guardrails are mostly represented in the artifact layer. The handoff renderer explicitly says the brief is human-readable, not for recipient-side import or promotion, and requires manual dossier reconstruction in the new environment at app/ondamm_cli.py lines 44-47, and the README repeats the support-not-diagnosis and manual transfer rules at README.md lines 51-55. Session summaries require an approver at app/ondamm_cli.py lines 260-268 and app/ondamm_models.py lines 35-56, and recommendations persist into plan history only when approved at app/ondamm_cli.py lines 188-203 and app/ondamm_models.py lines 87-110. The recommendation text is conservative support planning, not diagnosis, ranking, or compliance scoring at app/ondamm_recommendations.py lines 18-45.

The blocker is architectural coupling. app/paths.py imports mediapipe at module import time on line 3. The ON DAMM store imports ON DAMM path constants from that module at app/ondamm_store.py line 7, the CLI imports the store at app/ondamm_cli.py line 9, and the shell entrypoint always launches that CLI at scripts/ondamm_mvp.sh line 7. As a result, a story that is supposed to be local-first and sensor-free still requires the sensor runtime to be importable before any dossier command can start. That directly conflicts with the README claim that sensors are only secondary and with the acceptance constraint that there is no sensor dependency in this story at README.md lines 52-55.

## Root Cause
ON DAMM path configuration was added to the shared app/paths.py module instead of a sensor-free ON DAMM specific path module. Because paths.py also owns MediaPipe model helpers and imports mediapipe, the dossier-only CLI inherited an unnecessary runtime dependency on the sensor stack.

## Findings
- Severity: HIGH
  - File: app/paths.py line 3, app/ondamm_store.py line 7, app/ondamm_cli.py line 9, scripts/ondamm_mvp.sh line 7
  - Impact: The ON DAMM MVP cannot be considered sensor-independent because even simple dossier commands depend on successfully importing mediapipe. This breaks the story boundary and makes the local continuity workflow unavailable in environments where the sensor stack is intentionally absent.
  - Fix suggestion: Split ON DAMM filesystem constants into a dedicated sensor-free module, or move MediaPipe-specific imports and base_options into a separate camera or model helper module. The ON DAMM store and CLI should import only pure path constants with no sensor runtime side effects.

## Recommendations
1. Remove the transitive mediapipe dependency from the ON DAMM CLI path by separating ON DAMM path constants from MediaPipe helpers.
2. Keep the current product contract: approved session summaries, optional draft recommendations, approved recommendation persistence, and manual handoff wording are all aligned with the MVP scope.
3. After decoupling, add a narrow CLI-level verification that ondamm_mvp.sh can run dossier commands in a Python environment that does not have MediaPipe installed.

## Architectural Status
BLOCK

## Code Review Recommendation
REQUEST CHANGES

## Trade-offs
- Keep shared paths.py
  - Pros: one place for filesystem constants.
  - Cons: leaks sensor runtime requirements into dossier-only workflows and violates the story boundary.
- Split ON DAMM paths from MediaPipe helpers
  - Pros: preserves local-first MVP independence, simplifies packaging, and matches the no sensor dependency requirement.
  - Cons: one extra small module or import boundary to maintain.
