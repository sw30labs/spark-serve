# ADR 0010: Run NIM through the existing controller

Accepted · Recorded 2026-09-29 (retrospective)

**Context.** NIM's two-node launch and readiness differ from vLLM, but GPU
ownership and cleanup must stay consistent across models.

**Decision.** Build a validated NIM launch plan, then use the existing fenced
controller. Start rank zero, validate its advertised primary address, then start
rank one. Require NIM readiness and the expected served model. Audit the selected
and departing recipes' ports on all listening interfaces before launch.

**Trade-off.** NIM needs backend-specific sequencing and health checks. Its
transport discovery stays inside NIM; model preparation and qualification remain
explicit recipe steps.

**Evidence.** [Launch plan](../../spark_serve_nim.py),
[CLI orchestration](../../spark-serve), [startup tests](../../tests/test_nim_startup.py),
[controller tests](../../tests/test_nim_controller.py), [GLM runbook](../glm53-nvfp4.md).
