# ADR 0004: Prepare recipes before switching

Accepted · Recorded 2026-09-26 (retrospective) · Amended 2026-09-29

**Context.** Missing weights or runtime assets should not be discovered after
stopping a healthy model.

**Decision.** Keep host configuration in private TOML and launch settings in
catalogued recipes. Use CLI `pull` for preparation. Recipes with `preflight_args`
verify prepared assets before entering the disruptive controller transition.
Preflight resolves the local image to its immutable ID and uses read-only mounts,
no network, and no GPU access. Keep runtime and transport overrides recipe-specific.
Pin prepared runtime sources and checkpoints; verify their recorded hashes.
Bound each preflight with a recipe-configurable timeout before stopping workloads.

**Trade-off.** Preparation and qualification remain explicit, recipe-dependent
steps. Preflight is opt-in; recipes without it skip the check. A configured
context limit or a successful preflight does not establish inference quality.

**Evidence.** [CLI preparation and preflight](../../spark-serve),
[example catalog](../../models.example.toml),
[preflight tests](../../tests/test_model_preflight.py), [Qwen recipe](../qwen38-nvfp4.md),
[GLM preparation](../glm53-nvfp4.md), [Qwen v0.30 preparation](../qwen38-v030.md).
