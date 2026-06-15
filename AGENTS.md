# AGENTS.md

Use `AGENT.md` as the canonical project guidance file for this repository.

The current active direction is SRSA staged online family replay V1:

- keep each online train job single-assembly;
- save current replay after each stage;
- mix current / `01125` anchor / history replay during later stages;
- keep `online_family_replay_*` separate from offline `multitask_continuation_*`.
