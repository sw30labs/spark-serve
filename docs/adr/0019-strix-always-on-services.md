# ADR 0019: Show always-on Strix services and leave their box alone

Accepted · Recorded 2026-11-05

**Context.** On 2026-11-04 strix-amd (192.168.86.51, formerly `amd-halo-2`)
got two systemd user units, both outside halo-ctl and both always on
(Restart=always, linger): `strix-llm.service`, gpt-oss-120b Q4_K_M in
llama.cpp b11849 on :8080 for the EFI and Pascal newsletters, and
`strix-embed.service`, Qwen3-Embedding-0.6B f16 on :8081 for Pascal's
weekly embeddings. The Strix tab then showed them wrongly (2026-11-05, read
with `halo-ctl status --json` and `spark-serve strix status --json`):

- halo-ctl finds its engines by port. Its cluster check is `GET :8081/health`,
  so it reported `{"engine": "cluster", "phase": "ready", "model":
  "qwen3-embedding-0.6b"}`. spark-serve then made strix-amd a cluster master,
  made the Minisforum its worker, and matched no catalog entry.
- gpt-oss-120b on :8080 was invisible: no halo-ctl engine uses that port. The
  only gpt-oss-120b the tab showed was Lemonade's on-disk copy, whose card was
  retired in 3cc8dd3, so it read "Not in the catalog".
- The danger was worse than the display. Any halo-ctl `up` or `stop` on that box
  runs `stop_cluster`, which kills whatever listens on :8081, so it would kill the
  embedding service. Starting an engine there would also take GPU memory from
  the 120B model, and the cluster could never get its worker port back.

**Decision.**
- **A new engine kind, `service`.** A `[strix.models.<id>]` entry with
  `engine = "service"` names `box`, `unit`, `port` and `served_name`, with the
  usual card fields, plus `outputs` (for example `["embeddings"]`) and
  `covers` (`<engine>:<halo-ctl id>` copies on disk that the card stands for).
- **Read-only health.** The probe reads `systemctl --user is-active <unit>`
  and the unit's `/v1/models`, `/props` and `/metrics` on localhost. It sends no
  key, because these servers take none. A service is `ready` only when its port
  answers with the catalog's `served_name`. If it answers with another name it
  is `mismatch`; if the unit is active but nothing answers it is `starting`;
  otherwise it is `stopped`.
- **Correct halo-ctl's misreading.** A halo-ctl report whose port and model are
  one of the box's services is treated as that service, so the box reads as
  "no engine". The box keeps a `shadowed` note that says so.
- **Pin the box.** spark-serve never starts or stops a service, and runs no
  halo-ctl `up` or `stop` on a box that has one. That includes using it as the
  cluster's worker. `stop` with no box skips it and says so. `connection` and
  `logs` cover the services.
- **App.** An "Always on" column holds the service cards. The box panel lists each
  service with its unit, phase, URL and served context. Stop is disabled on
  a pinned box, and Copy connection details works for its services.

**Trade-off.** While the services are configured, strix-amd cannot run any halo-ctl
engine or the cluster from the app. That is intended: they serve production
newsletters. To change what that box serves, stop or disable the units on the
box by hand, and remove or edit the catalog entries. Only the Minisforum
stays switchable from the app.

**Evidence.** `tests/test_strix.py` (the ADR 0019 tests). The live
`spark-serve strix status` showed strix-amd as "no engine" with both services
ready, and the Minisforum as stopped with no cluster role (2026-11-05).
