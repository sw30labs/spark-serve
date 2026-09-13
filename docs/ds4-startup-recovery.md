# DeepSeek V4 Flash startup recovery on two DGX Sparks

## Failure and diagnosis

A failed head process left its peer holding loaded weights while the launcher
continued to display “booting.” Both ranks had loaded the model before the
head failed in the startup sampler's logits all-gather. The fatal message was
`ibv_reg_mr_iova2: Cannot allocate memory`, with Docker reporting exit 1 and
`OOMKilled=false`.

On a controlled retry, the same registration failure occurred before model
loading, with over 113 GiB available. A simultaneous kernel trace matched the
NCCL CUDA host-buffer address to the RDMA registration request. Linux failed
to migrate 37 pages while establishing a long-term pin and returned `ENOMEM`
through `migrate_longterm_unpinnable_folios`, `pin_user_pages_fast`,
`ib_umem_get`, and `mlx5_ib_reg_user_mr`.

This proves the immediate pinning/migration failure. CMA involvement and CUDA
pinning are plausible underlying interactions; the failed page classifications
and original pin owner were not captured. It is not a proven hardware fault.
Unlimited memlock, smaller buffers, fewer channels, extra HCAs and disabling
PCI relaxed ordering did not independently restore native RDMA. Basic
registration and small NCCL tests often passed, so those checks alone are
insufficient to accept a remedy.

The captured runtime used NVIDIA driver 580.173.02, kernel 7.0.0-1019-nvidia,
and NCCL 2.30.4 on GB10. PyTorch and vLLM loaded the same NCCL shared library;
PyTorch's different compile-time version string was not a second runtime.

## Reversible working recipe

The example catalog's text `ds4` recipe uses NCCL Socket over the existing
right-port QSFP interfaces to bypass verbs registration:

```toml
# In the existing [models.ds4] table:
docker_extra = ["--ulimit", "memlock=-1:-1"]

# In the existing [models.ds4.env] table:
NCCL_NET = "Socket"
NCCL_IB_DISABLE = "1"
NCCL_SOCKET_IFNAME = "=enp1s0f1np1,enP2p1s0f1np1"
NCCL_SOCKET_NTHREADS = "2"
NCCL_NSOCKS_PERTHREAD = "4"
NCCL_BUFFSIZE = "1048576"
NCCL_MAX_NCHANNELS = "8"
NCCL_DEBUG = "INFO"
NCCL_NVLS_ENABLE = "0"
```

Merge these keys into the existing tables; do not create duplicate TOML tables.
Confirm interface names match your hosts. Existing private `models.toml` files
are not overwritten by changing the example catalog. Shared cluster settings
and other models retain their configured transport.

The image remains `aidendle94/sparkrun-vllm-ds4-gb10:production-ready`, with a
1,048,576-token maximum context, FP8 KV cache and MTP 2. No driver, kernel,
network or host-reboot change is part of this workaround. It bypasses the
underlying RDMA problem rather than repairing it.

## Verification on September 12, 2026

- Full start succeeded in 299 seconds including stop/start orchestration;
  the readiness event reported 284 seconds of waiting.
- Both model containers remained running after two generated responses.
- `/v1/models` reported 1,048,576 maximum tokens. A million-token request was
  not tested.
- An exact-output request returned the expected text, with first output in
  5.60 seconds and total duration 5.89 seconds.
- A short paragraph produced 76 tokens, with first output in 2.27 seconds,
  total duration 5.93 seconds and approximately 20.49 output tokens/second.
- The GUI visibly showed the model serving. Additional Triton compilation
  occurred during the first requests; startup warmup did not cover every shape.

These two requests are smoke checks, not a sustained performance benchmark.
No controlled native-RDMA comparison establishes a numerical speedup or
slowdown. Restore RDMA only after validating a supported remedy with full
startup and generation, including the previously failing collective.

## Launcher behavior

The launcher tracks immutable IDs for every started rank, checks both during
readiness polling and rechecks survival before success. A failure saves bounded
container logs and state under
`~/.local/state/spark-serve/diagnostics/startup-*/failure.json` before the
existing guarded cleanup. SSH failures are distinguished from confirmed
container exits. The GUI preserves final errors, shows a green model check only
after readiness, and keeps controls busy until cleanup has finished.

Offline startup/controller tests and focused Swift process-output checks passed;
real failed starts exercised diagnostics and cleanup before the successful run.

## Primary references

- [NCCL 2.30.4 host allocation](https://github.com/NVIDIA/nccl/blob/v2.30.4-1/src/include/alloc.h#L115)
  calls CUDA mapped host allocation for this buffer path.
- [Linux long-term pinning](https://github.com/torvalds/linux/blob/master/mm/gup.c)
  explains migration failure propagation; the runtime uses NVIDIA's kernel build.
- [NVIDIA page locking](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/main/kernel-open/nvidia/os-mlock.c)
  gates the cited `FOLL_LONGTERM` branch on x86, also verified in the installed
  driver's source. This supports investigation of an ARM pinning interaction.
- [NVIDIA R535 issue 4429264](https://docs.nvidia.com/datacenter/tesla/tesla-release-notes-535-183-01/index.html)
  describes a previous CUDA/CMA/RDMA interaction. It is precedent, not proof
  that this captured failure is the identical defect.
