"""Pure launch-plan helpers for the two-Spark NVIDIA NIM backend.

The CLI owns SSH, generation fencing, immutable container receipts and polling.
These helpers never inspect files, start processes or contact either Spark.
Prepared checkpoint contents must still pass the recipe's offline verifier before
the controller stops a workload. The launch plan pins the image and checkpoint
location that verifier must inspect.

NVIDIA's Spark contract starts rank zero, reads its NIM_PRIMARY_NODE log value,
then starts rank one. Hardware transport settings are discovered by NIM itself:
https://docs.nvidia.com/nim/vision-language-models/latest/deploy-on-dgx-spark.html
"""
from __future__ import annotations

from dataclasses import dataclass
import ipaddress
import json
import math
from pathlib import PurePosixPath
import re

from spark_serve_controller import ControllerError


_DIGEST = re.compile(r"(?:sha256:[0-9a-f]{64}|[a-z0-9][a-z0-9._:/-]*@sha256:[0-9a-f]{64})\Z")
_NAME = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*\Z")
_ENV_NAME = re.compile(r"[A-Z_][A-Z0-9_]*\Z")
_TRANSPORT_PREFIXES = ("NCCL_", "UCX_", "GLOO_", "TP_", "NVSHMEM_", "OMPI_MCA_")
_RESERVED_ENV = {
    "NIM_PRIMARY_NODE", "NIM_NODE_MANAGER_PORT", "NIM_SERVER_PORT",
    "NIM_MODEL_PATH", "NIM_CACHE_PATH", "MASTER_ADDR", "MASTER_PORT",
    "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES",
    "NIM_PASSTHROUGH_ARGS", "NIM_TENSOR_PARALLEL_SIZE", "NIM_PIPELINE_PARALLEL_SIZE",
}
_NIM_FIELDS = {
    "model_source", "model_revision", "manager_port", "worker_port",
    "handshake_timeout", "runtime_cache_head", "runtime_cache_worker",
    "ffmpeg_path_head", "ffmpeg_path_worker",
}


@dataclass(frozen=True)
class NimPlan:
    model_id: str
    image: str
    container: str
    hosts: tuple[str, str]
    model_source: str
    model_revision: str
    hf_mount: str
    serve_path: str
    cache_paths: tuple[str, str]
    runtime_cache_paths: tuple[str, str]
    ffmpeg_paths: tuple[str | None, str | None]
    server_port: int
    worker_port: int
    manager_port: int
    handshake_timeout: int
    shm_size: str
    env: tuple[tuple[str, str], ...]


def _text(value, name: str) -> str:
    if not isinstance(value, str) or not value or any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise ControllerError(f"NIM {name} must be a nonempty string without control characters")
    return value


def _path(value, name: str) -> str:
    value = _text(value, name)
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts or ":" in value or str(path) == "/":
        raise ControllerError(f"NIM {name} must be an absolute non-root POSIX path without '..' or ':'")
    return str(path)


def _integer(value, name: str, minimum: int = 1, maximum: int = 65535) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ControllerError(f"NIM {name} must be an integer in {minimum}..{maximum}")
    return value


def validate_nim_plan(cfg: dict, mid: str) -> NimPlan:
    """Validate every static launch input before any workload is drained.

    ``image`` must be an immutable local image ID or registry digest. NGC models
    use an explicit version directory; HF models use
    the conventional complete cache plus a 40-hex snapshot revision. Checking
    file contents and the local image identity remains the preflight's job.
    """
    try:
        cluster, model = cfg["cluster"], cfg["models"][mid]
    except (KeyError, TypeError) as exc:
        raise ControllerError(f"NIM model {mid!r} is missing from the catalog") from exc
    if not isinstance(cluster, dict) or not isinstance(model, dict):
        raise ControllerError("NIM cluster and model configuration must be tables")
    if model.get("wrapper") != "nim":
        raise ControllerError("NIM recipes must declare wrapper = 'nim'")
    if model.get("ready_path") != "/v1/health/ready":
        raise ControllerError("NIM ready_path must be /v1/health/ready; model listing is not engine readiness")
    _integer(model.get("nnodes"), "nnodes", 2, 2)
    if "tensor_parallel" in model:
        _integer(model["tensor_parallel"], "tensor_parallel", 2, 2)
    hosts = tuple(_text(cluster.get(role), f"cluster.{role}") for role in ("head", "worker"))
    if hosts[0] == hosts[1]:
        raise ControllerError("NIM requires two distinct Spark hosts")
    image = _text(model.get("image"), "image")
    if not _DIGEST.fullmatch(image):
        raise ControllerError("NIM image must be pinned by @sha256 digest or immutable sha256 image ID")
    nim = model.get("nim")
    if not isinstance(nim, dict):
        raise ControllerError("NIM recipes require a [models.<id>.nim] table")
    unknown = set(nim) - _NIM_FIELDS
    if unknown:
        raise ControllerError(f"unknown NIM settings: {', '.join(sorted(unknown))}")
    source = nim.get("model_source")
    if source not in ("hf", "ngc"):
        raise ControllerError("NIM model_source must explicitly be 'hf' or 'ngc'")
    revision = _text(nim.get("model_revision"), "model_revision")
    if not _NAME.fullmatch(revision) or revision.lower() in {"main", "master", "latest", "stable", "current", "default"}:
        raise ControllerError("NIM model_revision must identify a pinned checkpoint version")
    mount = _path(model.get("hf_mount", "/cache/huggingface"), "hf_mount")
    serve = _path(model.get("serve_path"), "serve_path")
    if not PurePosixPath(serve).is_relative_to(mount) or serve == mount:
        raise ControllerError("NIM serve_path must be inside the read-only model cache mount")
    if source == "hf":
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ControllerError("NIM Hugging Face model_revision must be a 40-hex commit")
        repository = _text(model.get("hf_id"), "hf_id")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
            raise ControllerError("NIM hf_id must be a namespace/repository")
        expected = str(PurePosixPath(mount) / "hub" / ("models--" + repository.replace("/", "--")) / "snapshots" / revision)
        if serve != expected:
            raise ControllerError("NIM serve_path does not identify the pinned HF snapshot")
    elif PurePosixPath(serve).name not in (revision, "model-" + revision):
        raise ControllerError("NIM NGC serve_path must end with its pinned model_revision")
    cache_head = _path(cluster.get("hf_cache_host"), "hf_cache_host")
    cache_worker = _path(cluster.get("worker_hf_cache_host") or cache_head, "worker_hf_cache_host")
    cache_paths = (cache_head, cache_worker)
    runtime_paths = []
    ffmpeg_paths = []
    for role, cache in zip(("head", "worker"), cache_paths):
        runtime = _path(nim.get(f"runtime_cache_{role}") or str(PurePosixPath(cache) / "spark-serve/glm53-nvfp4/runtime-cache"), f"runtime_cache_{role}")
        weights = PurePosixPath(cache) / PurePosixPath(serve).relative_to(mount)
        # A writable mount must not make the model tree or HF blobs writable.
        if (PurePosixPath(cache).is_relative_to(runtime)
                or weights.is_relative_to(runtime) or PurePosixPath(runtime).is_relative_to(weights)
                or PurePosixPath(runtime).is_relative_to(PurePosixPath(cache) / "hub")):
            raise ControllerError("NIM runtime cache must be separate from checkpoint and HF hub files")
        runtime_paths.append(runtime)
        ffmpeg = nim.get(f"ffmpeg_path_{role}")
        ffmpeg_paths.append(_path(ffmpeg, f"ffmpeg_path_{role}") if ffmpeg is not None else None)
    if (ffmpeg_paths[0] is None) != (ffmpeg_paths[1] is None):
        raise ControllerError("NIM FFmpeg must be configured on both Sparks or neither")
    container = _text(model.get("container") or cluster.get("container") or "nim_glm53", "container")
    if not _NAME.fullmatch(container):
        raise ControllerError("NIM container must be a Docker container name")
    server_port = _integer(cluster.get("port", 8000), "server_port")
    worker_port = _integer(nim.get("worker_port", 8002), "worker_port")
    manager_port = _integer(nim.get("manager_port", 20000), "manager_port")
    if len({server_port, worker_port, manager_port}) != 3:
        raise ControllerError("NIM server, worker and manager ports must differ")
    timeout = _integer(nim.get("handshake_timeout", 300), "handshake_timeout", 1, 3600)
    shm = _text(model.get("shm_size", "16g"), "shm_size")
    if not re.fullmatch(r"[1-9][0-9]*[bkmgBKMG]?", shm):
        raise ControllerError("NIM shm_size must be a positive Docker memory size")
    vllm = model.get("vllm") or {}
    if not isinstance(vllm, dict):
        raise ControllerError("NIM vllm configuration must be an empty table")
    if model.get("docker_extra") or model.get("mounts") or model.get("skip_default_cache_mount") or vllm:
        raise ControllerError("NIM uses its validated launch plan; Docker/mount/vLLM overrides are unsupported")
    raw_env = model.get("env") or {}
    if not isinstance(raw_env, dict):
        raise ControllerError("NIM environment must be a table")
    env = {}
    for key, value in raw_env.items():
        if not isinstance(key, str) or not _ENV_NAME.fullmatch(key):
            raise ControllerError("NIM environment keys must be uppercase variable names")
        if key in _RESERVED_ENV or key.startswith(_TRANSPORT_PREFIXES):
            raise ControllerError(f"NIM discovers its transport and owns structural setting {key}")
        if key.endswith(("_KEY", "_TOKEN", "_PASSWORD", "_SECRET")):
            raise ControllerError("NIM offline launch must not embed credentials in the catalog")
        if not isinstance(value, (str, int, float, bool)):
            raise ControllerError(f"NIM environment {key} must be a scalar")
        rendered = str(value) if not isinstance(value, bool) else ("1" if value else "0")
        env[key] = _text(rendered, f"environment {key}")
    if env.get("NIM_LOG_LEVEL", "INFO").upper() not in {"TRACE", "DEBUG", "INFO"}:
        raise ControllerError("NIM_LOG_LEVEL must retain INFO messages for the rank-zero handshake")
    for key in ("NIM_KV_CACHE_PERCENT", "NIM_KVCACHE_PERCENT"):
        if key in env:
            try:
                fraction = float(env[key])
            except ValueError as exc:
                raise ControllerError(f"{key} must be a finite fraction between zero and one") from exc
            if not math.isfinite(fraction) or not 0 < fraction < 1:
                raise ControllerError(f"{key} must be a finite fraction between zero and one")
    served = _text(model.get("served_name"), "served_name")
    context = _integer(model.get("max_model_len"), "max_model_len", 1, 1048576)
    if "NIM_SERVED_MODEL_NAME" in env and env["NIM_SERVED_MODEL_NAME"] != served:
        raise ControllerError("NIM_SERVED_MODEL_NAME must match the catalog served_name")
    if "NIM_MAX_MODEL_LEN" in env and env["NIM_MAX_MODEL_LEN"] != str(context):
        raise ControllerError("NIM_MAX_MODEL_LEN must match the catalog max_model_len")
    env.update(NIM_SERVED_MODEL_NAME=served, NIM_MAX_MODEL_LEN=str(context))
    return NimPlan(mid, image, container, hosts, source, revision, mount, serve,
                   cache_paths, tuple(runtime_paths), tuple(ffmpeg_paths), server_port,
                   worker_port, manager_port, timeout, shm, tuple(sorted(env.items())))


def _primary_ip(value: str) -> str:
    if not isinstance(value, str):
        raise ControllerError("NIM_PRIMARY_NODE must be an IP address advertised by rank zero")
    try:
        address = ipaddress.ip_address(value)
    except (ValueError, TypeError) as exc:
        raise ControllerError("NIM_PRIMARY_NODE must be an IP address advertised by rank zero") from exc
    if address.is_unspecified or address.is_loopback or address.is_multicast or address.is_link_local or "%" in str(address):
        raise ControllerError("NIM_PRIMARY_NODE must be a reachable unicast IP address")
    return str(address)


def parse_nim_primary_node(logs: str) -> str | None:
    """Read a single unambiguous handshake from *this launch's rank-zero logs*.

    Return None while the expected line has not appeared. Malformed or conflicting
    advertisements fail closed rather than starting rank one at an arbitrary host.
    """
    if not isinstance(logs, str):
        raise ControllerError("NIM rank-zero logs must be text")
    plain = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", logs)
    values = re.findall(r"(?<![A-Za-z0-9_])NIM_PRIMARY_NODE\s*=\s*([^\s]+)", plain)
    if not values:
        return None
    addresses = {_primary_ip(value.strip("\"'")) for value in values}
    if len(addresses) != 1:
        raise ControllerError("rank zero advertised conflicting NIM_PRIMARY_NODE addresses")
    return addresses.pop()


def nim_docker_argv(plan: NimPlan, rank: int, generation: str, primary_node: str | None = None) -> list[str]:
    """Build Docker *create* argv; the CLI fences create/start and records its ID.

    No shell, host environment or cluster transport settings are interpolated.
    Rank one cannot be constructed until rank zero has advertised its address.
    """
    if isinstance(rank, bool) or rank not in (0, 1):
        raise ControllerError("NIM rank must be zero or one")
    generation = _text(generation, "allocation generation")
    if rank == 0 and primary_node is not None:
        raise ControllerError("NIM rank zero must discover its own primary address")
    primary = _primary_ip(primary_node) if rank == 1 else None
    argv = ["docker", "create", "--pull", "never", "--name", plan.container,
            "--network", "host", "--gpus", "all", "--shm-size", plan.shm_size,
            "--device", "/dev/infiniband:/dev/infiniband:rwm", "--ulimit", "memlock=-1:-1"]
    for key, value in {"ai.spark-serve.model": plan.model_id,
                       "ai.spark-serve.hosts": json.dumps(plan.hosts, separators=(",", ":")),
                       "ai.spark-serve.allocation": generation}.items():
        argv.extend(["--label", f"{key}={value}"])
    argv.extend(["-v", f"{plan.cache_paths[rank]}:{plan.hf_mount}:ro",
                 "-v", f"{plan.runtime_cache_paths[rank]}:/opt/nim/.cache"])
    if plan.ffmpeg_paths[rank] is not None:
        argv.extend(["-v", f"{plan.ffmpeg_paths[rank]}:/opt/ffmpeg8:ro"])
    env = dict(plan.env)
    env.update(NIM_MODEL_PATH=plan.serve_path, NIM_CACHE_PATH="/opt/nim/.cache",
               NIM_NODE_MANAGER_PORT=str(plan.manager_port),
               NIM_SERVER_PORT=str(plan.server_port if rank == 0 else plan.worker_port))
    if primary is not None:
        env["NIM_PRIMARY_NODE"] = primary
    for key, value in sorted(env.items()):
        argv.extend(["-e", f"{key}={value}"])
    argv.append(plan.image)
    return argv
