"""Validated, single-Spark TensorFold launch arguments and CUDA health checks."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import PurePosixPath
import re

from spark_serve_controller import ControllerError


IMMUTABLE_IMAGE = re.compile(r"(?:sha256:[0-9a-f]{64}|[a-z0-9][a-z0-9._:/-]*@sha256:[0-9a-f]{64})\Z")
_RUNTIME = "/opt/spark-serve/qwen38-tensorfold"
_ENTRYPOINT = _RUNTIME + "/entrypoint.sh"
_REPOSITORY = "Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP"
_REVISION = "dadefa8066e3be900a0d148d0f5a2f4eb1cf6534"
_MANIFEST_SHA256 = "583d1e6bd1992016e37d9c86c18bbd16a2f08ecca5da221f335e36a7e662f34a"
_SETTINGS = {"model_revision", "parallel", "kv_dtype", "vision", "ple_on_ssd", "mtp_drafts", "mtp_confidence",
             "thinking", "temperature", "top_p", "top_k"}


@dataclass(frozen=True)
class TensorFoldPlan:
    model_id: str
    image: str
    container: str
    host: str
    cache: str
    mount: str
    runtime_cache: str
    argv: tuple[str, ...]


def _text(value, name):
    if not isinstance(value, str) or not value or any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise ControllerError(f"TensorFold {name} must be a nonempty string without control characters")
    return value


def _path(value, name):
    value = _text(value, name)
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts or ":" in value or str(path) == "/":
        raise ControllerError(f"TensorFold {name} must be an absolute non-root POSIX path without '..' or ':'")
    return str(path)


def _integer(value, name, low=1, high=65535):
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ControllerError(f"TensorFold {name} must be an integer in {low}..{high}")
    return value


def _number(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high or not math.isfinite(value):
        raise ControllerError(f"TensorFold {name} must be a finite number in {low}..{high}")
    return value


def validate_tensorfold_plan(cfg: dict, mid: str) -> TensorFoldPlan:
    """Reject unsupported flags and ambiguous checkpoint paths before stopping."""
    cluster, model = cfg["cluster"], cfg["models"][mid]
    if model.get("wrapper") != "tensorfold" or model.get("recipe") != "qwen38-tensorfold":
        raise ControllerError("TensorFold requires wrapper=tensorfold and recipe=qwen38-tensorfold")
    _integer(model.get("nnodes"), "nnodes", 1, 1)
    _integer(model.get("tensor_parallel", 1), "tensor_parallel", 1, 1)
    if model.get("ready_path") != "/health":
        raise ControllerError("TensorFold ready_path must be /health")
    settings = model.get("tensorfold")
    if not isinstance(settings, dict) or set(settings) - _SETTINGS:
        raise ControllerError("TensorFold requires a table containing only supported settings")
    if (model.get("docker_extra") or model.get("mounts") or model.get("env") or model.get("vllm")
            or model.get("skip_default_cache_mount")):
        raise ControllerError("TensorFold uses its validated plan; Docker/mount/environment/vLLM overrides are unsupported")
    image = _text(model.get("image"), "image")
    if not IMMUTABLE_IMAGE.fullmatch(image) and not re.fullmatch(r"[a-z0-9][a-z0-9._/-]*:[A-Za-z0-9_.-]+", image):
        raise ControllerError("TensorFold image must be a local tagged image or immutable digest")
    container = _text(model.get("container"), "container")
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", container):
        raise ControllerError("TensorFold container must be a Docker container name")
    host = _text(cluster.get("head"), "head")
    cache = _path(cluster.get("hf_cache_host"), "hf_cache_host")
    mount = _path(model.get("hf_mount", "/cache/huggingface"), "hf_mount")
    if mount != "/cache/huggingface":
        raise ControllerError("TensorFold hf_mount must be /cache/huggingface")
    repository = _text(model.get("hf_id"), "hf_id")
    if repository != _REPOSITORY:
        raise ControllerError("TensorFold hf_id must match the pinned Vontra checkpoint")
    revision = settings.get("model_revision", "")
    if revision != _REVISION:
        raise ControllerError("TensorFold model_revision must match the prepared recipe revision")
    serve = _path(model.get("serve_path"), "serve_path")
    expected = str(PurePosixPath(mount) / "hub" / ("models--" + repository.replace("/", "--")) / "snapshots" / revision)
    if serve != expected:
        raise ControllerError("TensorFold serve_path must identify the pinned HF snapshot")
    if model.get("preflight_args") != [_RUNTIME + "/verify.py", serve,
                                     "--expected-manifest-sha256", _MANIFEST_SHA256, "--verify-runtime"]:
        raise ControllerError("TensorFold preflight must authenticate its launch checkpoint and runtime")
    parallel = _integer(settings.get("parallel", 4), "parallel", 1, 4)
    if "max_num_seqs" in model and _integer(model["max_num_seqs"], "max_num_seqs", 1, 4) != parallel:
        raise ControllerError("TensorFold max_num_seqs must match parallel")
    context = _integer(model.get("max_model_len"), "max_model_len", 1, 262144)
    kv = settings.get("kv_dtype", "int8")
    if kv != "int8":
        raise ControllerError("this experimental TensorFold recipe requires int8 KV cache")
    for key in ("vision", "ple_on_ssd", "thinking"):
        if key in settings and not isinstance(settings[key], bool):
            raise ControllerError(f"TensorFold {key} must be a boolean")
    if not settings.get("ple_on_ssd", True):
        raise ControllerError("this TensorFold recipe requires ple_on_ssd")
    argv = [serve, "--name", _text(model.get("served_name"), "served_name"),
            "--backend", "cuda", "--tp", "1", "--host", "0.0.0.0", "--port", str(_integer(cluster.get("port", 8000), "port")),
            "--parallel", str(parallel), "--context", str(context), "--kv-dtype", kv,
            "--mtp-drafts", str(_integer(settings.get("mtp_drafts", 6), "mtp_drafts", 0, 8)),
            "--mtp-confidence", str(_number(settings.get("mtp_confidence", 0.60), "mtp_confidence", 0, 1)),
            "--temperature", str(_number(settings.get("temperature", 1.0), "temperature", 0, 2)),
            "--top-p", str(_number(settings.get("top_p", 0.95), "top_p", 0, 1)),
            "--top-k", str(_integer(settings.get("top_k", 20), "top_k", 0)), "--ple-on-ssd",
            "--thinking" if settings.get("thinking", True) else "--no-thinking"]
    if settings.get("vision", True):
        argv.append("--vision")
    return TensorFoldPlan(mid, image, container, host, cache, mount,
                          str(PurePosixPath(cache) / "spark-serve/qwen38-tensorfold/runtime-cache"), tuple(argv))


def tensorfold_docker_argv(plan: TensorFoldPlan, generation: str = "") -> list[str]:
    if generation and not IMMUTABLE_IMAGE.fullmatch(plan.image):
        raise ControllerError("TensorFold startup requires the immutable image verified by preflight")
    command = ["docker", "create", "--pull", "never", "--name", plan.container,
               "--gpus", "all", "--ipc", "host", "--network", "host",
               "--ulimit", "memlock=-1", "--ulimit", "stack=67108864"]
    for key, value in {
        "ai.spark-serve.model": plan.model_id,
        "ai.spark-serve.hosts": json.dumps([plan.host], separators=(",", ":")),
        "ai.spark-serve.allocation": generation,
    }.items():
        command += ["--label", f"{key}={value}"]
    command += ["-v", f"{plan.runtime_cache}:/cache", "-v", f"{plan.cache}:{plan.mount}:ro"]
    for key, value in {
        "HF_HOME": plan.mount, "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1", "HF_TOKEN": "", "TRANSFORMERS_OFFLINE": "1",
        "TORCH_EXTENSIONS_DIR": "/cache/torch_extensions", "TRITON_CACHE_DIR": "/cache/triton",
        "CUDA_CACHE_PATH": "/cache/cuda", "XDG_CACHE_HOME": "/cache/xdg",
        "TENSORFOLD_NO_UPDATE_CHECK": "1", "TENSORFOLD_PREFILL_ROWS": "2048" if "--vision" in plan.argv else "4096",
        "TENSORFOLD_VISION_WORKSPACE_MIB": "0", "TENSORFOLD_MAX_IMAGES": "50", "TENSORFOLD_IMAGE_TOKENS": "16384",
        "TENSORFOLD_VIDEO_TOKENS": "16384", "TENSORFOLD_MTP_COPY": "1",
    }.items():
        command += ["-e", f"{key}={value}"]
    return command + ["--entrypoint", _ENTRYPOINT, plan.image, *plan.argv]


def tensorfold_health_ready(body: str, context: int | None = None) -> bool:
    """The pinned CUDA server publishes health only after engine construction.

    Mia's patch adds backend identity and counters. Busy remains ready: it is
    neither a loading state nor an indication of unavailable model capacity.
    """
    try:
        if not isinstance(body, str) or len(body) > 65536:
            return False
        health = json.loads(body)
        if not isinstance(health, dict) or health.get("ok") is not True or health.get("backend") != "tensorfold":
            return False
        if not isinstance(health.get("busy"), bool):
            return False
        for name in ("requests_running", "prompt_tokens_total", "completion_tokens_total"):
            value = health.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                return False
        seconds = health.get("prefill_seconds_total")
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
            return False
        if context is not None and (type(health.get("context_length")) is not int or health["context_length"] != context):
            return False
        return health["busy"] == (health["requests_running"] > 0)
    except (ValueError, TypeError, OverflowError, RecursionError):
        return False
