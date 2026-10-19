"""Validated TensorFold launch arguments and CUDA health checks."""
from __future__ import annotations

from dataclasses import dataclass
import json
import ipaddress
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
_GLM_RUNTIME = "/opt/spark-serve/glm53-tensorfold"
_GLM_REPOSITORY = "Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold"
_GLM_REVISION = "078455ffe6472f9a52fbc1139f58b9db2881b25c"
_GLM_DRAFT_REVISION = "bf582e4eacc1810f76656d1811693ff6c6737d2a"
_GLM_MANIFEST_SHA256 = "2967c429558c6dfbf69374db14591e07c754f7b32f69b2d4b1e02cf3b8460780"
_GLM_SETTINGS = {"model_revision", "draft_revision", "parallel", "kv_dtype", "dense", "drafter",
                 "vision", "vision_urls", "thinking", "temperature", "top_p", "top_k", "max_tokens",
                 "communication", "master_port"}
_QWEN_DUAL_RUNTIME = "/opt/spark-serve/qwen38-dual-tensorfold"
_QWEN_DUAL_REPOSITORY = "nvidia/Qwen3.8-Flash-Next-NVFP4"
_QWEN_DUAL_REVISION = "fc694b54fb0174e0913e6adf86691ef85a4ead47"
_QWEN_DUAL_MANIFEST_SHA256 = "a3ec3c29bb6f68217a1f109c5ee02b89cf4fdfa1158f4e1626acb2343334d8ad"
_QWEN_DUAL_SETTINGS = {"model_revision", "parallel", "kv_dtype", "drafts", "vision", "vision_urls", "max_images",
                       "max_videos", "thinking", "max_tokens", "roce", "master_port"}
_NATIVE_YARN_CONTEXT = 262144


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
    runtime: str = _RUNTIME
    hosts: tuple[str, ...] = ()
    environment: tuple[tuple[str, str], ...] = ()
    rank: int = 0


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


def validate_tensorfold_plan(cfg: dict, mid: str, rank: int = 0) -> TensorFoldPlan:
    if cfg["models"][mid].get("recipe") == "glm53-tensorfold":
        return _validate_glm_plan(cfg, mid, rank)
    if cfg["models"][mid].get("recipe") == "qwen38-dual-tensorfold":
        return _validate_qwen_dual_plan(cfg, mid, rank)
    _integer(rank, "rank", 0, 0)
    return _validate_qwen_plan(cfg, mid)


def _validate_qwen_plan(cfg: dict, mid: str) -> TensorFoldPlan:
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


def _two_rank_model(cfg: dict, mid: str, rank: int, name: str, allowed: set[str]):
    """Common TP2 identity: image, container, two hosts and this rank's cache."""
    cluster, model = cfg["cluster"], cfg["models"][mid]
    if model.get("wrapper") != "tensorfold":
        raise ControllerError(f"{name} requires wrapper=tensorfold")
    _integer(rank, "rank", 0, 1)
    _integer(model.get("nnodes"), "nnodes", 2, 2)
    _integer(model.get("tensor_parallel"), "tensor_parallel", 2, 2)
    if model.get("ready_path") != "/health":
        raise ControllerError("TensorFold ready_path must be /health")
    settings = model.get("tensorfold")
    if not isinstance(settings, dict) or set(settings) - allowed:
        raise ControllerError(f"{name} requires a table containing only supported settings")
    if (model.get("docker_extra") or model.get("mounts") or model.get("env") or model.get("vllm")
            or model.get("skip_default_cache_mount")):
        raise ControllerError("TensorFold uses its validated plan; Docker/mount/environment/vLLM overrides are unsupported")
    image = _text(model.get("image"), "image")
    if not IMMUTABLE_IMAGE.fullmatch(image) and not re.fullmatch(r"[a-z0-9][a-z0-9._/-]*:[A-Za-z0-9_.-]+", image):
        raise ControllerError("TensorFold image must be a local tagged image or immutable digest")
    container = _text(model.get("container"), "container")
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", container):
        raise ControllerError("TensorFold container must be a Docker container name")
    hosts = tuple(_text(cluster.get(role), role) for role in ("head", "worker"))
    if hosts[0] == hosts[1] or any(h.startswith("-") or any(c.isspace() for c in h) for h in hosts):
        raise ControllerError(f"{name} requires two distinct valid SSH hosts")
    cache = _path((cluster.get("worker_hf_cache_host") or cluster.get("hf_cache_host"))
                  if rank else cluster.get("hf_cache_host"), "hf_cache_host")
    mount = _path(model.get("hf_mount", "/cache/huggingface"), "hf_mount")
    if mount != "/cache/huggingface":
        raise ControllerError("TensorFold hf_mount must be /cache/huggingface")
    return model, settings, image, container, hosts, cache, mount


def _fabric(cluster: dict, settings: dict, name: str) -> tuple[str, int, dict[str, str]]:
    """The head's rendezvous address and port, and the RoCE NCCL environment."""
    master = _text(cluster.get("master_addr"), "master_addr")
    try:
        address = ipaddress.IPv4Address(master)
        if address.is_loopback or address.is_unspecified or address.is_multicast:
            raise ValueError("unreachable rendezvous")
    except ValueError as exc:
        raise ControllerError(f"{name} master_addr must be the head's reachable fabric IPv4 address") from exc
    nccl = cluster.get("nccl") or {}
    interface = _text(nccl.get("NCCL_SOCKET_IFNAME"), "NCCL_SOCKET_IFNAME")
    hcas = _text(nccl.get("NCCL_IB_HCA"), "NCCL_IB_HCA").removeprefix("=")
    gid = _text(nccl.get("NCCL_IB_GID_INDEX"), "NCCL_IB_GID_INDEX")
    if (not re.fullmatch(r"[A-Za-z0-9_.:-]+", interface)
            or not re.fullmatch(r"[A-Za-z0-9_.:-]+(?:,[A-Za-z0-9_.:-]+)*", hcas)
            or not gid.isdigit() or not 0 <= int(gid) <= 255):
        raise ControllerError(f"{name} requires explicit fabric interface, HCAs and RoCE GID index")
    master_port = _integer(settings.get("master_port", 29551), "master_port")
    if master_port == _integer(cluster.get("port", 8000), "port"):
        raise ControllerError("TensorFold master_port must differ from the API port")
    environment = {"NCCL_SOCKET_IFNAME": interface, "NCCL_IB_HCA": hcas, "NCCL_IB_GID_INDEX": gid,
                   "NCCL_MIN_NCHANNELS": "4", "NCCL_MAX_NCHANNELS": "4"}
    if "NCCL_DEBUG" in nccl:
        debug = _text(nccl["NCCL_DEBUG"], "NCCL_DEBUG")
        if debug not in ("VERSION", "WARN", "INFO", "TRACE"):
            raise ControllerError("unsupported TensorFold NCCL_DEBUG")
        environment["NCCL_DEBUG"] = debug
    return master, master_port, environment


def _validate_glm_plan(cfg: dict, mid: str, rank: int) -> TensorFoldPlan:
    cluster = cfg["cluster"]
    model, settings, image, container, hosts, cache, mount = _two_rank_model(
        cfg, mid, rank, "GLM TensorFold", _GLM_SETTINGS)
    if (model.get("hf_id") != _GLM_REPOSITORY or settings.get("model_revision") != _GLM_REVISION
            or settings.get("draft_revision") != _GLM_DRAFT_REVISION):
        raise ControllerError("GLM TensorFold requires its pinned target and draft checkpoints")
    root = str(PurePosixPath(mount) / "spark-serve/glm53-tensorfold")
    serve = root + "/models/target/" + _GLM_REVISION
    if model.get("serve_path") != serve:
        raise ControllerError("GLM TensorFold serve_path must identify its pinned target snapshot")
    if model.get("preflight_args") != [_GLM_RUNTIME + "/verify.py", root,
                                      "--expected-manifest-sha256", _GLM_MANIFEST_SHA256, "--verify-runtime"]:
        raise ControllerError("GLM TensorFold preflight must authenticate both checkpoints and runtime")
    parallel = _integer(settings.get("parallel", 4), "parallel", 1, 8)
    if _integer(model.get("max_num_seqs", parallel), "max_num_seqs", 1, 8) != parallel:
        raise ControllerError("TensorFold max_num_seqs must match parallel")
    context = _integer(model.get("max_model_len"), "max_model_len", 1, 1048576)
    if settings.get("kv_dtype", "fp8") != "fp8" or settings.get("dense", "q4") != "q4":
        raise ControllerError("this GLM TensorFold recipe requires fp8 KV and q4 dense weights")
    drafter = settings.get("drafter", "dflash2")
    if drafter not in ("dflash2", "mtp") or (drafter == "mtp" and parallel != 1):
        raise ControllerError("GLM TensorFold drafter must be dflash2, or mtp with parallel=1")
    communication = settings.get("communication", "roce")
    if communication not in ("roce", "nccl"):
        raise ControllerError("GLM TensorFold communication must be roce or nccl")
    for key in ("vision", "vision_urls", "thinking"):
        if key in settings and not isinstance(settings[key], bool):
            raise ControllerError(f"TensorFold {key} must be a boolean")
    if settings.get("vision_urls", False) and not settings.get("vision", True):
        raise ControllerError("TensorFold vision_urls requires vision")
    master, master_port, fabric = _fabric(cluster, settings, "GLM TensorFold")
    api_port = _integer(cluster.get("port", 8000), "port")
    argv = [serve, "--backend", "cuda", "--tp", "2", "--rank", str(rank),
            "--master", master, "--master-port", str(master_port),
            "--drafter", root + "/models/draft/" + _GLM_DRAFT_REVISION if drafter == "dflash2" else "none",
            "--parallel", str(parallel), "--context", str(context),
            "--max-tokens", str(_integer(settings.get("max_tokens", 32768), "max_tokens", 1, 1048576)),
            "--temperature", str(_number(settings.get("temperature", 1.0), "temperature", 0, 2)),
            "--top-p", str(_number(settings.get("top_p", 0.95), "top_p", 0, 1)),
            "--top-k", str(_integer(settings.get("top_k", 0), "top_k", 0)),
            "--thinking" if settings.get("thinking", True) else "--no-thinking"]
    if rank == 0:
        argv += ["--name", _text(model.get("served_name"), "served_name"), "--host", "0.0.0.0",
                 "--port", str(api_port)]
    if settings.get("vision", True):
        argv.append("--vision")
        if settings.get("vision_urls", False):
            argv.append("--vision-urls")
    window = 64 if parallel > 4 else 32
    environment = {
        **{key: fabric[key] for key in ("NCCL_SOCKET_IFNAME", "NCCL_IB_HCA", "NCCL_IB_GID_INDEX",
                                        "NCCL_MIN_NCHANNELS", "NCCL_MAX_NCHANNELS")},
        "TF_GLM_KV": "fp8", "TF_GLM_DENSE": "q4", "TF_GLM_COMM": communication,
        "TF_GLM_MTP": "auto", "TF_ROCE_MAX_KB": str(window * 16), "TF_ROCE_WAIT_S": "300",
        "TF_GLM_COPY_DRAFTS": "1", "TF_GLM_COPY_MAX": "15", "TF_GLM_DFLASH_POLICY": "fnc7:0.3",
        "TF_GLM_HC_SPLIT": "1", "TF_GLM_PREFILL_OVERLAP": "2", "TF_GLM_KDA_CHUNKED": "1",
        "TF_GLM_WIDE_GRAPHS": "16", "TF_GLM_COPY_REPLY_MATCH": "16", "TF_GLM_MULTI_LONE": "0",
        "TF_GLM_MULTI_WINDOW": str(window), "TF_GLM_CACHE_ENTRIES": "32", "TF_GLM_CLEAR_THINKING": "0",
        "TF_GLM_MULTI_PREFILL": "1", "TF_GLM_STREAM_SMOOTH": "1", "TF_GLM_STREAM_SMOOTH_MS": "400",
        "TF_GLM_FILL_BUDGET_MS": "200", "TF_GLM_FILL_DRAFTS": "1", "TF_GLM_L2PF": "1",
        "TF_GLM_EXL3_LOADS": "nc", "TF_GLM_SHARED_PREFIX": "1", "TF_GLM_CACHE_GIB": "12.5",
        "TF_GLM_DISPLAY_KV_MIB": "0",
        "TENSORFOLD_MEMORY_RESERVE_GIB": str(round(14.5 + max(0, parallel - 4) * 0.95 + (window - 32) * 0.04, 1)),
    }
    if "NCCL_DEBUG" in fabric:
        environment["NCCL_DEBUG"] = fabric["NCCL_DEBUG"]
    return TensorFoldPlan(mid, image, container, hosts[rank], cache, mount,
                          str(PurePosixPath(cache) / "spark-serve/glm53-tensorfold/runtime-cache"),
                          tuple(argv), _GLM_RUNTIME, hosts, tuple(environment.items()), rank)


def native_memory_need_gib(context: int, kv_dtype: str, vision: bool) -> int:
    """Upstream's measured MemAvailable floor for one native rank at start.

    Measured on two Sparks with NVFP4 (2026-10-07): ~48 GiB fixed, KV held from
    the start (9 GiB per million tokens in FP8, 17 in BF16), ~31 GiB per million
    prompt tokens during prefill, 2 GiB vision workspace, the 10 GiB engine
    reserve and 2 GiB of slack.
    """
    kv = 9 if kv_dtype == "fp8" else 17
    return 48 + (2 if vision else 0) + ((kv + 31) * context + 1048575) // 1048576 + 10 + 2


def _validate_qwen_dual_plan(cfg: dict, mid: str, rank: int) -> TensorFoldPlan:
    cluster = cfg["cluster"]
    model, settings, image, container, hosts, cache, mount = _two_rank_model(
        cfg, mid, rank, "Qwen dual TensorFold", _QWEN_DUAL_SETTINGS)
    if model.get("hf_id") != _QWEN_DUAL_REPOSITORY or settings.get("model_revision") != _QWEN_DUAL_REVISION:
        raise ControllerError("Qwen dual TensorFold requires the pinned NVIDIA NVFP4 checkpoint")
    serve = str(PurePosixPath(mount) / "hub" / ("models--" + _QWEN_DUAL_REPOSITORY.replace("/", "--"))
                / "snapshots" / _QWEN_DUAL_REVISION)
    if model.get("serve_path") != serve:
        raise ControllerError("Qwen dual TensorFold serve_path must identify the pinned HF snapshot")
    if model.get("preflight_args") != [_QWEN_DUAL_RUNTIME + "/verify.py", serve, "--expected-manifest-sha256",
                                      _QWEN_DUAL_MANIFEST_SHA256, "--verify-runtime"]:
        raise ControllerError("Qwen dual TensorFold preflight must authenticate its checkpoint and runtime")
    parallel = _integer(settings.get("parallel", 16), "parallel", 1, 16)
    if _integer(model.get("max_num_seqs", parallel), "max_num_seqs", 1, 16) != parallel:
        raise ControllerError("TensorFold max_num_seqs must match parallel")
    context = _integer(model.get("max_model_len"), "max_model_len", 1, 1048576)
    kv = settings.get("kv_dtype", "fp8")
    if kv not in ("fp8", "bf16"):
        raise ControllerError("Qwen dual TensorFold kv_dtype must be fp8 or bf16")
    for key in ("drafts", "vision", "vision_urls", "thinking", "roce"):
        if key in settings and not isinstance(settings[key], bool):
            raise ControllerError(f"TensorFold {key} must be a boolean")
    vision = settings.get("vision", True)
    if settings.get("vision_urls", False) and not vision:
        raise ControllerError("TensorFold vision_urls requires vision")
    master, master_port, fabric = _fabric(cluster, settings, "Qwen dual TensorFold")
    # Both ranks take the same command, as upstream's start.sh runs them; only
    # rank 0 serves HTTP.
    argv = [serve, "--name", _text(model.get("served_name"), "served_name"), "--host", "0.0.0.0",
            "--port", str(_integer(cluster.get("port", 8000), "port")),
            "--context", str(context), "--parallel", str(parallel),
            "--max-tokens", str(_integer(settings.get("max_tokens", 32768), "max_tokens", 1, 1048576)),
            "--thinking" if settings.get("thinking", True) else "--no-thinking"]
    if not settings.get("drafts", True):
        argv.append("--no-drafts")
    argv += ["--kv-dtype", kv]
    images = _integer(settings.get("max_images", 50), "max_images", 1, 50)
    if vision:
        argv += ["--vision", "--vision-max-images", str(images),
                 "--vision-max-videos", str(_integer(settings.get("max_videos", 4), "max_videos", 0, 16)),
                 "--vision-image-tokens", "16384"]
        if settings.get("vision_urls", False):
            argv.append("--vision-urls")
    argv += ["--tp", "2", "--rank", str(rank), "--master", master, "--master-port", str(master_port)]
    environment = {
        **fabric,
        "TENSORFOLD_CUDA_KERNELS": "/opt/tensorfold/share/tensorfold/cuda/sm121",
        # Qwen's documented YaRN (factor 4) past the native 262,144 window.
        "TF_FLASHNEXT_YARN": "4" if context > _NATIVE_YARN_CONTEXT else "0",
        "TF_FLASHNEXT_ROCE": "1" if settings.get("roce", True) else "0",
        "TF_FLASHNEXT_L2PF": "1", "TF_FLASHNEXT_DEPTH": "15",
        "TENSORFOLD_MAX_IMAGES": str(images), "TENSORFOLD_IMAGE_TOKENS": "16384",
        "TENSORFOLD_VIDEO_TOKENS": "16384", "TENSORFOLD_VISION_WORKSPACE_MIB": "2048",
        "TENSORFOLD_MEMORY_RESERVE_GIB": "10",
        "SPARK_SERVE_MIN_AVAILABLE_GIB": str(native_memory_need_gib(context, kv, vision)),
    }
    return TensorFoldPlan(mid, image, container, hosts[rank], cache, mount,
                          str(PurePosixPath(cache) / "spark-serve/qwen38-dual-tensorfold/runtime-cache"),
                          tuple(argv), _QWEN_DUAL_RUNTIME, hosts, tuple(environment.items()), rank)


def tensorfold_docker_argv(plan: TensorFoldPlan, generation: str = "") -> list[str]:
    if generation and not IMMUTABLE_IMAGE.fullmatch(plan.image):
        raise ControllerError("TensorFold startup requires the immutable image verified by preflight")
    command = ["docker", "create", "--pull", "never", "--name", plan.container,
               "--gpus", "all", "--ipc", "host", "--network", "host",
               "--ulimit", "memlock=-1", "--ulimit", "stack=67108864"]
    for key, value in {
        "ai.spark-serve.model": plan.model_id,
        "ai.spark-serve.hosts": json.dumps(plan.hosts or (plan.host,), separators=(",", ":")),
        "ai.spark-serve.allocation": generation,
    }.items():
        command += ["--label", f"{key}={value}"]
    command += ["-v", f"{plan.runtime_cache}:/cache", "-v", f"{plan.cache}:{plan.mount}:ro"]
    environment = {
        "HF_HOME": plan.mount, "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1", "HF_TOKEN": "", "TRANSFORMERS_OFFLINE": "1",
        "TORCH_EXTENSIONS_DIR": "/cache/torch_extensions", "TRITON_CACHE_DIR": "/cache/triton",
        "CUDA_CACHE_PATH": "/cache/cuda", "XDG_CACHE_HOME": "/cache/xdg",
        "TENSORFOLD_NO_UPDATE_CHECK": "1", "TENSORFOLD_PREFILL_ROWS": "2048" if "--vision" in plan.argv else "4096",
        "TENSORFOLD_VISION_WORKSPACE_MIB": "0", "TENSORFOLD_MAX_IMAGES": "50", "TENSORFOLD_IMAGE_TOKENS": "16384",
        "TENSORFOLD_VIDEO_TOKENS": "16384", "TENSORFOLD_MTP_COPY": "1",
    }
    if plan.runtime in (_GLM_RUNTIME, _QWEN_DUAL_RUNTIME):
        command += ["--shm-size", "16g", "--device", "/dev/infiniband", "--cap-add", "IPC_LOCK"]
        environment = {key: value for key, value in environment.items()
                       if not key.startswith("TENSORFOLD_") or key == "TENSORFOLD_NO_UPDATE_CHECK"}
        environment.update(plan.environment)
    for key, value in environment.items():
        command += ["-e", f"{key}={value}"]
    return command + ["--entrypoint", plan.runtime + "/entrypoint.sh", plan.image, *plan.argv]


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


def tensorfold_native_health_ready(body: str, served_name: str) -> bool:
    """The Zig server's /health: it listens only after both ranks load.

    Ready means the configured model is served and graph warming has ended.
    The schema has no context or counters; /metrics carries those.
    """
    try:
        if not isinstance(body, str) or len(body) > 65536:
            return False
        health = json.loads(body)
        if not isinstance(health, dict) or health.get("status") != "ok" or health.get("model") != served_name:
            return False
        lanes = health.get("max_batch_size")
        return health.get("warming") is False and type(lanes) is int and lanes > 0
    except (ValueError, TypeError, OverflowError, RecursionError):
        return False


def tensorfold_model_ready(body: str, model: dict) -> bool:
    """Dispatch on the catalog recipe: the native engine has its own schema."""
    if model.get("recipe") == "qwen38-dual-tensorfold":
        return tensorfold_native_health_ready(body, model.get("served_name"))
    return tensorfold_health_ready(body, model.get("max_model_len"))
