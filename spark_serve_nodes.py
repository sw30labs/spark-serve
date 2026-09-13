"""Resolve physical node placement without changing the shared cluster catalog."""
from __future__ import annotations

import copy
from urllib.parse import urlsplit, urlunsplit

from spark_serve_controller import ControllerError


def node_url(cfg: dict, node: str) -> str:
    if node not in ("head", "worker"):
        raise ControllerError("node must be head or worker")
    cluster = cfg["cluster"]
    if node == "head":
        url = cluster["lan_url"]
    else:
        url = cluster.get("worker_lan_url")
        if not url:
            # Existing catalogs already distinguish the HTTP hostname from an
            # SSH alias for YuE. Reuse only its hostname, never its service port.
            worker = urlsplit(str(cfg.get("yue", {}).get("worker_url", "")))
            hostname = worker.hostname or str(cluster["worker"])
            authority = f"[{hostname}]" if ":" in hostname else hostname
            url = f"http://{authority}:{int(cluster.get('port') or 8000)}"
    try:
        parsed = urlsplit(str(url))
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError as exc:
        raise ControllerError(f"invalid {node} serving URL") from exc
    if (parsed.scheme not in ("http", "https") or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path not in ("", "/")):
        raise ControllerError(f"{node} serving URL must be an HTTP(S) origin without credentials")
    if port != int(cluster.get("port") or 8000):
        raise ControllerError(f"{node} serving URL port must match cluster.port")
    if node == "worker":
        head = urlsplit(node_url(cfg, "head"))
        head_port = head.port or (443 if head.scheme == "https" else 80)
        if (parsed.hostname.lower(), port) == (head.hostname.lower(), head_port):
            raise ControllerError("head and worker serving URLs must identify different Sparks")
    return urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))


def placement(cfg: dict, model: dict | None, node: str | None = None) -> tuple[str, list[str]]:
    """Resolve a model's required nodes; a solo recipe never becomes TP2."""
    cluster = cfg["cluster"]
    if cluster["head"] == cluster["worker"]:
        raise ControllerError("independent Sparks require two distinct SSH hosts")
    nnodes = 2 if model is None else int(model.get("nnodes") or cluster.get("nnodes") or 2)
    if nnodes not in (1, 2):
        raise ControllerError("Spark Serve supports recipes using one or two nodes")
    selected = node or ("head" if nnodes == 1 else "both")
    if selected not in ("head", "worker", "both"):
        raise ControllerError("node must be head, worker, or both")
    if nnodes == 2 and selected != "both":
        raise ControllerError("this recipe needs both Sparks; use --node both")
    if nnodes == 1 and selected == "both":
        raise ControllerError("choose --node head or --node worker for a single-Spark recipe")
    roles = ("head", "worker") if selected == "both" else (selected,)
    return selected, [str(cluster[role]) for role in roles]


def launch_config(cfg: dict, mid: str, node: str | None = None) -> dict:
    """Remap a solo rank-zero launch to its physical node, including mounts."""
    selected, hosts = placement(cfg, cfg["models"][mid], node)
    result = copy.deepcopy(cfg)
    result["_spark_serve_placement"] = {"node": selected, "hosts": hosts, "model": mid}
    cluster, model = result["cluster"], result["models"][mid]
    if selected == "worker":
        original = cfg["cluster"]
        cluster["head"], cluster["worker"] = original["worker"], original["head"]
        cluster["lan_url"] = node_url(cfg, "worker")
        cluster["hf_cache_host"] = original.get("worker_hf_cache_host") or original["hf_cache_host"]
        # A local single-GPU process must not rendezvous at the other Spark.
        cluster["master_addr"] = str(original.get("worker_master_addr") or "127.0.0.1")
        model["master_addr"] = cluster["master_addr"]
        for mount in model.get("mounts") or []:
            if not isinstance(mount, dict):
                continue
            if mount.get("head") and not mount.get("worker"):
                raise ControllerError(f"{mid}: a worker path is required for mount {mount.get('container')}")
            mount["head"], mount["worker"] = mount.get("worker"), mount.get("head")
        worker_ip = model.get("host_ip_worker")
        model.pop("host_ip_head", None)
        if worker_ip:
            model["host_ip_head"] = worker_ip
        # Keep each provider's endpoint stable when another node is selected.
        model["hermes_provider"] = str(model.get("hermes_provider") or "spark") + "-worker"
    elif selected == "head":
        cluster["lan_url"] = node_url(cfg, "head")
    return result
