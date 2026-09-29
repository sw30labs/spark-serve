"""Offline integrity, resume and node-placement checks for the NGC GLM recipe."""
import hashlib
import importlib.util
import io
import json
import os
import shlex
import subprocess
import sys
import tarfile
from unittest.mock import patch

import pytest

from tools import prepare_glm53 as prepare


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


verify = module("glm53_checkpoint_verify", prepare.RECIPE / "verify.py")
with patch.dict(sys.modules, {"verify": verify}):
    download = module("glm53_checkpoint_download", prepare.RECIPE / "download.py")


@pytest.fixture
def snapshot(tmp_path):
    root = tmp_path / "cache with spaces" / "models" / "nim-test-nvfp4"
    root.mkdir(parents=True)
    contents = {
        "config.json": b'{"architectures":["Glm5NextForConditionalGeneration"]}',
        "model.safetensors.index.json": b'{"weight_map":{"weight":"weights.safetensors"}}',
        "tokenizer_config.json": b"{}", "generation_config.json": b"{}",
        "tokenizer.json": b'{"version":"1.0"}', "processor_config.json": b"{}",
        "chat_template.jinja": b"original", "weights.safetensors": b"correct weights",
    }
    manifest = {"model": "nim/zai-org/glm-5.3-flash", "revision": root.name,
                "architectures": ["Glm5NextForConditionalGeneration"],
                "download_base_url": "https://api.ngc.nvidia.com/v2/models/nim/zai-org/glm-5.3-flash/versions/" + root.name + "/files",
                "files": []}
    for name, data in contents.items():
        (root / name).write_bytes(data)
        manifest["files"].append({"path": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    return root, manifest, contents


def test_complete_snapshot_is_authenticated_without_model_imports(snapshot):
    root, manifest, contents = snapshot
    report = verify.verify(root, manifest, full_hash=True)
    assert report["bytes"] == sum(map(len, contents.values()))
    assert report["full_hash_verified"] is True
    assert report["architectures"] == ["Glm5NextForConditionalGeneration"]


@pytest.mark.parametrize("filename", ["chat_template.jinja", "tokenizer.json", "config.json", "model.safetensors.index.json"])
def test_same_size_metadata_corruption_is_caught_at_startup(snapshot, filename):
    root, manifest, contents = snapshot
    (root / filename).write_bytes(b"x" * len(contents[filename]))
    with pytest.raises(verify.CheckpointFileError) as failure:
        verify.verify(root, manifest)
    assert failure.value.filename == filename


def test_startup_checks_large_shard_size_and_setup_checks_its_content(snapshot, monkeypatch):
    root, manifest, contents = snapshot
    item = next(item for item in manifest["files"] if item["path"] == "weights.safetensors")
    monkeypatch.setattr(verify, "SMALL_FILE_LIMIT", 1)
    (root / item["path"]).write_bytes(b"x" * len(contents[item["path"]]))
    verify.verify_file(root, item, full_hash=False)
    with pytest.raises(verify.CheckpointFileError, match="SHA-256"):
        verify.verify_file(root, item, full_hash=True)
    (root / item["path"]).write_bytes(b"short")
    with pytest.raises(verify.CheckpointFileError, match="incomplete"):
        verify.verify_file(root, item, full_hash=False)


def test_pins_architecture_and_index_membership(snapshot):
    root, manifest, _ = snapshot
    manifest["architectures"] = ["WrongModel"]
    with pytest.raises(ValueError, match="architecture"):
        verify.verify(root, manifest)
    manifest["architectures"] = ["Glm5NextForConditionalGeneration"]
    manifest["revision"] = "wrong-version"
    with pytest.raises(ValueError, match="pinned checkpoint"):
        verify.verify(root, manifest)
    manifest["revision"] = root.name
    changed = b'{"weight_map":{"weight":"unlisted.safetensors"}}'
    (root / "model.safetensors.index.json").write_bytes(changed)
    item = next(item for item in manifest["files"] if item["path"] == "model.safetensors.index.json")
    item.update(size=len(changed), sha256=hashlib.sha256(changed).hexdigest())
    with pytest.raises(ValueError, match="unverified weight"):
        verify.verify(root, manifest)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a/../escape", "", "./config.json"])
def test_manifest_rejects_unsafe_paths(snapshot, name):
    root, manifest, _ = snapshot
    manifest["files"][0]["path"] = name
    with pytest.raises(ValueError, match="path"):
        verify.verify(root, manifest)


def test_symlink_outside_snapshot_is_rejected(snapshot, tmp_path):
    root, manifest, contents = snapshot
    path = root / "weights.safetensors"
    outside = tmp_path / "outside"
    outside.write_bytes(contents[path.name])
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        verify.verify(root, manifest)


class Response(io.BytesIO):
    def __init__(self, data, status=200, content_range=None):
        super().__init__(data)
        self.status = status
        self.headers = {"Content-Range": content_range} if content_range else {}


def file_case(tmp_path):
    data = b"abcdef"
    item = {"path": "weights.safetensors", "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    return item, data, tmp_path / item["path"]


def test_download_resumes_authenticated_partial_without_restarting(tmp_path):
    item, data, target = file_case(tmp_path)
    partial = target.with_name(target.name + ".incomplete")
    partial.write_bytes(data[:3])
    requests = []
    def opener(request, **kwargs):
        requests.append(request)
        return Response(data[3:], 206, "bytes 3-5/6")
    download.download_file(tmp_path, item, "https://example.test/files", opener=opener)
    assert requests[0].headers["Range"] == "bytes=3-"
    assert target.read_bytes() == data and not partial.exists()


def test_ignored_range_replaces_partial_instead_of_appending(tmp_path):
    item, data, target = file_case(tmp_path)
    target.with_name(target.name + ".incomplete").write_bytes(data[:3])
    download.download_file(tmp_path, item, "https://example.test/files", opener=lambda *a, **k: Response(data))
    assert target.read_bytes() == data


def test_bad_resume_range_never_publishes(tmp_path):
    item, data, target = file_case(tmp_path)
    target.with_name(target.name + ".incomplete").write_bytes(data[:3])
    with pytest.raises(ValueError, match="resume range"):
        download.download_file(tmp_path, item, "https://example.test/files", attempts=1,
                               opener=lambda *a, **k: Response(data[3:], 206, "bytes 0-2/6"))
    assert not target.exists()


def test_interrupted_transfer_retries_from_durable_partial(tmp_path):
    item, data, target = file_case(tmp_path)
    class Interrupted(Response):
        def read(self, size=-1):
            if self.tell() == 0:
                return super().read(3)
            raise ConnectionResetError("simulated connection loss")
    requests = []
    def opener(request, **kwargs):
        requests.append(request)
        return Interrupted(data) if len(requests) == 1 else Response(data[3:], 206, "bytes 3-5/6")
    download.download_file(tmp_path, item, "https://example.test/files", opener=opener, pause=lambda _: None)
    assert requests[1].headers["Range"] == "bytes=3-"
    assert target.read_bytes() == data


def test_checksum_failure_retains_old_file_and_repairs_with_fresh_download(tmp_path):
    item, data, target = file_case(tmp_path)
    target.write_bytes(b"xxxxxx")
    responses = iter([b"badbad", data])
    download.download_file(tmp_path, item, "https://example.test/files", pause=lambda _: None,
                           opener=lambda *a, **k: Response(next(responses)))
    assert target.read_bytes() == data


def test_existing_verified_download_makes_no_http_request(tmp_path):
    item, data, target = file_case(tmp_path)
    target.write_bytes(data)
    def forbidden(*args, **kwargs):
        raise AssertionError("existing verified files must be reused")
    download.download_file(tmp_path, item, "https://example.test/files", opener=forbidden)


def test_download_rejects_unpinned_or_non_ngc_origin(snapshot):
    root, manifest, _ = snapshot
    manifest["download_base_url"] = "https://example.test/files"
    with pytest.raises(ValueError, match="pinned public NVIDIA"):
        download.download(root, manifest)


def write_catalog(tmp_path):
    image = json.loads((prepare.RECIPE / "source-pins.json").read_text())["image"]
    path = tmp_path / "models.toml"
    path.write_text('[cluster]\nhead="head-ssh"\nworker="worker-ssh"\n'
                    'hf_cache_host="/head/cache"\nworker_hf_cache_host="/worker/cache with spaces"\n'
                    '[models.glm53]\nnnodes=2\nrecipe="glm53-nvfp4"\nimage=' + json.dumps(image) + '\n')
    return path


@pytest.mark.parametrize("node,hosts", [("head", ["head-ssh"]), ("worker", ["worker-ssh"]), ("both", ["head-ssh", "worker-ssh"])])
@pytest.mark.parametrize("skip_download", [False, True])
def test_preparation_targets_configured_nodes_and_caches_without_workload_changes(tmp_path, monkeypatch, node, hosts, skip_download):
    catalog = write_catalog(tmp_path)
    monkeypatch.setattr(sys, "argv", ["prepare", "--catalog", str(catalog), "--node", node]
                        + (["--skip-download"] if skip_download else []))
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="/tmp/spark-serve-glm53.12345678\n")
    monkeypatch.setattr(prepare.subprocess, "run", run)
    prepare.main()
    scripts = [(command, options["input"]) for command, options in calls if "input" in options]
    assert len(scripts) == len(hosts)
    for host, (command, script) in zip(hosts, scripts):
        assert host in command
        cache = "/head/cache" if host == "head-ssh" else "/worker/cache with spaces"
        assert shlex.quote(cache + "/spark-serve/glm53-nvfp4/models/nim-aa28e1f-nvfp4") in script
        assert "docker pull nvcr.io/nim/zai-org/glm-5.3-flash@sha256:" in script
        assert all(token not in script for token in ("docker run", "docker stop", "docker rm", "systemctl", "--gpus", "pip install"))
        assert ("download.py" in script) is not skip_download
        assert ("--full-hash" in script) is skip_download


def test_mutable_or_wrong_image_is_rejected_before_ssh(tmp_path, monkeypatch):
    catalog = write_catalog(tmp_path)
    text = catalog.read_text()
    image = json.loads((prepare.RECIPE / "source-pins.json").read_text())["image"]
    catalog.write_text(text.replace(image, "nvcr.io/nim/zai-org/glm-5.3-flash:latest"))
    monkeypatch.setattr(sys, "argv", ["prepare", "--catalog", str(catalog)])
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid recipe must fail before SSH")
    monkeypatch.setattr(prepare.subprocess, "run", forbidden)
    with pytest.raises(ValueError, match="pinned NVIDIA"):
        prepare.main()


def test_published_manifest_is_complete_and_authenticated():
    raw = (prepare.RECIPE / "model-source.json").read_bytes()
    source = json.loads(raw)
    pins = json.loads((prepare.RECIPE / "source-pins.json").read_text())
    files = verify.manifest_files(source)
    assert len(files) == len({item["path"] for item in files}) == 130
    assert sum(item["size"] for item in files) == 194692707910
    assert sum(item["path"].endswith(".safetensors") for item in files) == 120
    assert hashlib.sha256(raw).hexdigest() == pins["manifest_sha256"]
    assert source["revision"] == pins["checkpoint_revision"] == "nim-aa28e1f-nvfp4"


def test_preflight_accepts_controller_manifest_pin(snapshot, tmp_path, monkeypatch, capsys):
    root, manifest, _ = snapshot
    path = tmp_path / "manifest.json"
    raw = json.dumps(manifest).encode()
    path.write_bytes(raw)
    expected = hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(sys, "argv", ["verify", str(root), "--manifest", str(path),
                                    "--expected-manifest-sha256", expected])
    verify.main()
    report = json.loads(capsys.readouterr().out)
    assert report["manifest_sha256"] == expected
    assert report["full_hash_verified"] is False


@pytest.mark.parametrize("expected", ["0" * 64, "invalid", "A" * 64])
def test_manifest_pin_failure_precedes_asset_verification(snapshot, tmp_path, monkeypatch, capsys, expected):
    root, manifest, _ = snapshot
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(sys, "argv", ["verify", str(root), "--manifest", str(path),
                                    "--expected-manifest-sha256", expected])
    def forbidden(*args, **kwargs):
        raise AssertionError("mismatched manifest must be rejected before touching checkpoint assets")
    monkeypatch.setattr(verify, "verify", forbidden)
    with pytest.raises(SystemExit) as failure:
        verify.main()
    assert failure.value.code == 1
    assert "manifest SHA-256" in capsys.readouterr().err


def derivative_pins(recipe=prepare.RECIPE):
    return {"image": "sha256:" + "1" * 64,
            "base_image": "nvcr.io/nim/zai-org/glm-5.3-flash@sha256:" + "2" * 64,
            "checkpoint_revision": "nim-aa28e1f-nvfp4",
            "image_patch": {"revision": "marlin-gate-up-maxglobal-e4m3-v1",
                            "files": {name: hashlib.sha256((recipe / name).read_bytes()).hexdigest()
                                      for name in prepare.DERIVATIVE_FILES}}}


def test_bootstrap_builds_accept_independent_context_and_iid_paths():
    commands = [prepare.derivative_build_command(context=f"/tmp/build-{node}",
                iidfile=f"/tmp/build-{node}/image-build.iid") for node in ("one", "two")]
    assert commands[0][commands[0].index("--iidfile") + 1] != commands[1][commands[1].index("--iidfile") + 1]
    for node, command in zip(("one", "two"), commands):
        assert command[-1] == f"/tmp/build-{node}"
        assert command[command.index("--iidfile") + 1] == f"/tmp/build-{node}/image-build.iid"
        assert command[command.index("--builder") + 1] == "default"


def test_derivative_recipe_requires_both_catalog_image_id_and_immutable_base():
    pins = derivative_pins()
    model = {"recipe": "glm53-nvfp4", "nnodes": 2, "image": pins["image"]}
    cfg = {"models": {"glm53": model}}
    source = {"revision": pins["checkpoint_revision"]}
    assert prepare.validate_recipe(cfg, source, pins) is model
    model["image"] = "sha256:" + "3" * 64
    with pytest.raises(ValueError, match="pinned NVIDIA"):
        prepare.validate_recipe(cfg, source, pins)
    model["image"] = pins["image"]
    pins["base_image"] = "nvcr.io/nim/zai-org/glm-5.3-flash:latest"
    with pytest.raises(ValueError, match="base_image"):
        prepare.validate_recipe(cfg, source, pins)


def test_build_files_are_authenticated_before_remote_work(tmp_path):
    pins = derivative_pins()
    pins["base_image"] = (prepare.RECIPE / "Dockerfile").read_text().splitlines()[0].removeprefix("FROM ")
    assert prepare.validate_image_patch(pins) == pins["image_patch"]
    for name in prepare.DERIVATIVE_FILES:
        (tmp_path / name).write_bytes((prepare.RECIPE / name).read_bytes())
    (tmp_path / "patch_runtime.py").write_text("unreviewed change")
    with pytest.raises(ValueError, match="SHA-256 mismatch: patch_runtime.py"):
        prepare.validate_image_patch(pins, tmp_path)
    pins["image_patch"]["files"].pop("Dockerfile")
    with pytest.raises(ValueError, match="all build-file hashes"):
        prepare.validate_image_patch(pins, tmp_path)


def test_payload_is_reproducible_despite_local_file_metadata(tmp_path):
    pins = derivative_pins()
    names = (*prepare.DERIVATIVE_FILES, "verify.py", "download.py", "model-source.json", "source-pins.json")
    for name in names:
        (tmp_path / name).write_bytes((prepare.RECIPE / name).read_bytes())
    first = io.BytesIO()
    prepare.write_payload(first, pins, tmp_path)
    for i, name in enumerate(names):
        os.utime(tmp_path / name, (1900000000 + i, 1900000000 + i))
        (tmp_path / name).chmod(0o755)
    second = io.BytesIO()
    prepare.write_payload(second, pins, tmp_path)
    assert first.getvalue() == second.getvalue()
    with tarfile.open(fileobj=io.BytesIO(first.getvalue())) as archive:
        members = archive.getmembers()
        assert len(members) == 8
        assert all(m.mtime == m.uid == m.gid == 0 and m.uname == m.gname == "" for m in members)
        assert all(m.mode == (0o755 if m.isdir() else 0o644) for m in members)
        assert set(m.name for m in members if m.name.startswith("image-context/")) == {
            "image-context/" + name for name in prepare.DERIVATIVE_FILES}


@pytest.mark.parametrize("fault", [None, "built_id", "inspected_id", "architecture", "user"])
def test_derivative_preparation_checks_built_identity_before_publishing(tmp_path, fault):
    remote = tmp_path / "remote staging"
    remote.mkdir()
    cache = tmp_path / "model cache"
    root = cache / prepare.RELATIVE_ROOT
    root.mkdir(parents=True)
    previous = b'{"previous":"verified preparation"}\n'
    (root / "prepared.json").write_bytes(previous)
    for name in ("verify.py", "download.py"):
        (remote / name).write_text("pass\n")
    source = {"revision": "nim-aa28e1f-nvfp4", "model": "nim/zai-org/glm-5.3-flash",
              "files": [{"path": "weights.safetensors", "size": 16}]}
    (remote / "model-source.json").write_text(json.dumps(source))
    pins = derivative_pins()
    (remote / "source-pins.json").write_text(json.dumps(pins))
    binary = tmp_path / "bin"
    binary.mkdir()
    docker = binary / "docker"
    docker.write_text("#!" + sys.executable + "\n" + '''import json, os, pathlib, sys
args = sys.argv[1:]
fault = os.environ.get('GLM_TEST_FAULT')
expected = 'sha256:' + '1' * 64
wrong = 'sha256:' + 'f' * 64
if args[:2] == ['buildx', 'build']:
    assert args[args.index('--network') + 1] == 'none'
    assert '--pull=false' in args and 'SOURCE_DATE_EPOCH=0' in args
    assert not any('gpu' in arg for arg in args)
    pathlib.Path(args[args.index('--iidfile') + 1]).write_text(wrong if fault == 'built_id' else expected)
elif args[:2] == ['image', 'inspect']:
    print(json.dumps([{'Id': wrong if fault == 'inspected_id' else expected,
                      'Architecture': 'amd64' if fault == 'architecture' else 'arm64', 'Os': 'linux',
                      'Config': {'User': 'root' if fault == 'user' else 'nvs:1000'}}]))
elif args[0] != 'pull':
    raise SystemExit('Unexpected Docker action: ' + repr(args))
''')
    docker.chmod(0o755)
    script = prepare.preparation_script(str(remote), str(cache), {"image": pins["image"]}, source,
                                        pins=pins, skip_download=True)
    env = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ["PATH"], GLM_TEST_FAULT=fault or "")
    result = subprocess.run(["bash", "-s"], input=script, text=True, capture_output=True, env=env)
    if fault:
        assert result.returncode != 0
        assert (root / "prepared.json").read_bytes() == previous
        assert not (root / "source-pins.json").exists()
    else:
        assert result.returncode == 0, result.stderr
        receipt = json.loads((root / "prepared.json").read_text())
        assert receipt["image_id"] == pins["image"]
        assert receipt["base_image"] == pins["base_image"]
        assert receipt["image_patch"] == pins["image_patch"]
        assert receipt["full_hash_verified"] is True
        assert (root / "runtime-cache").is_dir()
