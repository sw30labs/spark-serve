FROM nvcr.io/nim/zai-org/glm-5.3-flash@sha256:0bd2a1f4ffacf4ee61e1f8c1b48071a6713b51bad954ab0ca627c24be84f43f8

# Use the bundled Dockerfile frontend and one CPU-only changed layer. Binding
# the context to the existing /tmp directory avoids a separate COPY layer or
# a newly created mount-point directory with a wall-clock modification time.
USER root
RUN --mount=type=bind,target=/tmp,readonly python3 -B -c "import os, pathlib, runpy, shutil; source = pathlib.Path('/tmp'); target = pathlib.Path('/opt/spark-serve/glm53'); target.mkdir(parents=True, exist_ok=True); [shutil.copyfile(source / name, target / name) for name in ('patch_runtime.py', 'nvfp4_scale_reconcile.py')]; runpy.run_path(str(target / 'patch_runtime.py'), run_name='__main__'); paths = [*target.rglob('*'), target, target.parent, pathlib.Path('/opt')]; [(os.chmod(path, 0o755 if path.is_dir() else 0o644), os.utime(path, (0, 0))) for path in paths]"
LABEL ai.spark-serve.glm53.patch="marlin-gate-up-maxglobal-e4m3-v1"
USER nvs:1000
