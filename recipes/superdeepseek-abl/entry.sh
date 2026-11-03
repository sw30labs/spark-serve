#!/usr/bin/env bash
# SuperDeepseek V4 Flash abliterated MQ: in-container entry for spark-serve.
# Runs INSIDE the container as
#   bash -c "$(cat entry.sh)" spark-serve-entry <model_dir> <vllm serve args...>
# spark-serve embeds this reviewed file in the container command (docker inspect
# shows it), so the hosts need no launcher files. Vendored from the author's
# repro/scripts/run_superdeepseek_v4_vllm.sh @2a7dd6a1 through the reviewed
# ~/superdeepseek-abl/container-entry.sh on sparkone (sha256 8015f5b4...).
# Changes vs that file: the model dir and every vLLM flag come from argv
# (models.toml [models.ds4-super-abl.vllm].args + spark-serve rank flags) instead
# of env; nothing else. Same tokenizer copy and reasoning-effort patch.
# VLLM_PKG / VLLM_BIN exist only so tests can run this against a fake tree.
set -euo pipefail

export PATH="/usr/local/cuda/bin:/usr/local/bin:${PATH:-}"
export CUDA_HOME=/usr/local/cuda
export CUDA_PATH="$CUDA_HOME"
export CUDAToolkit_ROOT="$CUDA_HOME"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

vllm_pkg="${VLLM_PKG:-/usr/local/lib/python3.12/dist-packages/vllm}"
vllm_bin="${VLLM_BIN:-/usr/local/bin/vllm}"
model_dir="${1:?usage: entry.sh <model_dir> <vllm serve args...>}"

# 1) The model's encoding_dsv4.py is byte-identical to the official DeepSeek-V4-Flash-0731
#    file (same git blob). It replaces vLLM's bundled copy in this container's layer only.
cp "$model_dir/encoding/encoding_dsv4.py" "$vllm_pkg/tokenizers/deepseek_v4_encoding.py"

# 2) Reasoning-effort mapping: unknown/"low" efforts map to "low" instead of "high".
#    Exact-string patch; fails closed if vLLM's source differs from what the author patched.
python3 - "$vllm_pkg/tokenizers/deepseek_v4.py" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
source = path.read_text()
old = """elif reasoning_effort in ("max", "xhigh"):
                reasoning_effort = "max"
            else:
                reasoning_effort = "high"""
new = """elif reasoning_effort in ("max", "xhigh"):
                reasoning_effort = "max"
            elif reasoning_effort == "high":
                reasoning_effort = "high"
            else:
                reasoning_effort = "low"""
if new not in source:
    if source.count(old) != 1:
        raise RuntimeError("DeepSeek V4 reasoning-effort patch target changed")
    path.write_text(source.replace(old, new))
PY

echo "spark-serve-entry: SuperDeepseek tokenizer patch applied" >&2
exec "$vllm_bin" serve "$@"
