#!/bin/bash
set -euo pipefail
runtime=/opt/spark-serve/glm53-tensorfold
python3 "$runtime/verify_runtime.py"
if [ "${1:-}" = "--verify-runtime" ]; then
    exit 0
fi
# Retain NVIDIA's entrypoint and license notices from the pinned upstream image.
exec /opt/nvidia/nvidia_entrypoint.sh tensorfold serve "$@"
