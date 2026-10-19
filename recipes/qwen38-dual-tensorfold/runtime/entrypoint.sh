#!/bin/bash
set -euo pipefail
runtime=/opt/spark-serve/qwen38-dual-tensorfold
python3 "$runtime/verify_runtime.py"
if [ "${1:-}" = "--verify-runtime" ]; then
    exit 0
fi
# GB10 unified memory freezes the host instead of failing an allocation, so a
# rank starts only with the MemAvailable upstream measured for this window.
need="${SPARK_SERVE_MIN_AVAILABLE_GIB:?}"
have=$(awk '/^MemAvailable:/ {print int($2 / 1048576)}' /proc/meminfo)
if [ "$have" -lt "$need" ]; then
    echo "spark-serve: ${have} GiB available, this rank needs ${need} GiB; stop other GPU work or lower the context" >&2
    exit 75
fi
# Retain NVIDIA's entrypoint and license notices from the pinned upstream image.
exec /opt/nvidia/nvidia_entrypoint.sh /opt/tensorfold/native/bin/tensorfold-native serve "$@"
