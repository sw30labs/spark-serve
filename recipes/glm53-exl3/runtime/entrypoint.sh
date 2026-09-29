#!/bin/bash
set -euo pipefail

# Serving arguments, topology, transport, and persistent caches are supplied by
# the normal spark-serve catalog/controller. Keep optional precision changes off.
export ABLIT="${ABLIT:-0}"
export GLM53_DENSE_FP8="${GLM53_DENSE_FP8:-off}"
export GLM53_ADAPTIVE_K="${GLM53_ADAPTIVE_K:-off}"
export GLM53_EXL3_MOE_FAST="${GLM53_EXL3_MOE_FAST:-0}"
export GLM53_KDA_BF16_LARGE_M="${GLM53_KDA_BF16_LARGE_M:-0}"

runtime=/opt/spark-serve/glm53-exl3
python3 "$runtime/verify_runtime.py"
while IFS= read -r patch; do
    [ -n "$patch" ] || continue
    python3 "/opt/glm53/$patch"
done < "$runtime/overlay-order.txt"

# Preparation can exercise every source migration without GPU/model access.
if [ "${1:-}" = "--patches-only" ]; then
    exit 0
fi
exec vllm serve "$@"
