#!/bin/bash
set -euo pipefail
runtime=/opt/spark-serve/qwen38-v030
python3 "$runtime/verify_runtime.py"
if [ "${1:-}" = "--verify-runtime" ]; then
    exit 0
fi
# The controller supplies pinned model path, cache paths and serving arguments.
exec vllm serve "$@"
