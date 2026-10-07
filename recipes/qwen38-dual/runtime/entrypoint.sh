#!/bin/bash
set -euo pipefail
runtime=/opt/spark-serve/qwen38-dual
python3 "$runtime/verify_runtime.py"
if [ "${1:-}" = "--verify-runtime" ]; then
    exit 0
fi
exec vllm serve "$@"
