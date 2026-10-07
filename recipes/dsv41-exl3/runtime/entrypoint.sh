#!/bin/bash
set -euo pipefail

# The published image already contains the EXL3 overlay. This applies the one
# Responses API patch added after that image was published, then serves.
python3 /opt/dsv41/patch_responses_content_types.py

if [ "${1:-}" = "--patches-only" ]; then
    exit 0
fi
exec vllm serve "$@"
