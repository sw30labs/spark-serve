# Measured GB10 image, plus the three drop-in fixes required for TP2 on SM121.
# Upstream: tonyd2wild/MiMo-V2.6-Flash-DGX-Spark-Recipe @ 13621bb3cc6fd30a94d53609320599d1f1134686
# Patch 04 (DFlash value scale) is not applied. That measurement showed no gain.
ARG BASE_IMAGE=ghcr.io/tonyd2wild/vllm-glm53-flash@sha256:4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="spark-serve MiMo-V2.6-Flash-RL"
LABEL org.opencontainers.image.version="0.1.0"
LABEL org.opencontainers.image.source="https://github.com/tonyd2wild/MiMo-V2.6-Flash-DGX-Spark-Recipe"

RUN test -f /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/mimo_v2.py \
 && test -f /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/mimo_v2_omni.py \
 && test -f /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/triton_attn_diffkv.py

COPY patches/mimo_v2.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/mimo_v2.py
COPY patches/mimo_v2_omni.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/mimo_v2_omni.py
COPY patches/triton_attn_diffkv.py /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/triton_attn_diffkv.py
COPY verify.py /opt/spark-serve/mimo-v26-flash/verify.py

# Audio requests import these. The base image does not ship them.
RUN pip install --no-deps --root-user-action=ignore soundfile av \
 && python3 -c "import soundfile, av"
