FROM ghcr.io/miaai-lab/deepseek-v4.1-flash-exl3-2x-dgx-sparks@sha256:2f0cf3adc0f989c1d446be274df864eb799630175f604c3b22b71b7205971dce
LABEL org.opencontainers.image.description="DeepSeek-V4.1-Flash EXL3 2.9bpw for spark-serve" \
      org.opencontainers.image.source="https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks" \
      org.opencontainers.image.licenses="AGPL-3.0" \
      ai.spark-serve.recipe.commit="6f7d1590ad49a2b8995188e45d7b9db31e677452" \
      ai.spark-serve.recipe.base="sha256:2f0cf3adc0f989c1d446be274df864eb799630175f604c3b22b71b7205971dce" \
      ai.spark-serve.recipe.stamp="8c01a8543ff7e7b222ae1cd46b8be5051838dc7a7b075b4e647f3785c0429584"

# Kernels and the baked overlay stay in the published image. The Responses
# content-type patch is the only overlay file newer than that image.
COPY runtime/upstream/patch_responses_content_types.py /opt/dsv41/patch_responses_content_types.py
COPY runtime/entrypoint.sh verify.py model-source.json source-pins.json /opt/spark-serve/dsv41-exl3/
RUN chmod 755 /opt/spark-serve/dsv41-exl3/entrypoint.sh
ENTRYPOINT ["/opt/spark-serve/dsv41-exl3/entrypoint.sh"]
CMD []
