FROM ghcr.io/miaai-lab/qwen3.8-flash-dual-dgx-sparks-tensorfold@sha256:e188128ddabdc67ac02c05781f5c0c6b348db5f050cac297861d82b7f086e18c
LABEL org.opencontainers.image.description="Pinned Qwen3.8-Flash-Next NVFP4 native TensorFold TP2 recipe" \
      ai.spark-serve.recipe.commit="be3ed857ab1055ca47b36134d22691f00f2bb94a" \
      ai.spark-serve.recipe.base="sha256:e188128ddabdc67ac02c05781f5c0c6b348db5f050cac297861d82b7f086e18c"

COPY runtime/upstream/ /opt/spark-serve/qwen38-dual-tensorfold/upstream/
COPY runtime/verify_runtime.py runtime/check_compatibility.py runtime/entrypoint.sh runtime/upstream-files.json runtime/installed-runtime.json /opt/spark-serve/qwen38-dual-tensorfold/
COPY verify.py model-source.json source-pins.json /opt/spark-serve/qwen38-dual-tensorfold/
RUN python3 /opt/spark-serve/qwen38-dual-tensorfold/verify_runtime.py \
    && test -x /opt/nvidia/nvidia_entrypoint.sh \
    && chmod 755 /opt/spark-serve/qwen38-dual-tensorfold/entrypoint.sh
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TENSORFOLD_NO_UPDATE_CHECK=1
ENTRYPOINT ["/opt/spark-serve/qwen38-dual-tensorfold/entrypoint.sh"]
CMD []
