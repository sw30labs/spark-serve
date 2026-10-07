FROM ghcr.io/miaai-lab/glm-5.3-flash-exl3-2x-dgx-sparks-tensorfold@sha256:cbb4b3c66273e2965dd40a7227e7a5243db333fe250113fb3987462ad4f12588
LABEL org.opencontainers.image.description="Pinned GLM-5.3 Flash EXL3 TensorFold TP2 recipe" \
      ai.spark-serve.recipe.commit="33b50fde06fd7ea604cbc6a663880068ab1e2ee4" \
      ai.spark-serve.recipe.base="sha256:cbb4b3c66273e2965dd40a7227e7a5243db333fe250113fb3987462ad4f12588"

COPY runtime/upstream/ /opt/spark-serve/glm53-tensorfold/upstream/
COPY runtime/verify_runtime.py runtime/check_compatibility.py runtime/entrypoint.sh runtime/upstream-files.json runtime/installed-runtime.json /opt/spark-serve/glm53-tensorfold/
COPY verify.py model-source.json source-pins.json /opt/spark-serve/glm53-tensorfold/
RUN python3 /opt/spark-serve/glm53-tensorfold/verify_runtime.py \
    && test -x /opt/nvidia/nvidia_entrypoint.sh \
    && chmod 755 /opt/spark-serve/glm53-tensorfold/entrypoint.sh
ENV HF_HOME=/cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    TENSORFOLD_NO_UPDATE_CHECK=1 TORCH_EXTENSIONS_DIR=/cache/torch_extensions TRITON_CACHE_DIR=/cache/triton
ENTRYPOINT ["/opt/spark-serve/glm53-tensorfold/entrypoint.sh"]
CMD []
