FROM ghcr.io/miaai-lab/qwen3.8-flash-next-single-dgx-spark-tensorfold@sha256:fd30b84dce7a43daaa2be3443e1bd0e7c826923e9cb77a995c4720e76c752c64
LABEL org.opencontainers.image.description="Pinned Qwen3.8 MLX 4-bit TensorFold single-Spark experimental recipe" \
      ai.spark-serve.recipe.commit="a3aa89835022c55ca8e55008c37785954834e04f" \
      ai.spark-serve.recipe.base="sha256:fd30b84dce7a43daaa2be3443e1bd0e7c826923e9cb77a995c4720e76c752c64"

COPY runtime/upstream/ /opt/spark-serve/qwen38-tensorfold/upstream/
COPY runtime/verify_runtime.py runtime/check_compatibility.py runtime/entrypoint.sh runtime/upstream-files.json runtime/installed-runtime.json /opt/spark-serve/qwen38-tensorfold/
COPY verify.py model-source.json source-pins.json /opt/spark-serve/qwen38-tensorfold/
RUN python3 /opt/spark-serve/qwen38-tensorfold/verify_runtime.py \
    && test -x /opt/nvidia/nvidia_entrypoint.sh \
    && chmod 755 /opt/spark-serve/qwen38-tensorfold/entrypoint.sh
ENV HF_HOME=/cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    TENSORFOLD_NO_UPDATE_CHECK=1 TORCH_EXTENSIONS_DIR=/cache/torch_extensions TRITON_CACHE_DIR=/cache/triton
ENTRYPOINT ["/opt/spark-serve/qwen38-tensorfold/entrypoint.sh"]
CMD []
