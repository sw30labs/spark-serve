FROM vllm/vllm-openai@sha256:4864d46625cbc3307623e29ac742030655e27249feba7b97ec925ce4cc4dfb56
LABEL org.opencontainers.image.description="Qwen3.8-Flash-Next NVIDIA NVFP4, vLLM 0.30.0 TP2+EP+MTP3" \
      ai.spark-serve.recipe.commit="cd839d0b62f6c737178a0c2037733420e0524ebf" \
      ai.spark-serve.recipe.base="sha256:4864d46625cbc3307623e29ac742030655e27249feba7b97ec925ce4cc4dfb56"

COPY runtime/upstream/ /opt/spark-serve/qwen38-dual/upstream/
COPY runtime/apply_patches.py runtime/verify_runtime.py runtime/check_compatibility.py runtime/entrypoint.sh runtime/upstream-files.json /opt/spark-serve/qwen38-dual/
COPY verify.py model-source.json source-pins.json /opt/spark-serve/qwen38-dual/
RUN python3 /opt/spark-serve/qwen38-dual/apply_patches.py \
    && python3 /opt/spark-serve/qwen38-dual/verify_runtime.py \
    && chmod 755 /opt/spark-serve/qwen38-dual/entrypoint.sh
ENTRYPOINT ["/opt/spark-serve/qwen38-dual/entrypoint.sh"]
CMD []
