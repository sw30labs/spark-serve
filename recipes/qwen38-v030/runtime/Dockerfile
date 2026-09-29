FROM vllm/vllm-openai@sha256:4864d46625cbc3307623e29ac742030655e27249feba7b97ec925ce4cc4dfb56
LABEL org.opencontainers.image.description="Qwen3.8-Flash-Next NVIDIA NVFP4 on vLLM 0.30.0, pinned single-Spark overlays" \
      ai.spark-serve.recipe.commit="b8439110eec0230facbe4ddf0dffe01b8f769be0" \
      ai.spark-serve.recipe.base="sha256:4864d46625cbc3307623e29ac742030655e27249feba7b97ec925ce4cc4dfb56"

COPY runtime/upstream/ /opt/spark-serve/qwen38-v030/upstream/
COPY runtime/apply_patches.py runtime/verify_runtime.py runtime/check_compatibility.py runtime/entrypoint.sh runtime/upstream-files.json /opt/spark-serve/qwen38-v030/
COPY verify.py model-source.json source-pins.json /opt/spark-serve/qwen38-v030/
# Generate all four modules from the immutable base without CUDA/model loading.
# The byte-identical generators reject source drift; no legacy model overlays
# from the old Qwen runtime are mixed into vLLM 0.30.
RUN python3 /opt/spark-serve/qwen38-v030/apply_patches.py \
    && python3 /opt/spark-serve/qwen38-v030/verify_runtime.py \
    && chmod 755 /opt/spark-serve/qwen38-v030/entrypoint.sh
ENTRYPOINT ["/opt/spark-serve/qwen38-v030/entrypoint.sh"]
CMD []
