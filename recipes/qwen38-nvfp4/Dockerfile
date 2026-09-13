# Single-Spark overlay audited from tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark
# commit 6ad1c8f15cbab1ababd2048e8e5f94094dbfc4a0 (Apache-2.0).
FROM vllm/vllm-openai@sha256:a551e05307cd2e0092139d84db32af9c97e67d2eeeff072d21e429131d8c23f0
LABEL org.opencontainers.image.description="Qwen3.8-Flash-Next official NVIDIA NVFP4, staged disk PLE, MTP draft vocabulary overlay for DGX Spark" \
      ai.spark-serve.recipe.commit="6ad1c8f15cbab1ababd2048e8e5f94094dbfc4a0" \
      ai.spark-serve.recipe.source="https://github.com/tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark"
COPY single-spark-vllm-tp1/patch/ple_layer.py /usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/ple_layer.py
COPY single-spark-vllm-tp1/patch/ple_mmap.py /usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/ops/ple_mmap.py
COPY single-spark-vllm-tp1/patch/model_state.py /usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/model_state.py
COPY single-spark-vllm-tp1/patch/mtp_draft_vocab.py /usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/mtp.py
COPY single-spark-vllm-tp1/patch/upstream-overlays/ops_ple.py /usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/ops/ple.py
COPY single-spark-vllm-tp1/patch/upstream-overlays/ops_qsa.py /usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/ops/qsa.py
COPY single-spark-vllm-tp1/patch/upstream-overlays/qsa.py /usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/qsa.py
COPY single-spark-vllm-tp1/patch/upstream-overlays/platforms_interface.py /usr/local/lib/python3.12/dist-packages/vllm/platforms/interface.py
COPY single-spark-vllm-tp1/patch/upstream-overlays/modelopt.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/modelopt.py
COPY LICENSE NOTICE /usr/local/share/licenses/qwen38-spark-recipe/
COPY single-spark-vllm-tp1/patch/upstream-overlays/PROVENANCE.md /usr/local/share/licenses/qwen38-spark-recipe/PROVENANCE.md
RUN python3 -m py_compile \
    /usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/ple_layer.py \
    /usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/ops/ple_mmap.py \
    /usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/model_state.py \
    /usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/mtp.py \
    /usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/ops/ple.py \
    /usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/ops/qsa.py \
    /usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/qsa.py \
    /usr/local/lib/python3.12/dist-packages/vllm/platforms/interface.py \
    /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/modelopt.py

COPY verify.py model-source.json recipe.json overlays.sha256.json /opt/spark-serve/qwen38/
ENTRYPOINT ["vllm", "serve"]
CMD []
