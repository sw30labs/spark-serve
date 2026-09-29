FROM ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks@sha256:447114ee77d14c9b4732ee23978ada2a0ee9027868a231d6fd42700a8b25be1d
LABEL org.opencontainers.image.description="GLM-5.3-Flash EXL3 TP2, pinned MiaAI-Lab runtime overlays for spark-serve" \
      ai.spark-serve.recipe.commit="b2c5986324a7bac7b02bf0a47ed7b92b25e56016" \
      ai.spark-serve.recipe.base="sha256:447114ee77d14c9b4732ee23978ada2a0ee9027868a231d6fd42700a8b25be1d"

# The public image carries the compiled ARM64 EXL3/E3 kernels and InstantTensor.
# Python patches stay byte-identical to the pinned source and apply at startup
# in its launcher order, when the actual serving environment is available.
COPY runtime/upstream/ /opt/glm53/
COPY runtime/entrypoint.sh runtime/verify_runtime.py runtime/upstream-files.json runtime/overlay-order.txt /opt/spark-serve/glm53-exl3/
COPY verify.py model-source.json source-pins.json /opt/spark-serve/glm53-exl3/
RUN python3 /opt/spark-serve/glm53-exl3/verify_runtime.py \
    && chmod 755 /opt/spark-serve/glm53-exl3/entrypoint.sh
ENTRYPOINT ["/opt/spark-serve/glm53-exl3/entrypoint.sh"]
CMD []
