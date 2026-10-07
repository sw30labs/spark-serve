# GLM-5.3 Flash EXL3 with TensorFold on two Sparks

This recipe follows [Mia's TensorFold TP2 recipe](https://mia-ai.net/models/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold) at source commit `33b50fde06fd7ea604cbc6a663880068ab1e2ee4`. It serves the separate TensorFold EXL3 checkpoint, with DFlash2 drafts by default, through TensorFold 0.6.0 on both Sparks.

`source-pins.json` identifies the immutable ARM64 image, target and draft revisions, manifest hash, and every bundled runtime and preparation file. `model-source.json` contains full SHA-256 hashes for all 102 checkpoint files, totaling 178,058,596,393 bytes. The cache lives under `<hf_cache_host>/spark-serve/glm53-tensorfold/models/{target,draft}/<revision>` on each node.

Preparation downloads and authenticates both checkpoints on the head, builds a small derivative of the pinned upstream image, and copies the files and exact image to the worker over the configured QSFP interface. It authenticates the copied files, installed runtime and exact rank arguments before publishing either readiness receipt. Preparation leaves serving workloads running.

```sh
./spark-serve pull glm53-tensorfold
./spark-serve up glm53-tensorfold --node both
```

The runtime inventory was extracted from SHA-256-authenticated OCI image layers and independently matched against TensorFold source commit `c4646171139ee8a3c38103eaa1699dad226ec12b` with all 82 upstream patches applied in filename order. The image's extras-inclusive patch hash is `31557ed1cef6`; the installed package contains 426 authenticated files. Runtime startup checks these bytes, the package and dependency versions, both checkpoint snapshots, and each rank's CPU parser compatibility before serving.

The target checkpoint is MIT licensed. The DFlash2 checkpoint is CC BY-NC-ND 4.0. Setting `drafter = "mtp"` and `parallel = 1` uses the target's own MTP head; preparation retains both pinned checkpoints. Upstream patch, TensorFold and third-party notices are bundled under `runtime/upstream`, and NVIDIA's entrypoint and license notices remain in the derivative image.

The integrated defaults are TP2, q4 dense weights, FP8 KV, four concurrent requests and a 1,048,576-token prompt-plus-reply window. Hardware performance and quality qualification still require running this recipe on the two Sparks.
