# Provenance of the files in this directory

Base: vLLM main at commit 8a728663c1c3eeace834a95f5654fa653cc1998c (nightly image `vllm/vllm-openai:nightly-8a728663c1c3eeace834a95f5654fa653cc1998c`, 2026-09-04). The table retains the upstream recipe's attribution and original hashes, before the spark-serve amendment below. Upstream reports that its credits audit (`../../research/credits-audit-2026-09-05.md`) re-derived the listed changes and confirmed byte identity.

The credits-audit and `../modelopt_mtp_index.diff` references are paths in the
[upstream recipe at the pinned commit](https://github.com/tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark/tree/6ad1c8f15cbab1ababd2048e8e5f94094dbfc4a0),
not files included in this vendored subset. The local `overlays.sha256.json`
manifest records the exact redistributed files and their current hashes.

| File | Mounted at | Base + change | Author of the change | sha256 |
|---|---|---|---|---|
| ops_ple.py | vllm/models/qwen4_exp/nvidia/ops/ple.py | base + PR #55375 (fused PLE conv state-index stride fix, merged 2026-09-05) | peakcrosser7 | cc8994b653dd4b59d6ba35b6ec10af34017e9944482ec67cad6a32a02c5dd00a |
| ops_qsa.py | vllm/models/qwen4_exp/nvidia/ops/qsa.py | base + PR #54846 (fp8_e4m3 / nvfp4 KV on the QSA path, open; rebased onto the nightly, 3-way merge conflict-free) | andreasgru | 1e928e2994233496e9ef447768e8b63acbedc22e242d4b2fc9f4ed861a1616c3 |
| qsa.py | vllm/models/qwen4_exp/nvidia/qsa.py | base + PR #54846 (verbatim the PR-head file 1e11f40bbc) | andreasgru | f2fc3fb43f6c6e84e892a995697fb59a85ee8df2ef2959943c96cf0233078da7 |
| platforms_interface.py | vllm/platforms/interface.py | base + PR #54846 | andreasgru | 443b556b984b306c0274c58af8658ef853c664b8beae6deefda41908f789d26d |
| modelopt.py | vllm/model_executor/layers/quantization/modelopt.py | base + our two MTP-loading fixes (`../modelopt_mtp_index.diff`) | Kai (Tech2Wild), 2026-09-05; same gaps fixed independently the same day by sfxnz (MIT) and MiaAI-Lab (AGPL), no shared code | eff707e7f42b12a0483f41461aaaf786399e860077b767a76dc236a869c7e303 |

All files keep vLLM's Apache-2.0 SPDX header. vLLM is Copyright contributors to the vLLM project, Apache-2.0.

## spark-serve deployment amendment (2026-09-13)

`modelopt.py` is additionally modified to recognize `FP8_PB_WO` in the block-FP8 RoutedExperts branch. NVIDIA checkpoint revision `fc694b54fb0174e0913e6adf86691ef85a4ead47` uses that canonical label for the same 128x128 block-FP8 MTP tensors. This is the one-line correction from recipe [PR #1 by gdeyoung](https://github.com/tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark/pull/1), head `a85e8943514602b3fcb2be6fedd71d9368932248`, independently confirmed in [issue #2](https://github.com/tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark/issues/2) and current vLLM main (`_BLOCK_FP8_MOE_ALGOS`). NVIDIA weight files are unchanged.

Amended `modelopt.py` SHA256: `472b41a26267fe6bc78ea3d872863b518154d92b6672eb4050cd92e63e25c69b`. The original source hash remains in the table above. A prominent modification notice was added to the file header. `tests/test_qwen_modelopt.py` in the spark-serve repository executes the actual isolated dispatch/resolution methods with CUDA-facing constructor stubs; it fails on the unmodified recipe and passes after the amendment against the pinned NVIDIA metadata. This check does not substitute for a real model load and generation test.
