"""Local Marlin correction, not an NVIDIA checkpoint or upstream kernel change.

NVFP4 represents each block as fp4 * e4m3_block_scale * fp32_global_scale.
Marlin accepts one global scale for the fused gate/up projection. Re-express
both halves using max(gate_global, up_global), rounding their block scales to
E4M3 explicitly. This introduces additional block-scale quantization error;
it does not preserve the original quantized weights exactly. The packed FP4
values and checkpoint files are unchanged. Reconciliation precedes GEMM and
SwiGLU clipping, unlike an invalid downstream activation compensation.
"""

import torch


def _spark_reconcile_nvfp4_gate_up(block_scales, global_scales):
    """Return (reconciled E4M3 scales, shared globals, changed expert count).

    Input layout must be [expert, gate_rows + up_rows, groups]. Equal-scale
    experts retain their exact original bytes, including zero blocks. Zero
    globals are supported; negative or nonfinite scales fail closed. Work is
    chunked to bound temporary FP32 memory during model loading.
    """
    if block_scales.dtype != torch.float8_e4m3fn or block_scales.ndim != 3:
        raise ValueError("NVFP4 reconciliation requires 3D E4M3 block scales")
    experts, rows, groups = block_scales.shape
    if experts < 1 or rows < 2 or rows % 2 or groups < 1:
        raise ValueError("Malformed NVFP4 fused gate/up block-scale shape")
    if global_scales.device != block_scales.device:
        raise ValueError("NVFP4 block/global scale devices must match")
    if global_scales.ndim == 1 and global_scales.shape[0] == experts:
        paired = global_scales[:, None].expand(-1, 2)
    elif global_scales.ndim == 2 and tuple(global_scales.shape) == (experts, 1):
        paired = global_scales.expand(-1, 2)
    elif global_scales.ndim == 2 and tuple(global_scales.shape) == (experts, 2):
        paired = global_scales
    else:
        raise ValueError("Malformed NVFP4 gate/up global-scale shape")
    if paired.dtype != torch.float32:
        raise ValueError("NVFP4 gate/up global scales must be FP32")
    if not bool(torch.isfinite(paired).all()) or bool((paired < 0).any()):
        raise ValueError("NVFP4 global scales must be finite and nonnegative")
    shared = paired.amax(dim=1)
    denominator = torch.where(shared > 0, shared, torch.ones_like(shared))
    ratios = paired / denominator[:, None]
    # Both-zero experts need no conversion and must retain their original bytes.
    unequal = paired[:, 0] != paired[:, 1]
    changed = int(unequal.count_nonzero().item())
    result = block_scales.clone() if changed else block_scales
    half = rows // 2
    for start in range(0, experts, 8):
        stop = min(start + 8, experts)
        original = block_scales[start:stop].to(torch.float32)
        if not bool(torch.isfinite(original).all()) or bool((original < 0).any()):
            raise ValueError("NVFP4 block scales must be finite and nonnegative")
        if not changed:
            continue
        factors = ratios[start:stop, :, None, None]
        scaled = original.reshape(stop - start, 2, half, groups) * factors
        rounded = scaled.to(torch.float8_e4m3fn).reshape_as(original)
        # FP8 where/index kernels are not available in every supported torch
        # build; select exact bytes through the uint8 view instead.
        mask = unequal[start:stop, None, None]
        selected = torch.where(mask, rounded.view(torch.uint8),
                               block_scales[start:stop].view(torch.uint8))
        result[start:stop].copy_(selected.view(torch.float8_e4m3fn))
    return result, shared.contiguous(), changed
