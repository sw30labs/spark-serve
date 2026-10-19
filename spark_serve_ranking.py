"""Catalog-relative model scoring for the model cards' "best overall" star.

Scores are recomputed from whatever is in the catalog, so adding or removing a
model can move the star. Only models with a measured decode speed are eligible;
an unmeasured model never wins on paper.
"""
from __future__ import annotations

import math

# Weights sum to 1. Speed and quantization dominate; context and modality break ties.
WEIGHTS = {"speed": 0.40, "quant": 0.30, "context": 0.20, "modality": 0.10}

# Below 4 bits loses the most quality; 4-bit everywhere loses more than 4-bit experts
# with higher-precision attention and dense layers.
QUANT_QUALITY = {"aggressive": 0.3, "whole_model": 0.6, "experts_only": 1.0, "unknown": 0.5}

MODALITY_VALUE = {"text": 0.0, "image": 0.75, "video": 1.0}


def modalities(model: dict) -> list[str]:
    """Explicit catalog modalities, else text plus image when Hermes vision is enabled."""
    if model.get("modalities"):
        return list(model["modalities"])
    return ["text", "image"] if model.get("hermes_supports_vision") else ["text"]


def quant_tier(bits, scope) -> str:
    if not isinstance(bits, (int, float)):
        return "unknown"
    if bits < 4:
        return "aggressive"
    return {"all": "whole_model", "experts": "experts_only"}.get(scope, "unknown")


def _modality_score(mods) -> float:
    return max((MODALITY_VALUE.get(m, 0.0) for m in mods), default=0.0)


def score_models(items: list[dict]) -> dict[str, dict]:
    """items: list --json entries. Returns {id: {"score", "parts", "best"}} for measured models."""
    def speed(item):
        return ((item.get("performance") or {}).get("single_stream_tokens_per_second") or 0)

    eligible = [i for i in items if speed(i) > 0]
    if not eligible:
        return {}
    top_speed = max(speed(i) for i in eligible)
    contexts = [i["ctx"] for i in eligible if i.get("ctx", 0) > 0]
    low, high = (math.log(min(contexts)), math.log(max(contexts))) if contexts else (0.0, 0.0)
    out = {}
    for item in eligible:
        ctx = item.get("ctx") or 0
        parts = {
            "speed": speed(item) / top_speed,
            "quant": QUANT_QUALITY[quant_tier(item.get("quant_bits"), item.get("quant_scope"))],
            "context": (math.log(ctx) - low) / (high - low) if ctx > 0 and high > low else 1.0 if ctx > 0 else 0.0,
            "modality": _modality_score(item.get("modalities") or ["text"]),
        }
        out[item["id"]] = {"score": round(sum(WEIGHTS[k] * v for k, v in parts.items()), 4),
                           "parts": {k: round(v, 4) for k, v in parts.items()}, "weights": dict(WEIGHTS),
                           "best": False}
    # Ties go to the faster model, then catalog order.
    best = max(eligible, key=lambda i: (out[i["id"]]["score"], speed(i)))
    out[best["id"]]["best"] = True
    return out
