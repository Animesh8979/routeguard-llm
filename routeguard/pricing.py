"""Transparent 2026 LLM pricing catalog and USD token cost accounting."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    tier: str
    model_id: str
    provider: str
    input_usd_per_1m: float
    output_usd_per_1m: float
    base_latency_ms: float


MODEL_TIERS: dict[str, ModelSpec] = {
    "tier1_fast": ModelSpec(
        tier="tier1_fast",
        model_id="gemini-2.5-flash-lite",
        provider="google-fast",
        input_usd_per_1m=0.10,
        output_usd_per_1m=0.40,
        base_latency_ms=140.0,
    ),
    "tier2_balanced": ModelSpec(
        tier="tier2_balanced",
        model_id="gemini-2.5-flash",
        provider="google-balanced",
        input_usd_per_1m=0.30,
        output_usd_per_1m=1.20,
        base_latency_ms=320.0,
    ),
    "tier3_frontier": ModelSpec(
        tier="tier3_frontier",
        model_id="frontier-reasoning-large",
        provider="frontier-primary",
        input_usd_per_1m=3.00,
        output_usd_per_1m=15.00,
        base_latency_ms=1150.0,
    ),
}


def compute_cost_usd(tier: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Compute exact USD cost for a tier given input and output token counts."""
    spec = MODEL_TIERS.get(tier, MODEL_TIERS["tier3_frontier"])
    in_cost = (max(0, prompt_tokens) / 1_000_000.0) * spec.input_usd_per_1m
    out_cost = (max(0, completion_tokens) / 1_000_000.0) * spec.output_usd_per_1m
    return round(in_cost + out_cost, 8)


def frontier_baseline_cost_usd(prompt_tokens: int, completion_tokens: int) -> float:
    """Reference Always-Frontier cost for direct ROI measurement."""
    return compute_cost_usd("tier3_frontier", prompt_tokens, completion_tokens)
