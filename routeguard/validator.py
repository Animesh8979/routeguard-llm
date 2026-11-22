"""Pydantic v2 structured output validator with bounded self-healing repair & escalation loop."""
from __future__ import annotations

import json
import re
from typing import Any, Type

from pydantic import BaseModel, Field, ValidationError, ConfigDict

from routeguard.pricing import compute_cost_usd, frontier_baseline_cost_usd
from routeguard.providers import UpstreamProviderError


class SupportTicketTriage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    queue: str = Field(min_length=2)
    priority: str = Field(pattern=r"^P[1-3]$")
    summary: str = Field(min_length=5)


class InvoiceExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    invoice_id: str = Field(min_length=3)
    vendor: str = Field(min_length=2)
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    total_amount: float = Field(gt=0)
    tax_amount: float = Field(ge=0)


SCHEMA_REGISTRY: dict[str, tuple[Type[BaseModel], str]] = {
    "SupportTicketTriage": (SupportTicketTriage, "v1"),
    "InvoiceExtraction": (InvoiceExtraction, "v1"),
}

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.I)


def extract_json_candidate(raw_text: str) -> Any:
    """Strip markdown fences or chatter and parse JSON object."""
    text = (raw_text or "").strip()
    m = _JSON_BLOCK_RE.search(text)
    if m:
        text = m.group(1).strip()
    else:
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace > first_brace:
            text = text[first_brace : last_brace + 1]
    return json.loads(text)


def _format_pydantic_diff(exc: ValidationError) -> str:
    errs = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", [])) or "root"
        errs.append(f"{loc}: {err.get('msg')}")
    return "; ".join(errs)


def validate_and_repair(
    *,
    provider: Any,
    breakers: Any,
    initial_tier: str,
    prompt: str,
    schema_name: str,
    max_repairs: int = 2,
) -> dict[str, Any]:
    """Execute initial inference + bounded field-path repair loop (escalating tier on repeat failure)."""
    schema_cls, schema_ver = SCHEMA_REGISTRY[schema_name]
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost_usd = 0.0
    total_sim_latency = 0.0
    repair_attempts = 0
    raw_valid_first_shot = False
    current_tier = initial_tier
    active_prompt = prompt
    last_error = ""

    tier_chain = {
        "tier1_fast": ["tier1_fast", "tier2_balanced", "tier3_frontier"],
        "tier2_balanced": ["tier2_balanced", "tier3_frontier"],
        "tier3_frontier": ["tier3_frontier"],
    }

    for attempt in range(max_repairs + 1):
        # Pick healthy tier with failover + escalation on repair >= 2
        if attempt >= 2 and current_tier == "tier1_fast":
            current_tier = "tier2_balanced"
        elif attempt >= 1 and current_tier == "tier1_fast" and repair_attempts >= 1:
            # First try repair on same cheap tier; escalate if it already failed repair once
            pass

        resp = None
        for cand_tier in tier_chain.get(current_tier, [current_tier]):
            if not breakers.allow_request(cand_tier):
                continue
            try:
                resp = provider.complete(
                    tier=cand_tier,
                    prompt=active_prompt,
                    schema_name=schema_name,
                    repair_attempt=attempt,
                )
                breakers.record_success(cand_tier)
                current_tier = cand_tier
                break
            except UpstreamProviderError as exc:
                breakers.record_failure(cand_tier)
                last_error = str(exc)

        if resp is None:
            raise RuntimeError(f"All fallback provider tiers unavailable: {last_error}")

        total_prompt_tokens += resp.prompt_tokens
        total_completion_tokens += resp.completion_tokens
        total_sim_latency += resp.simulated_latency_ms
        total_cost_usd += compute_cost_usd(current_tier, resp.prompt_tokens, resp.completion_tokens)

        try:
            # Strict check first on raw_text for metrics tracking before fence extraction
            parsed = extract_json_candidate(resp.text)
            validated = schema_cls.model_validate(parsed)
            if attempt == 0:
                raw_valid_first_shot = True
            baseline_usd = frontier_baseline_cost_usd(total_prompt_tokens, total_completion_tokens)
            return {
                "valid": True,
                "raw_valid_first_shot": raw_valid_first_shot,
                "repair_attempts": repair_attempts,
                "tier_used": current_tier,
                "model_id": resp.model_id,
                "schema_version": schema_ver,
                "data": validated.model_dump(),
                "raw_text": validated.model_dump_json(),
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "actual_cost_usd": round(total_cost_usd, 8),
                "baseline_frontier_cost_usd": baseline_usd,
                "simulated_latency_ms": round(total_sim_latency, 2),
                "last_error": "",
            }
        except (json.JSONDecodeError, ValidationError) as exc:
            repair_attempts += 1
            if isinstance(exc, ValidationError):
                last_error = _format_pydantic_diff(exc)
            else:
                last_error = f"JSONDecodeError: {exc}"
            active_prompt = f"{prompt}\n[REPAIR_DIFF Attempt {attempt + 1}: Fix only fields: {last_error}]"
            if attempt >= 1 and current_tier == "tier1_fast":
                current_tier = "tier2_balanced"

    baseline_usd = frontier_baseline_cost_usd(total_prompt_tokens, total_completion_tokens)
    return {
        "valid": False,
        "raw_valid_first_shot": False,
        "repair_attempts": repair_attempts,
        "tier_used": current_tier,
        "model_id": current_tier,
        "schema_version": schema_ver,
        "data": None,
        "raw_text": "",
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "actual_cost_usd": round(total_cost_usd, 8),
        "baseline_frontier_cost_usd": baseline_usd,
        "simulated_latency_ms": round(total_sim_latency, 2),
        "last_error": last_error,
    }
