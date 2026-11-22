"""LLM provider abstractions: zero-key DeterministicReplayProvider and OpenAI-compatible HTTP client."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import re
from typing import Any

import httpx

from routeguard.pricing import MODEL_TIERS
from routeguard.security import validate_upstream_url


class UpstreamProviderError(RuntimeError):
    """Raised when an upstream LLM tier returns HTTP 429/5xx or times out."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass
class ProviderResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    model_id: str
    simulated_latency_ms: float


class DeterministicReplayProvider:
    """Reproducible local replay engine modeling realistic Tier-1 JSON failure modes & upstream faults."""

    def __init__(self, *, fault_tiers: dict[str, int] | None = None) -> None:
        self.fault_tiers = dict(fault_tiers or {})

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(16, int(len((text or "").split()) * 1.45))

    def complete(
        self,
        *,
        tier: str,
        prompt: str,
        schema_name: str | None = None,
        repair_attempt: int = 0,
    ) -> ProviderResponse:
        if tier in self.fault_tiers:
            code = self.fault_tiers[tier]
            raise UpstreamProviderError(code, f"Upstream HTTP {code} simulated fault on {tier}")

        spec = MODEL_TIERS[tier]
        p_tokens = self._estimate_tokens(prompt) + (65 if repair_attempt > 0 else 0)
        digest = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:6], 16)

        if schema_name == "InvoiceExtraction":
            inv_m = re.search(r"(INV-[A-Z0-9-]+)", prompt, re.I)
            vend_m = re.search(r"Vendor:\s*([A-Za-z0-9 &.-]+)", prompt, re.I)
            amt_m = re.search(r"(?:Total|Amount):\s*(?:Rs\.?|INR|₹|\$)?\s*([\d,]+(?:\.\d+)?)", prompt, re.I)
            inv_id = inv_m.group(1) if inv_m else f"INV-{1000 + (digest % 8999)}"
            vendor = vend_m.group(1).strip() if vend_m else "CloudHost India"
            amt = float(amt_m.group(1).replace(",", "")) if amt_m else 1499.0

            # Simulate realistic Tier-1 cold-shot formatting glitches fixed on repair attempt >= 1 or Tier 2+
            if tier == "tier1_fast" and repair_attempt == 0 and (digest % 3 != 0):
                if digest % 2 == 0:
                    # Fenced markdown + dirty string float + renamed key -> fails strict Pydantic float/required key!
                    raw_obj = {
                        "invoice_id": inv_id,
                        "vendor": vendor,
                        "total_amount": f"Rs. {amt:,.2f}",
                        "curr": "INR",
                    }
                    text = f"Sure, here is the extracted invoice:\n```json\n{json.dumps(raw_obj)}\n```"
                else:
                    # Missing required float tax_amount and trailing chatter
                    text = f'{{"invoice_id": "{inv_id}", "vendor": "{vendor}", "currency": "INR"}}'
            else:
                text = json.dumps({
                    "invoice_id": inv_id,
                    "vendor": vendor,
                    "currency": "USD" if "$" in prompt else "INR",
                    "total_amount": round(amt, 2),
                    "tax_amount": round(amt * 0.18, 2),
                })

        elif schema_name == "SupportTicketTriage":
            low = prompt.lower()
            prio = "P1" if any(k in low for k in ("down", "outage", "double charge", "500", "crash", "urgent")) else ("P2" if any(k in low for k in ("slow", "delay", "invoice", "refund")) else "P3")
            queue = "billing_ops" if any(k in low for k in ("refund", "charge", "invoice", "billing", "payment")) else ("platform_sre" if any(k in low for k in ("500", "timeout", "outage", "latency", "api")) else "account_support")
            if tier == "tier1_fast" and repair_attempt == 0 and (digest % 5 == 0):
                # Invalid enum value on initial Tier-1 attempt -> caught and healed by Pydantic repair!
                text = json.dumps({"queue": queue, "priority": "HIGHEST", "summary": "Escalated support request."})
            else:
                text = json.dumps({
                    "queue": queue,
                    "priority": prio,
                    "summary": f"Routed to {queue} under SLA {prio}.",
                })

        elif tier == "tier2_balanced":
            text = (
                "WITH ordered_events AS (\n"
                "  SELECT tenant_id, DATE(created_at) AS event_day,\n"
                "         ROW_NUMBER() OVER (PARTITION BY tenant_id ORDER BY created_at) AS seq_id\n"
                "  FROM events\n"
                ")\n"
                "SELECT event_day, COUNT(DISTINCT tenant_id) AS active_tenants FROM ordered_events GROUP BY 1;"
            )
        elif tier == "tier3_frontier":
            text = (
                "Root-Cause Diagnosis: Under concurrent retry storms, unindexed lock contention triggers connection-pool exhaustion "
                "across downstream microservices. Recommended fix: add idempotency keys, bound exponential jitter retries, "
                "and isolate slow webhook consumers behind circuit breakers with explicit dead-letter queues."
            )
        else:
            text = json.dumps({"status": "processed", "tier": tier, "summary": "Completed fast-path classification."})

        c_tokens = self._estimate_tokens(text)
        return ProviderResponse(
            text=text,
            prompt_tokens=p_tokens,
            completion_tokens=c_tokens,
            model_id=spec.model_id,
            simulated_latency_ms=spec.base_latency_ms,
        )


class OpenAICompatibleProvider:
    """Optional live HTTP provider targeting OpenAI/LiteLLM/Ollama/Gemini-compatible endpoints."""

    def __init__(self, base_url: str | None = None, api_key: str | None = None, *, allow_localhost: bool = False) -> None:
        raw_url = (base_url or os.environ.get("ROUTEGUARD_BASE_URL") or "").rstrip("/")
        self.base_url = validate_upstream_url(raw_url, allow_localhost=allow_localhost) if raw_url else ""
        self.allow_localhost = allow_localhost
        self.api_key = api_key or os.environ.get("ROUTEGUARD_API_KEY") or ""

    def complete(self, *, tier: str, prompt: str, schema_name: str | None = None, repair_attempt: int = 0) -> ProviderResponse:
        if not self.base_url:
            raise RuntimeError("ROUTEGUARD_BASE_URL not configured.")
        validate_upstream_url(self.base_url, allow_localhost=self.allow_localhost)
        spec = MODEL_TIERS[tier]
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": spec.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
        }
        resp = httpx.post(f"{self.base_url}/chat/completions", headers=headers, json=payload, timeout=25.0)
        if resp.status_code in (429, 500, 502, 503, 504):
            raise UpstreamProviderError(resp.status_code, resp.text[:200])
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return ProviderResponse(
            text=text,
            prompt_tokens=int(usage.get("prompt_tokens", 50)),
            completion_tokens=int(usage.get("completion_tokens", 50)),
            model_id=spec.model_id,
            simulated_latency_ms=0.0,
        )
