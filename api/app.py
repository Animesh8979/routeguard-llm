"""OpenAI-compatible FastAPI proxy exposing cost routing, self-healing extraction, metrics, and CI eval."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from routeguard.cache import SemanticCache
from routeguard.eval_harness import GatewayRuntime, handle_request, run_eval_suite
from routeguard.ledger import TelemetryLedger
from routeguard.providers import DeterministicReplayProvider
from routeguard.router import BudgetExceededError, HybridComplexityRouter
from routeguard.security import MAX_PROMPT_CHARS, verify_api_key

app = FastAPI(
    title="RouteGuard LLM Cost-Routing Proxy",
    version="0.1.0",
    description="Hybrid complexity router, Pydantic self-healing JSON extraction, SQLite WAL semantic cache, and circuit breakers.",
)


def _enforce_key(x_routeguard_key: Optional[str]) -> None:
    expected = os.environ.get("ROUTEGUARD_API_KEY")
    if expected and not verify_api_key(x_routeguard_key, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing X-RouteGuard-Key.")


def _get_runtime() -> GatewayRuntime:
    data_dir = Path(os.environ.get("ROUTEGUARD_DATA_DIR", "data_runtime/api"))
    data_dir.mkdir(parents=True, exist_ok=True)
    return GatewayRuntime(
        cache=SemanticCache(data_dir / "cache.db"),
        ledger=TelemetryLedger(data_dir / "ledger.db"),
        router=HybridComplexityRouter(),
        provider=DeterministicReplayProvider(),
    )


class ChatCompletionRequest(BaseModel):
    prompt: str = Field(min_length=1)
    tenant_id: str = "default"
    force_tier: Optional[str] = None
    use_cache: bool = True


class ExtractRequest(BaseModel):
    prompt: str = Field(min_length=1)
    schema_name: str = Field(pattern=r"^(InvoiceExtraction|SupportTicketTriage)$")
    schema_version: str = "v1"
    tenant_id: str = "default"
    use_cache: bool = True


@app.get("/health")
def health() -> dict[str, Any]:
    rt = _get_runtime()
    return {"status": "ok", "breakers": rt.router.breakers.snapshot()}


@app.post("/v1/chat/completions")
def chat_completions(
    req: ChatCompletionRequest,
    x_routeguard_key: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    _enforce_key(x_routeguard_key)
    if len(req.prompt) > MAX_PROMPT_CHARS:
        raise HTTPException(status_code=413, detail=f"Prompt exceeds MAX_PROMPT_CHARS={MAX_PROMPT_CHARS}.")
    rt = _get_runtime()
    try:
        out = handle_request(rt, prompt=req.prompt, tenant_id=req.tenant_id, force_tier=req.force_tier, use_cache=req.use_cache)
    except BudgetExceededError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    if out.get("blocked"):
        raise HTTPException(status_code=400, detail={"error": "Prompt blocked by RouteGuard guardrails", "reasons": out["block_reasons"]})
    return out


@app.post("/v1/extract")
def structured_extract(
    req: ExtractRequest,
    x_routeguard_key: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    _enforce_key(x_routeguard_key)
    if len(req.prompt) > MAX_PROMPT_CHARS:
        raise HTTPException(status_code=413, detail=f"Prompt exceeds MAX_PROMPT_CHARS={MAX_PROMPT_CHARS}.")
    rt = _get_runtime()
    try:
        out = handle_request(
            rt,
            prompt=req.prompt,
            tenant_id=req.tenant_id,
            schema_name=req.schema_name,
            schema_version=req.schema_version,
            use_cache=req.use_cache,
        )
    except BudgetExceededError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    if out.get("blocked"):
        raise HTTPException(status_code=400, detail={"error": "Prompt blocked by RouteGuard guardrails", "reasons": out["block_reasons"]})
    return out


@app.get("/v1/metrics")
def get_metrics(x_routeguard_key: Optional[str] = Header(default=None)) -> dict[str, Any]:
    _enforce_key(x_routeguard_key)
    rt = _get_runtime()
    return {
        "metrics": rt.ledger.get_metrics(),
        "circuit_breakers": rt.router.breakers.snapshot(),
    }


@app.post("/v1/eval/run")
def trigger_eval(x_routeguard_key: Optional[str] = Header(default=None)) -> dict[str, Any]:
    _enforce_key(x_routeguard_key)
    data_dir = Path(os.environ.get("ROUTEGUARD_DATA_DIR", "data_runtime/api"))
    return run_eval_suite(data_dir / "eval_run")
