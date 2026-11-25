"""Tests for entropy-margin abstention, OpenAI SSE streaming, and public RouterBench OOD evaluation."""
from __future__ import annotations

import os
from pathlib import Path

from fastapi.testclient import TestClient

from api.app import app
from routeguard.router import HybridComplexityRouter
from routeguard.routerbench_eval import run_routerbench_ood_eval


def test_entropy_margin_escalation_triggers_on_ambiguous_ood_query() -> None:
    router = HybridComplexityRouter()
    out = router.select_tier(
        "Explain how monetary policy transmission affects yield curves",
        has_schema=False,
        uncertainty_margin_threshold=0.95,
    )
    assert "entropy" in out
    assert "confidence_margin" in out
    if out["complexity_label"] == "simple":
        assert out["uncertainty_escalated"] is True
        assert out["tier"] == "tier2_balanced"


def test_openai_sse_streaming_endpoint(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ROUTEGUARD_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("ROUTEGUARD_API_KEY", raising=False)
    client = TestClient(app)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Classify support ticket urgency for invoice timeout"}],
            "stream": True,
            "use_cache": False,
        },
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")
    assert resp.headers.get("x-routeguard-tier") in {"tier1_fast", "tier2_balanced", "tier3_frontier"}
    body = resp.text
    assert "chat.completion.chunk" in body
    assert "data: [DONE]" in body


def test_routerbench_ood_evaluation_and_honest_pareto_metrics() -> None:
    res = run_routerbench_ood_eval()
    assert res["train_cases"] >= 700
    assert res["test_cases"] >= 350
    rec = res["recommended_operating_point"]
    assert 35.0 < rec["accuracy_pct"] < 100.0
    assert rec["cost_savings_vs_gpt4_pct"] > 30.0
    assert res["baselines"]["gpt-4-1106-preview"]["accuracy_pct"] < 100.0
