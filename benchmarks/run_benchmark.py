"""Execute RouteGuard 60-case regression & circuit-breaker benchmark and emit results.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.pareto_chart import render_pareto_chart
from routeguard.cache import SemanticCache
from routeguard.eval_harness import GatewayRuntime, handle_request, run_eval_suite
from routeguard.ledger import TelemetryLedger
from routeguard.providers import DeterministicReplayProvider
from routeguard.router import CircuitBreakerRegistry, HybridComplexityRouter
from routeguard.routerbench_eval import run_routerbench_ood_eval


def run_benchmark(out_dir: str | Path = "data_runtime/bench_run") -> dict:
    root = Path(__file__).resolve().parents[1]
    out = Path(out_dir)
    if not out.is_absolute():
        out = root / out
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    eval_res = run_eval_suite(out)
    elapsed_sec = round(time.perf_counter() - t0, 3)

    # Verify automatic circuit-breaker failover when Tier 1 returns HTTP 429
    cb_dir = out / "failover_probe"
    cb_reg = CircuitBreakerRegistry(failure_threshold=1, base_cooldown_sec=5.0)
    cb_reg.record_failure("tier1_fast")  # opens tier1_fast
    rt = GatewayRuntime(
        cache=SemanticCache(cb_dir / "cache.db"),
        ledger=TelemetryLedger(cb_dir / "ledger.db"),
        router=HybridComplexityRouter(breakers=cb_reg),
        provider=DeterministicReplayProvider(fault_tiers={"tier1_fast": 429}),
    )
    failover_out = handle_request(
        rt,
        prompt="Classify ticket: billing duplicate charge refund urgent",
        schema_name="SupportTicketTriage",
        use_cache=False,
    )

    ood_eval = run_routerbench_ood_eval()
    pareto_png = render_pareto_chart(ood_eval, root / "docs" / "assets" / "routeguard_pareto.png")
    rec_pt = ood_eval["recommended_operating_point"]

    lm = eval_res["ledger_metrics"]
    results = {
        "eval_cases_total": eval_res["eval_cases_total"],
        "repeat_queries_tested": eval_res["repeat_queries_tested"],
        "total_requests_logged": lm["total_requests"],
        "cost_savings_vs_frontier_pct": lm["cost_savings_pct"],
        "actual_cost_usd": lm["actual_cost_usd"],
        "baseline_frontier_cost_usd": lm["baseline_frontier_cost_usd"],
        "raw_first_shot_valid_pct": eval_res["raw_first_shot_valid_pct"],
        "healed_schema_valid_pct": eval_res["healed_schema_valid_pct"],
        "repairs_executed": eval_res["repairs_executed"],
        "routing_accuracy_pct": eval_res["routing_accuracy_pct"],
        "injection_block_rate_pct": eval_res["injection_block_rate_pct"],
        "pii_redaction_rate_pct": eval_res["pii_redaction_rate_pct"],
        "repeat_cache_hits": eval_res["repeat_cache_hits"],
        "p50_latency_ms": lm["p50_latency_ms"],
        "p95_latency_ms": lm["p95_latency_ms"],
        "circuit_breaker_failover_verified": bool(failover_out["failover_used"] and failover_out["schema_valid"] and failover_out["tier_selected"] == "tier2_balanced"),
        "runtime_seconds": elapsed_sec,
        "tier_distribution": lm["tier_distribution"],
        "routerbench_ood": ood_eval,
        "pareto_png": str(pareto_png),
    }

    bench_dir = root / "benchmarks"
    (bench_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    gpt4_acc = ood_eval["baselines"]["gpt-4-1106-preview"]["accuracy_pct"]
    md = [
        "# RouteGuard Benchmark Results (RouterBench OOD + Guardrail CI)",
        "",
        "| Suite / Metric | Measured Value |",
        "|---|---:|",
        f"| Public OOD Dataset (`withmartian/routerbench`) | {ood_eval['train_cases']} train / {ood_eval['test_cases']} held-out test prompts |",
        f"| RouteGuard OOD Operating Point (`margin=0.25`) | {rec_pt['accuracy_pct']:.2f}% accuracy vs Always-GPT-4 {gpt4_acc:.2f}% |",
        f"| Held-Out Cost Reduction vs Always-GPT-4 | {rec_pt['cost_savings_vs_gpt4_pct']:.2f}% (`${rec_pt['total_cost_usd']:.4f}` vs `${ood_eval['baselines']['gpt-4-1106-preview']['total_cost_usd']:.4f}`) |",
        f"| Deterministic Regression Suite (+ Cache Probes) | {results['eval_cases_total']} (+{results['repeat_queries_tested']} repeat probes) |",
        f"| Proxy Regression Cost Reduction | {results['cost_savings_vs_frontier_pct']:.2f}% |",
        f"| Schema Validity (Cold Shot -> Self-Healed) | {results['raw_first_shot_valid_pct']:.2f}% -> {results['healed_schema_valid_pct']:.2f}% ({results['repairs_executed']} repairs) |",
        f"| Prompt-Injection Block / PII Redaction | {results['injection_block_rate_pct']:.2f}% / {results['pii_redaction_rate_pct']:.2f}% |",
        f"| Repeat Semantic Cache Hit Rate | {results['repeat_cache_hits']}/{results['repeat_queries_tested']} |",
        f"| Proxy Overhead Latency (p50 / p95) | {results['p50_latency_ms']} ms / {results['p95_latency_ms']} ms |",
        f"| Circuit-Breaker Failover Verified | {results['circuit_breaker_failover_verified']} |",
    ]
    (bench_dir / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data_runtime/bench_run")
    args = parser.parse_args()
    res = run_benchmark(args.out_dir)
    print(json.dumps(res, indent=2))
