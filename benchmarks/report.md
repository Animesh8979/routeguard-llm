# RouteGuard Benchmark Results (RouterBench OOD + Guardrail CI)

| Suite / Metric | Measured Value |
|---|---:|
| Public OOD Dataset (`withmartian/routerbench`) | 800 train / 400 held-out test prompts |
| RouteGuard OOD Operating Point (`margin=0.25`) | 66.50% accuracy vs Always-GPT-4 79.50% |
| Held-Out Cost Reduction vs Always-GPT-4 | 49.78% (`$0.6656` vs `$1.3252`) |
| Deterministic Regression Suite (+ Cache Probes) | 60 (+15 repeat probes) |
| Proxy Regression Cost Reduction | 76.68% |
| Schema Validity (Cold Shot -> Self-Healed) | 50.00% -> 100.00% (16 repairs) |
| Prompt-Injection Block / PII Redaction | 100.00% / 100.00% |
| Repeat Semantic Cache Hit Rate | 14/15 |
| Proxy Overhead Latency (p50 / p95) | 5.3 ms / 20.84 ms |
| Circuit-Breaker Failover Verified | True |
