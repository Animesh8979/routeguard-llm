# RouteGuard Benchmark Results (60-Case Golden Eval + Cache Pass)

| Metric | Measured Value |
|---|---:|
| Golden Eval Suite Cases | 60 (+15 repeat cache probes) |
| Cost Reduction vs Always-Frontier | 76.68% |
| Routing Tier Accuracy | 100.00% |
| Schema Validity (Raw Cold Shot -> Healed) | 50.00% -> 100.00% |
| Self-Healing Repairs Executed | 16 |
| Prompt-Injection Block Rate | 100.00% |
| PII Redaction Rate | 100.00% |
| Repeat Semantic Cache Hit Rate | 14/15 |
| Proxy Overhead Latency (p50 / p95) | 11.34 ms / 33.14 ms |
| Circuit-Breaker Failover Verified | True |
