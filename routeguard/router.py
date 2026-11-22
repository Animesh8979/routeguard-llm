"""Sub-2ms Hybrid Complexity Cost Router, Tenant Budget Cap & Exponential Circuit Breaker."""
from __future__ import annotations

from dataclasses import dataclass, field
import re
import time
from typing import Any, Literal

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from routeguard.pricing import MODEL_TIERS

TierName = Literal["tier1_fast", "tier2_balanced", "tier3_frontier"]


class BudgetExceededError(RuntimeError):
    """Raised when a tenant hits 100% of their daily USD budget."""


@dataclass
class ProviderCircuitState:
    state: Literal["CLOSED", "OPEN", "HALF_OPEN"] = "CLOSED"
    consecutive_failures: int = 0
    open_count: int = 0
    opened_at: float = 0.0
    cooldown_sec: float = 0.0


class CircuitBreakerRegistry:
    """Exponential-backoff circuit breaker per model tier (`CLOSED -> OPEN -> HALF_OPEN`)."""

    def __init__(self, *, failure_threshold: int = 3, base_cooldown_sec: float = 10.0, max_cooldown_sec: float = 60.0) -> None:
        self.failure_threshold = failure_threshold
        self.base_cooldown_sec = base_cooldown_sec
        self.max_cooldown_sec = max_cooldown_sec
        self._states: dict[str, ProviderCircuitState] = {t: ProviderCircuitState() for t in MODEL_TIERS}

    def allow_request(self, tier: str, now: float | None = None) -> bool:
        st = self._states.setdefault(tier, ProviderCircuitState())
        t_now = time.monotonic() if now is None else now
        if st.state == "CLOSED":
            return True
        if st.state == "OPEN":
            if (t_now - st.opened_at) >= st.cooldown_sec:
                st.state = "HALF_OPEN"
                return True
            return False
        # HALF_OPEN allows one probe
        return True

    def record_success(self, tier: str) -> None:
        st = self._states.setdefault(tier, ProviderCircuitState())
        st.state = "CLOSED"
        st.consecutive_failures = 0

    def record_failure(self, tier: str, now: float | None = None) -> None:
        st = self._states.setdefault(tier, ProviderCircuitState())
        t_now = time.monotonic() if now is None else now
        st.consecutive_failures += 1
        if st.state == "HALF_OPEN" or st.consecutive_failures >= self.failure_threshold:
            st.state = "OPEN"
            st.open_count += 1
            st.opened_at = t_now
            st.cooldown_sec = min(self.max_cooldown_sec, self.base_cooldown_sec * (2 ** (st.open_count - 1)))

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return {
            tier: {
                "state": st.state,
                "consecutive_failures": st.consecutive_failures,
                "open_count": st.open_count,
                "cooldown_sec": st.cooldown_sec,
            }
            for tier, st in self._states.items()
        }


_TRAIN_SAMPLES: list[tuple[str, str]] = [
    # simple -> tier1_fast
    ("Classify customer support ticket priority and billing category", "simple"),
    ("Extract invoice number, vendor name, currency, and total from receipt text", "simple"),
    ("Tag sentiment positive neutral negative for product review", "simple"),
    ("Route password reset email to IT helpdesk queue JSON", "simple"),
    ("Parse shipping tracking ID and delivery status from SMS notification", "simple"),
    ("Summarize a 2 sentence customer refund confirmation", "simple"),
    ("Convert unstructured receipt line items into structured JSON schema", "simple"),
    ("Label bug report urgency low medium high", "simple"),
    ("Extract customer GSTIN and date from payment advice", "simple"),
    ("Identify language code and intent of inbound support chat", "simple"),
    # moderate -> tier2_balanced
    ("Write an ANSI SQL query using window functions to compute 7-day rolling retention", "moderate"),
    ("Generate parameterized PostgreSQL query with CTEs for monthly cohort revenue", "moderate"),
    ("Debug Python FastAPI pagination bug returning duplicate rows on tie-break sort", "moderate"),
    ("Compare index selectivity for composite key vs partial index on orders table", "moderate"),
    ("Refactor SQL join to deduplicate split settlement rows cleanly", "moderate"),
    ("Write pandas aggregation with groupwise z-score outlier filtering", "moderate"),
    ("Draft SQL migration adding NOT NULL constraint safely on active table", "moderate"),
    ("Explain N+1 query overhead in SQLAlchemy and show joinedload query fix", "moderate"),
    # complex -> tier3_frontier
    ("Perform multi-service root cause analysis across payment gateway timeouts deadlock telemetry and cache stampede", "complex"),
    ("Design distributed idempotency architecture balancing CAP trade-offs across two active-active regions", "complex"),
    ("Analyze conflicting financial audit logs, webhook retries, and race conditions to formulate reconciliation invariants", "complex"),
    ("Evaluate latency SLA failure propagation across message broker backpressure, thread pools, and downstream locks", "complex"),
    ("Derive formal state machine transitions with linearizability proofs under network partitions", "complex"),
    ("Synthesize multi-quarter unit economics sensitivity model balancing CAC payback, churn elasticity, and gross margin", "complex"),
    ("Audit subtle concurrency race between WAL checkpointing, connection pooling, and partial commit failure", "complex"),
    ("Propose phased zero-downtime database sharding migration with dual-write reconciliation and rollback triggers", "complex"),
]


class HybridComplexityRouter:
    """Combines TF-IDF + LogisticRegression complexity scoring with deterministic structural guards."""

    def __init__(self, breakers: CircuitBreakerRegistry | None = None) -> None:
        self.breakers = breakers or CircuitBreakerRegistry()
        self.tenant_daily_caps: dict[str, float] = {}
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
        texts = [t for t, _ in _TRAIN_SAMPLES]
        labels = [y for _, y in _TRAIN_SAMPLES]
        X = self.vectorizer.fit_transform(texts)
        self.clf = LogisticRegression(max_iter=300)
        self.clf.fit(X, labels)

    def set_tenant_budget(self, tenant_id: str, daily_cap_usd: float) -> None:
        self.tenant_daily_caps[tenant_id] = float(daily_cap_usd)

    def _raw_complexity(self, prompt: str, has_schema: bool = False) -> tuple[str, dict[str, Any]]:
        low = (prompt or "").lower()
        t0 = time.perf_counter()

        complex_rx = re.compile(
            r"\b(?:root[\s-]cause|architecture|trade[\s-]?offs?|distributed|deadlock|race\s+condition|"
            r"linearizability|partition|multi[\s-]region|backpressure|unit\s+economics|migration\s+plan|"
            r"invariant|retry\s+storm|cascading)\b"
        )
        sql_code_rx = re.compile(
            r"\b(?:sql|cte|window\s+function|postgres|sqlite|query|index|pandas|fastapi|python|debug|refactor|schema\s+migration)\b"
        )
        simple_rx = re.compile(
            r"\b(?:extract|receipt|invoice|classify|triage|ticket|sentiment|route|tag|parse)\b"
        )

        # Evaluate classifier probability
        probs = self.clf.predict_proba(self.vectorizer.transform([low]))[0]
        class_probs = {cls: float(p) for cls, p in zip(self.clf.classes_, probs)}
        pred = max(class_probs, key=class_probs.get)

        word_count = len(low.split())
        if complex_rx.search(low) or (word_count >= 55 and "compare" in low and "trade" in low):
            label = "complex"
        elif sql_code_rx.search(low) and not simple_rx.search(low):
            label = "moderate"
        elif has_schema and simple_rx.search(low) and word_count < 85:
            label = "simple"
        else:
            label = pred

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        return label, {"complexity_label": label, "class_probs": {k: round(v, 3) for k, v in class_probs.items()}, "router_ms": elapsed_ms}

    def select_tier(
        self,
        prompt: str,
        *,
        tenant_id: str = "default",
        tenant_spent_usd: float = 0.0,
        has_schema: bool = False,
        force_tier: str | None = None,
    ) -> dict[str, Any]:
        cap = self.tenant_daily_caps.get(tenant_id)
        if cap is not None and tenant_spent_usd >= cap:
            raise BudgetExceededError(f"Tenant '{tenant_id}' reached daily USD cap ({tenant_spent_usd:.6f} >= {cap:.6f})")

        label, meta = self._raw_complexity(prompt, has_schema=has_schema)
        if force_tier in MODEL_TIERS:
            desired: TierName = force_tier  # type: ignore[assignment]
        elif label == "complex":
            desired = "tier3_frontier"
        elif label == "moderate":
            desired = "tier2_balanced"
        else:
            desired = "tier1_fast"

        # Soft budget downgrade at >= 85% daily cap
        budget_downgraded = False
        if cap is not None and cap > 0 and (tenant_spent_usd / cap) >= 0.85 and desired == "tier3_frontier":
            desired = "tier2_balanced"
            budget_downgraded = True

        # Circuit breaker failover order
        failover_chains: dict[str, list[TierName]] = {
            "tier1_fast": ["tier1_fast", "tier2_balanced", "tier3_frontier"],
            "tier2_balanced": ["tier2_balanced", "tier1_fast", "tier3_frontier"],
            "tier3_frontier": ["tier3_frontier", "tier2_balanced", "tier1_fast"],
        }
        chosen = desired
        failover_used = False
        for candidate in failover_chains[desired]:
            if self.breakers.allow_request(candidate):
                chosen = candidate
                failover_used = (candidate != desired)
                break

        spec = MODEL_TIERS[chosen]
        return {
            "tier": chosen,
            "desired_tier": desired,
            "model_id": spec.model_id,
            "failover_used": failover_used,
            "budget_downgraded": budget_downgraded,
            **meta,
        }
