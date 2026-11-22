"""Interactive Streamlit Observability & Cost-Routing Cockpit for RouteGuard."""
from __future__ import annotations

import os
from pathlib import Path
import pandas as pd
import streamlit as st

from routeguard.cache import SemanticCache
from routeguard.eval_harness import GatewayRuntime, handle_request, run_eval_suite
from routeguard.ledger import TelemetryLedger
from routeguard.providers import DeterministicReplayProvider
from routeguard.router import HybridComplexityRouter

st.set_page_config(page_title="RouteGuard LLM Observability Cockpit", layout="wide")
st.title("RouteGuard - LLM Cost-Routing & Self-Healing Schema Gateway")

data_dir = Path(os.environ.get("ROUTEGUARD_DATA_DIR", "data_runtime/dashboard"))
data_dir.mkdir(parents=True, exist_ok=True)

if st.sidebar.button("Run 60-Case Regression Eval Suite") or not (data_dir / "ledger.db").exists():
    with st.spinner("Running 60-case evaluation suite..."):
        report = run_eval_suite(data_dir)
        st.session_state["last_eval"] = report

ledger = TelemetryLedger(data_dir / "ledger.db")
m = ledger.get_metrics()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Requests Logged", f"{m['total_requests']}")
c2.metric("Cost Savings vs Frontier", f"{m['cost_savings_pct']:.1f}%")
c3.metric("Schema Validity (Healed)", f"{m['schema_validity_pct']:.1f}%")
c4.metric("Injections Blocked", f"{m['injections_blocked']}")

if m["tier_distribution"]:
    st.subheader("Model Tier Routing Breakdown")
    tier_df = pd.DataFrame(
        [{"tier": k, "requests": v} for k, v in m["tier_distribution"].items()]
    ).set_index("tier")
    st.bar_chart(tier_df)

st.subheader("Live Prompt & Schema Tester")
with st.form("live_test"):
    prompt_in = st.text_area("Prompt", "Extract invoice fields into JSON. Invoice INV-2026-909 | Vendor: RazorPay | Total: Rs. 2,400.50")
    schema_in = st.selectbox("Schema Validation", ["None", "InvoiceExtraction", "SupportTicketTriage"], index=1)
    submitted = st.form_submit_button("Route & Execute")
    if submitted and prompt_in.strip():
        rt = GatewayRuntime(
            cache=SemanticCache(data_dir / "cache.db"),
            ledger=ledger,
            router=HybridComplexityRouter(),
            provider=DeterministicReplayProvider(),
        )
        res = handle_request(
            rt,
            prompt=prompt_in,
            schema_name=None if schema_in == "None" else schema_in,
        )
        st.json(res)
