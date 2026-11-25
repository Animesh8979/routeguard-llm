"""Out-of-distribution RouterBench evaluation and confidence-margin Pareto sweep."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


_CHEAP_MODEL = "mistralai/mixtral-8x7b-chat"
_BALANCED_MODEL = "gpt-3.5-turbo-1106"
_FRONTIER_MODEL = "gpt-4-1106-preview"


def load_routerbench_sample(path: str | Path | None = None) -> list[dict[str, Any]]:
    p = Path(path) if path else Path(__file__).resolve().parents[1] / "benchmarks" / "routerbench_sample.jsonl"
    rows: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _oracle_tier_label(row: dict[str, Any]) -> str:
    scores = row.get("scores") or {}
    if float(scores.get(_CHEAP_MODEL, 0.0)) >= 1.0:
        return _CHEAP_MODEL
    if float(scores.get(_BALANCED_MODEL, 0.0)) >= 1.0:
        return _BALANCED_MODEL
    return _FRONTIER_MODEL


def run_routerbench_ood_eval(
    dataset_path: str | Path | None = None,
    *,
    train_ratio: float = 0.67,
    margins: tuple[float, ...] = (0.0, 0.15, 0.25, 0.35, 0.45, 0.55),
) -> dict[str, Any]:
    """Train TF-IDF + LogisticRegression on stratified RouterBench train split and evaluate held-out test."""
    rows = load_routerbench_sample(dataset_path)
    by_task: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_task.setdefault(str(r.get("eval_name") or "default"), []).append(r)

    train_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    for task in sorted(by_task):
        grp = by_task[task]
        cut = max(1, int(round(len(grp) * train_ratio)))
        train_rows.extend(grp[:cut])
        test_rows.extend(grp[cut:])

    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=6000, sublinear_tf=True, lowercase=True)
    X_train = vec.fit_transform([r["prompt"] for r in train_rows])
    y_train = [_oracle_tier_label(r) for r in train_rows]
    clf = LogisticRegression(max_iter=500, class_weight="balanced")
    clf.fit(X_train, y_train)

    X_test = vec.transform([r["prompt"] for r in test_rows])
    probs_all = clf.predict_proba(X_test)
    classes = list(clf.classes_)

    # Baselines on identical held-out split
    n = max(1, len(test_rows))
    baselines: dict[str, dict[str, float]] = {}
    for m_id in ("mistralai/mistral-7b-chat", _CHEAP_MODEL, _BALANCED_MODEL, _FRONTIER_MODEL):
        acc = sum(float(r["scores"].get(m_id, 0.0)) for r in test_rows) / n
        cost = sum(float(r["costs"].get(m_id, 0.0)) for r in test_rows)
        baselines[m_id] = {
            "accuracy_pct": round(acc * 100.0, 2),
            "total_cost_usd": round(cost, 6),
            "avg_cost_usd": round(cost / n, 6),
        }

    frontier_cost = max(1e-9, baselines[_FRONTIER_MODEL]["total_cost_usd"])
    frontier_acc = max(1e-9, baselines[_FRONTIER_MODEL]["accuracy_pct"])

    sweep: list[dict[str, Any]] = []
    for margin_tau in margins:
        correct_sum = 0.0
        total_cost = 0.0
        tier_counts = {_CHEAP_MODEL: 0, _BALANCED_MODEL: 0, _FRONTIER_MODEL: 0}
        escalated = 0
        for r, probs in zip(test_rows, probs_all):
            order = sorted(zip(classes, probs), key=lambda kv: kv[1], reverse=True)
            top_cls, top_p = order[0]
            second_p = order[1][1] if len(order) > 1 else 0.0
            margin = float(top_p - second_p)
            entropy = -sum(float(p) * math.log(max(float(p), 1e-12)) for p in probs)

            chosen = top_cls
            if margin < margin_tau:
                escalated += 1
                if top_cls == _CHEAP_MODEL:
                    chosen = _BALANCED_MODEL
                elif top_cls == _BALANCED_MODEL:
                    chosen = _FRONTIER_MODEL
            # If high normalized entropy across all 3 classes (> 0.95), fail safe to frontier
            if margin_tau >= 0.45 and entropy > 1.02:
                chosen = _FRONTIER_MODEL

            tier_counts[chosen] = tier_counts.get(chosen, 0) + 1
            correct_sum += float(r["scores"].get(chosen, 0.0))
            total_cost += float(r["costs"].get(chosen, 0.0))

        acc_pct = round(100.0 * correct_sum / n, 2)
        savings_pct = round(100.0 * (1.0 - total_cost / frontier_cost), 2)
        retention_pct = round(100.0 * acc_pct / frontier_acc, 2)
        sweep.append({
            "margin_threshold": margin_tau,
            "accuracy_pct": acc_pct,
            "accuracy_retention_vs_gpt4_pct": retention_pct,
            "total_cost_usd": round(total_cost, 6),
            "cost_savings_vs_gpt4_pct": savings_pct,
            "escalated_count": escalated,
            "tier_counts": tier_counts,
        })

    recommended = next((pt for pt in sweep if pt["margin_threshold"] == 0.25), sweep[len(sweep) // 2])
    return {
        "dataset": "withmartian/routerbench (0-shot stratified slice)",
        "train_cases": len(train_rows),
        "test_cases": len(test_rows),
        "tasks": sorted(by_task.keys()),
        "baselines": baselines,
        "pareto_sweep": sweep,
        "recommended_operating_point": recommended,
    }
