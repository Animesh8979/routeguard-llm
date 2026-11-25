"""Render Cost-vs-Accuracy Pareto Frontier PNG chart from live RouterBench OOD sweep."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def render_pareto_chart(ood_res: dict[str, Any], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    w, h = 920, 500
    img = Image.new("RGB", (w, h), "#0f172a")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    baselines = ood_res["baselines"]
    sweep = ood_res["pareto_sweep"]
    max_cost = max(b["total_cost_usd"] for b in baselines.values()) * 1.08
    min_acc = 25.0
    max_acc = 90.0

    draw.text((40, 22), "RouteGuard OOD Pareto Frontier on withmartian/routerbench (n=400 Held-Out)", fill="#f8fafc", font=font)
    draw.text((40, 44), f"Trained on {ood_res['train_cases']} prompts | Evaluated strictly out-of-sample across entropy-margin thresholds", fill="#94a3b8", font=font)

    left, top, right, bottom = 90, 90, 860, 415
    draw.rectangle((left, top, right, bottom), outline="#334155", width=2)

    def to_xy(cost: float, acc: float) -> tuple[int, int]:
        x = left + int((cost / max(1e-9, max_cost)) * (right - left))
        y = bottom - int(((max(min_acc, min(max_acc, acc)) - min_acc) / (max_acc - min_acc)) * (bottom - top))
        return x, y

    # Grid lines
    for pct in (30, 45, 60, 75, 85):
        _, gy = to_xy(0.0, pct)
        draw.line((left, gy, right, gy), fill="#1e293b", width=1)
        draw.text((34, gy - 6), f"{pct}%", fill="#94a3b8", font=font)

    # Pareto curve points
    pts = [to_xy(pt["total_cost_usd"], pt["accuracy_pct"]) for pt in sweep]
    for i in range(len(pts) - 1):
        draw.line((*pts[i], *pts[i + 1]), fill="#38bdf8", width=3)
    for pt, (px, py) in zip(sweep, pts):
        r = 6 if pt["margin_threshold"] == 0.25 else 4
        col = "#22c55e" if pt["margin_threshold"] == 0.25 else "#38bdf8"
        draw.ellipse((px - r, py - r, px + r, py + r), fill=col)
        if pt["margin_threshold"] in (0.0, 0.25, 0.55):
            draw.text((px + 8, py - 14), f"m={pt['margin_threshold']:.2f} ({pt['accuracy_pct']:.1f}%)", fill="#e2e8f0", font=font)

    # Baseline diamonds
    base_labels = {
        "mistralai/mistral-7b-chat": ("Mistral-7B", "#94a3b8"),
        "mistralai/mixtral-8x7b-chat": ("Mixtral-8x7B", "#a78bfa"),
        "gpt-3.5-turbo-1106": ("GPT-3.5-Turbo", "#f59e0b"),
        "gpt-4-1106-preview": ("Always GPT-4", "#fb7185"),
    }
    for m_key, (short_name, color) in base_labels.items():
        b = baselines[m_key]
        bx, by = to_xy(b["total_cost_usd"], b["accuracy_pct"])
        draw.rectangle((bx - 5, by - 5, bx + 5, by + 5), fill=color)
        dx = -95 if bx > 720 else 8
        draw.text((bx + dx, by + 6), f"{short_name} ({b['accuracy_pct']:.1f}%)", fill=color, font=font)

    rec = ood_res["recommended_operating_point"]
    draw.text((40, 445), f"Operating Point (margin=0.25): {rec['accuracy_pct']:.2f}% OOD Accuracy | {rec['cost_savings_vs_gpt4_pct']:.2f}% Cost Reduction vs GPT-4", fill="#22c55e", font=font)
    draw.text((40, 466), "X-axis: Held-Out Inference Cost (USD) -> Right is Higher Spend | Y-axis: RouterBench Exact Accuracy (%)", fill="#94a3b8", font=font)
    img.save(out_path)
    return out_path
