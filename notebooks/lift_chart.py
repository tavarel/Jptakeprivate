"""docs/img/lift_by_date.svg: lift@top10% per screening date (rules v0), reference line at 1×.

Run after `radar backtest`:  uv run python notebooks/lift_chart.py
Single series (no legend); lighter bars = 12-month label window not complete yet, also stated
in the subtitle and the hover title so color is not the only cue. Transparent background with
mid-gray text: readable on GitHub's light and dark themes.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from radar.config import get_settings

WIDTH, HEIGHT, LEFT, RIGHT, TOP, BOTTOM = 780, 300, 48, 72, 28, 64
Y_MAX = 3.5
INK, MUTED, GRID = "#6b7280", "#9ca3af", "#9ca3af"
BLUE, LIGHT = "#2a78d6", "#a9c8ee"  # dataviz default slot 1, and a lighter step of it


def y(value: float) -> float:
    plot_h = HEIGHT - TOP - BOTTOM
    return TOP + plot_h * (1 - value / Y_MAX)


def bar_path(x: float, width: float, value: float) -> str:
    """Bar with a 4px rounded data-end, square at the baseline."""
    top, base = y(value), y(0)
    r = min(4.0, (base - top) / 2)
    return (
        f"M{x:.1f},{base:.1f} V{top + r:.1f} Q{x:.1f},{top:.1f} {x + r:.1f},{top:.1f} "
        f"H{x + width - r:.1f} Q{x + width:.1f},{top:.1f} {x + width:.1f},{top + r:.1f} "
        f"V{base:.1f} Z"
    )


def chart(metrics: pd.DataFrame) -> str:
    slot = (WIDTH - LEFT - RIGHT) / len(metrics)
    bar = min(24.0, slot * 0.6)
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" font-family="-apple-system, Segoe UI, Helvetica, '
        'Arial, sans-serif" role="img" aria-label="Lift of the top 10% per screening date">'
    ]
    for v in (0, 1, 2, 3):
        out.append(
            f'<line x1="{LEFT}" x2="{WIDTH - RIGHT}" y1="{y(v):.1f}" y2="{y(v):.1f}" '
            f'stroke="{GRID}" stroke-opacity="0.25"/>'
        )
        out.append(
            f'<text x="{LEFT - 8}" y="{y(v) + 4:.1f}" font-size="11" fill="{INK}" '
            f'text-anchor="end">{v}×</text>'
        )
    for i, row in enumerate(metrics.itertuples()):
        x = LEFT + slot * i + (slot - bar) / 2
        note = "" if row.label_complete else " (window not complete)"
        color = BLUE if row.label_complete else LIGHT
        out.append(
            f'<path d="{bar_path(x, bar, float(row.lift_top))}" fill="{color}"><title>'
            f"{row.as_of}: lift {row.lift_top:.2f}×, {row.positives} targets{note}</title></path>"
        )
        label = pd.Timestamp(str(row.as_of)).strftime("%y-%m")
        out.append(
            f'<text x="{x + bar / 2:.1f}" y="{y(0) + 16:.1f}" font-size="10.5" fill="{INK}" '
            f'text-anchor="middle">{label}</text>'
        )
    out.append(
        f'<line x1="{LEFT}" x2="{WIDTH - RIGHT}" y1="{y(1):.1f}" y2="{y(1):.1f}" '
        f'stroke="{INK}" stroke-width="1.5" stroke-dasharray="5 4"/>'
    )
    out.append(
        f'<text x="{WIDTH - RIGHT + 8}" y="{y(1) + 4:.1f}" font-size="11" '
        f'fill="{INK}">chance</text>'
    )
    mean = metrics.loc[metrics["label_complete"], "lift_top"].mean()
    out.append(
        f'<text x="{LEFT}" y="{TOP - 10}" font-size="12" fill="{INK}">Mean of complete dates: '
        f"{mean:.2f}×  ·  lighter bars: 12-month window not complete yet</text>"
    )
    out.append(
        f'<text x="{LEFT}" y="{HEIGHT - 14}" font-size="11" fill="{MUTED}">Screening date '
        "(quarter-end, yy-mm). Lift = take-private rate in the top 10% of scores ÷ rate of all "
        "listed companies.</text>"
    )
    out.append("</svg>")
    return "\n".join(out)


if __name__ == "__main__":
    marts = get_settings().marts_dir
    metrics = pd.read_parquet(marts / "backtest_metrics_scoring_v0.parquet")
    metrics = metrics[metrics["lift_top"].notna()]  # a date with no target yet has no lift
    path = Path("docs/img/lift_by_date.svg")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(chart(metrics), encoding="utf-8")
    print(path)
