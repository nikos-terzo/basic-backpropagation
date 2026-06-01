from __future__ import annotations

from html import escape
from pathlib import Path


VALIDATION_COST_GRAPH_PATH = Path("validation-cost.svg")


def save_validation_cost_graph(
    validation_history: dict[str, list[tuple[int, float]]],
) -> None:
    width = 960
    height = 560
    margin_left = 72
    margin_right = 28
    margin_top = 42
    margin_bottom = 64
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    all_points = [
        point for points in validation_history.values() for point in points
    ]
    max_iteration = max(iteration for iteration, _cost in all_points)
    max_cost = max(cost for _iteration, cost in all_points)
    min_cost = min(cost for _iteration, cost in all_points)
    cost_range = max_cost - min_cost

    def x_position(iteration: int) -> float:
        return margin_left + (iteration / max_iteration) * plot_width

    def y_position(cost: float) -> float:
        if cost_range == 0:
            return margin_top + plot_height / 2
        return margin_top + ((max_cost - cost) / cost_range) * plot_height

    colors = ["#1d4f73", "#9b4d21", "#2f6b47", "#6f2f43", "#4e4675"]
    line_parts: list[str] = []
    legend_parts: list[str] = []

    for index, (model_name, points) in enumerate(validation_history.items()):
        color = colors[index % len(colors)]
        polyline_points = " ".join(
            f"{x_position(iteration):.1f},{y_position(cost):.1f}"
            for iteration, cost in points
        )
        line_parts.append(
            f'<polyline points="{polyline_points}" fill="none" '
            + f'stroke="{color}" stroke-width="2.5" />'
        )
        for iteration, cost in points:
            line_parts.append(
                f'<circle cx="{x_position(iteration):.1f}" '
                + f'cy="{y_position(cost):.1f}" r="3" fill="{color}" />'
            )

        legend_y = margin_top + 22 * index
        legend_parts.append(
            f'<rect x="{margin_left + 14}" y="{legend_y - 9}" '
            + f'width="14" height="3" fill="{color}" />'
            + f'<text x="{margin_left + 36}" y="{legend_y - 4}" '
            + 'font-size="13" fill="#202124">'
            + f"{escape(model_name)}</text>"
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#f7f7f3" />
  <text x="{margin_left}" y="26" font-size="20" font-family="Arial, sans-serif" fill="#202124">Validation cost over training</text>

  <line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#62635f" />
  <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#62635f" />

  <text x="{margin_left + plot_width / 2 - 54:.1f}" y="{height - 18}" font-size="14" font-family="Arial, sans-serif" fill="#62635f">Mini-batch iteration</text>
  <text x="18" y="{margin_top + plot_height / 2 + 48:.1f}" font-size="14" font-family="Arial, sans-serif" fill="#62635f" transform="rotate(-90 18 {margin_top + plot_height / 2 + 48:.1f})">Validation cost</text>

  <text x="{margin_left - 54}" y="{margin_top + 4}" font-size="12" font-family="Arial, sans-serif" fill="#62635f">{max_cost:.3f}</text>
  <text x="{margin_left - 54}" y="{margin_top + plot_height + 4}" font-size="12" font-family="Arial, sans-serif" fill="#62635f">{min_cost:.3f}</text>
  <text x="{margin_left - 8}" y="{margin_top + plot_height + 22}" font-size="12" font-family="Arial, sans-serif" fill="#62635f" text-anchor="end">0</text>
  <text x="{margin_left + plot_width}" y="{margin_top + plot_height + 22}" font-size="12" font-family="Arial, sans-serif" fill="#62635f" text-anchor="middle">{max_iteration}</text>

  {''.join(line_parts)}
  {''.join(legend_parts)}
</svg>
"""
    _ = VALIDATION_COST_GRAPH_PATH.write_text(svg, encoding="utf-8")
