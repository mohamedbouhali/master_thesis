#!/usr/bin/env python3
"""Generate publication-quality visualizations for reward score results."""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Any, Iterable

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "reward_scores" / "gemma-4-31B-it" / "all_results.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "plots"
COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "gray": "#6B7280",
}
EXPECTED_FILES = (
    "reward_distribution.png",
    "component_distributions.png",
    "metric_correlation_heatmap.png",
    "reward_vs_key_metrics.png",
    "component_summary.png",
    "reward_by_agentbench_result.png",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot distributions and relationships from generated reward scores."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Reward results JSON (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for PNG files (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


def numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def pretty(name: str) -> str:
    replacements = {"llm": "LLM", "ms": "ms"}
    return " ".join(replacements.get(part, part.capitalize()) for part in name.split("_"))


def load_results(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not read reward data from {path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise SystemExit("Input JSON must be an object containing a 'results' list.")

    rows = []
    for item in payload["results"]:
        if not isinstance(item, dict) or item.get("status") not in (None, "completed"):
            continue
        reward = numeric(item.get("reward"))
        if reward is None:
            continue
        row = dict(item)
        row["reward"] = reward
        rows.append(row)
    if not rows:
        raise SystemExit("No completed rows with finite numeric rewards were found.")
    return rows


def discover_fields(rows: list[dict[str, Any]], section: str) -> list[str]:
    names = set()
    for row in rows:
        values = row.get(section)
        if isinstance(values, dict):
            names.update(key for key, value in values.items() if numeric(value) is not None)
    return sorted(names)


def series(rows: list[dict[str, Any]], section: str, field: str) -> np.ndarray:
    values = []
    for row in rows:
        container = row if section == "" else row.get(section, {})
        value = numeric(container.get(field)) if isinstance(container, dict) else None
        if value is not None:
            values.append(value)
    return np.asarray(values, dtype=float)


def aligned_series(
    rows: list[dict[str, Any]], section: str, field: str
) -> tuple[np.ndarray, np.ndarray, list[bool | None]]:
    rewards, values, labels = [], [], []
    for row in rows:
        container = row if section == "" else row.get(section, {})
        value = numeric(container.get(field)) if isinstance(container, dict) else None
        reward = numeric(row.get("reward"))
        if value is not None and reward is not None:
            values.append(value)
            rewards.append(reward)
            labels.append(row.get("passed"))
    return np.asarray(values), np.asarray(rewards), labels


def load_pass_labels(rows: list[dict[str, Any]], input_path: Path) -> tuple[int, int] | None:
    labels_path = input_path.with_name("input.json")
    if not labels_path.is_file():
        warnings.warn(f"No sibling input.json found; pass/fail plot will be omitted: {labels_path}")
        return None
    try:
        source = json.loads(labels_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        warnings.warn(f"Could not load pass/fail labels: {exc}")
        return None
    if not isinstance(source, list):
        warnings.warn("Sibling input.json is not a list; pass/fail plot will be omitted.")
        return None

    by_session: dict[str, bool] = {}
    ambiguous = set()
    by_index: dict[int, bool] = {}
    for item in source:
        if not isinstance(item, dict):
            continue
        label = item.get("agentbench", {}).get("result") if isinstance(item.get("agentbench"), dict) else None
        if not isinstance(label, bool):
            continue
        session_id = item.get("session_id")
        if session_id is not None:
            key = str(session_id)
            if key in by_session and by_session[key] != label:
                ambiguous.add(key)
            by_session[key] = label
        index = item.get("agentbench", {}).get("index")
        if isinstance(index, int) and not isinstance(index, bool):
            by_index[index] = label

    matched = []
    for row in rows:
        key = str(row.get("session_id")) if row.get("session_id") is not None else None
        label = by_session.get(key) if key is not None and key not in ambiguous else None
        if label is None:
            index = row.get("dataset_index")
            if isinstance(index, int) and not isinstance(index, bool):
                label = by_index.get(index)
        row["passed"] = label
        if isinstance(label, bool):
            matched.append(label)
    if not matched or len(matched) != len(rows):
        warnings.warn(
            f"Pass/fail labels matched {len(matched)}/{len(rows)} rows; stratified plots require a complete unambiguous join."
        )
        for row in rows:
            row["passed"] = None
        return None
    return sum(matched), len(matched) - sum(matched)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 240,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linestyle": "--",
            "legend.frameon": False,
        }
    )


def save(fig: plt.Figure, output: Path) -> None:
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Created {output}")


def kde(values: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if len(values) < 5 or np.ptp(values) <= 0:
        return None
    std = float(np.std(values, ddof=1))
    iqr = float(np.subtract(*np.percentile(values, [75, 25])))
    scale_candidates = [candidate for candidate in (std, iqr / 1.34) if candidate > 0]
    if not scale_candidates:
        return None
    bandwidth = 0.9 * min(scale_candidates) * len(values) ** (-0.2)
    if not math.isfinite(bandwidth) or bandwidth <= 0:
        return None
    margin = max(np.ptp(values) * 0.08, bandwidth * 2)
    grid = np.linspace(values.min() - margin, values.max() + margin, 300)
    density = np.exp(-0.5 * ((grid[:, None] - values[None, :]) / bandwidth) ** 2).sum(axis=1)
    density /= len(values) * bandwidth * math.sqrt(2 * math.pi)
    return grid, density


def add_histogram(ax: plt.Axes, values: np.ndarray, color: str, title: str) -> None:
    bins = max(5, min(20, int(math.sqrt(len(values)) * 1.5))) if len(values) > 1 else 1
    ax.hist(values, bins=bins, density=True, color=color, alpha=0.68, edgecolor="white")
    estimate = kde(values)
    if estimate is not None:
        ax.plot(*estimate, color=COLORS["red"], linewidth=2, label="KDE")
    elif np.ptp(values) == 0:
        ax.text(0.5, 0.82, "Constant values; KDE omitted", ha="center", transform=ax.transAxes)
    else:
        ax.text(0.5, 0.82, "Too few values for KDE", ha="center", transform=ax.transAxes)
    mean, median = float(np.mean(values)), float(np.median(values))
    ax.axvline(mean, color=COLORS["orange"], linewidth=2, label=f"Mean {mean:.3g}")
    ax.axvline(median, color=COLORS["green"], linewidth=2, linestyle="--", label=f"Median {median:.3g}")
    ax.set_title(f"{title} (n={len(values)})")
    ax.set_ylabel("Density")


def plot_reward_distribution(rows: list[dict[str, Any]], output: Path) -> None:
    values = series(rows, "", "reward")
    fig, ax = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    add_histogram(ax, values, COLORS["blue"], "Overall reward distribution")
    ax.set_xlabel("Reward score")
    ax.legend(loc="best")
    save(fig, output / "reward_distribution.png")


def plot_component_distributions(
    rows: list[dict[str, Any]], fields: list[str], output: Path
) -> None:
    cols = 2
    rows_count = math.ceil(len(fields) / cols)
    fig, axes = plt.subplots(rows_count, cols, figsize=(12, 3.35 * rows_count), constrained_layout=True)
    axes_flat = np.asarray(axes).reshape(-1)
    for ax, field in zip(axes_flat, fields):
        values = series(rows, "components", field)
        add_histogram(ax, values, COLORS["blue"], pretty(field))
        ax.set_xlabel("Component score")
        ax.legend(fontsize=8, loc="best")
    for ax in axes_flat[len(fields) :]:
        ax.remove()
    fig.suptitle("Reward component distributions", fontsize=16)
    save(fig, output / "component_distributions.png")


def correlation_matrix(columns: list[np.ndarray]) -> np.ndarray:
    count = len(columns)
    result = np.full((count, count), np.nan)
    for i in range(count):
        for j in range(count):
            mask = np.isfinite(columns[i]) & np.isfinite(columns[j])
            x, y = columns[i][mask], columns[j][mask]
            if len(x) >= 2 and np.ptp(x) > 0 and np.ptp(y) > 0:
                result[i, j] = np.corrcoef(x, y)[0, 1]
    return result


def plot_correlation(
    rows: list[dict[str, Any]], components: list[str], metrics: list[str], output: Path
) -> None:
    candidates = [("", "reward")] + [("components", field) for field in components] + [
        ("metrics", field) for field in metrics
    ]
    labels, columns = [], []
    for section, field in candidates:
        values = np.asarray(
            [
                numeric((row if not section else row.get(section, {})).get(field))
                if isinstance(row if not section else row.get(section, {}), dict)
                else None
                for row in rows
            ],
            dtype=object,
        )
        numeric_values = np.asarray([np.nan if value is None else value for value in values], dtype=float)
        finite = numeric_values[np.isfinite(numeric_values)]
        if len(finite) >= 2 and np.ptp(finite) > 0:
            labels.append(pretty(field))
            columns.append(numeric_values)
    corr = correlation_matrix(columns)
    size = max(9.5, 0.7 * len(labels))
    fig, ax = plt.subplots(figsize=(size, size), constrained_layout=True)
    image = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(labels)), labels, rotation=55, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.grid(False)
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = corr[i, j]
            if math.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(value) > 0.55 else "black")
    fig.colorbar(image, ax=ax, shrink=0.78, label="Pearson correlation")
    ax.set_title(f"Reward, component, and metric correlations (pairwise n≤{len(rows)})")
    save(fig, output / "metric_correlation_heatmap.png")


def choose_key_metrics(
    rows: list[dict[str, Any]], components: list[str], metrics: list[str]
) -> list[tuple[str, str]]:
    requested = [
        ("metrics", "total_tokens"),
        ("metrics", "num_tool_calls"),
        ("metrics", "duration_ms"),
        ("components", "llm_judge"),
    ]
    available = {"components": set(components), "metrics": set(metrics)}
    chosen = [
        item
        for item in requested
        if item[1] in available[item[0]] and np.ptp(series(rows, item[0], item[1])) > 0
    ]
    if len(chosen) < 4:
        fallbacks = [("components", field) for field in components] + [
            ("metrics", field) for field in metrics
        ]
        for item in fallbacks:
            if item in chosen or np.ptp(series(rows, item[0], item[1])) == 0:
                continue
            chosen.append(item)
            if len(chosen) == 4:
                break
    return chosen[:4]


def plot_scatter(
    rows: list[dict[str, Any]], components: list[str], metrics: list[str], output: Path
) -> None:
    chosen = choose_key_metrics(rows, components, metrics)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for ax, (section, field) in zip(axes.flat, chosen):
        x, y, labels = aligned_series(rows, section, field)
        passed = np.asarray([label is True for label in labels])
        failed = np.asarray([label is False for label in labels])
        unknown = ~(passed | failed)
        if passed.any():
            ax.scatter(x[passed], y[passed], s=28, alpha=0.7, color=COLORS["blue"], label="Passed")
        if failed.any():
            ax.scatter(x[failed], y[failed], s=28, alpha=0.7, color=COLORS["orange"], marker="^", label="Failed")
        if unknown.any() or not labels:
            mask = unknown if len(unknown) else np.ones(len(x), dtype=bool)
            ax.scatter(x[mask], y[mask], s=28, alpha=0.65, color=COLORS["gray"], label="All samples")
        use_log_x = field in {"total_tokens", "duration_ms"} and np.all(x > 0) and np.ptp(x) > 0
        if len(x) >= 3 and np.ptp(x) > 0:
            fit_x = np.log10(x) if use_log_x else x
            coefficients = np.polyfit(fit_x, y, 1)
            line_x = np.geomspace(x.min(), x.max(), 100) if use_log_x else np.linspace(x.min(), x.max(), 100)
            line_y = np.polyval(coefficients, np.log10(line_x) if use_log_x else line_x)
            trend_label = "Linear trend in log(x)" if use_log_x else "Linear trend"
            ax.plot(line_x, line_y, color=COLORS["red"], linewidth=2, label=trend_label)
        if use_log_x:
            ax.set_xscale("log")
        unit = " (ms, log scale)" if field == "duration_ms" else " (log scale)" if field == "total_tokens" else ""
        ax.set_xlabel(pretty(field) + unit)
        ax.set_ylabel("Reward score")
        ax.set_title(f"Reward vs {pretty(field)} (n={len(x)})")
        ax.legend(fontsize=8)
    for ax in axes.flat[len(chosen) :]:
        ax.remove()
    fig.suptitle("Reward relationships with key metrics", fontsize=16)
    save(fig, output / "reward_vs_key_metrics.png")


def plot_component_summary(
    rows: list[dict[str, Any]], fields: list[str], output: Path
) -> None:
    values = [series(rows, "components", field) for field in fields]
    means = np.asarray([np.mean(item) for item in values])
    stds = np.asarray([np.std(item, ddof=1) if len(item) > 1 else 0 for item in values])
    medians = np.asarray([np.median(item) for item in values])
    positions = np.arange(len(fields))
    fig, ax = plt.subplots(figsize=(9.5, 5.8), constrained_layout=True)
    ax.errorbar(means, positions, xerr=stds, fmt="o", color=COLORS["blue"], ecolor=COLORS["gray"],
                capsize=4, markersize=7, label="Mean ± 1 SD")
    ax.scatter(medians, positions, marker="D", s=38, color=COLORS["orange"], label="Median", zorder=3)
    for pos, item in zip(positions, values):
        if np.ptp(item) == 0:
            ax.annotate("constant", (item[0], pos), xytext=(7, 6), textcoords="offset points", fontsize=8)
    ax.set_yticks(positions, [pretty(field) for field in fields])
    ax.invert_yaxis()
    ax.set_xlabel("Observed component score (unbounded to preserve actual values)")
    ax.set_title(f"Component score summary (n≤{len(rows)})")
    ax.legend()
    save(fig, output / "component_summary.png")


def plot_reward_by_result(rows: list[dict[str, Any]], output: Path) -> bool:
    passed = np.asarray([row["reward"] for row in rows if row.get("passed") is True])
    failed = np.asarray([row["reward"] for row in rows if row.get("passed") is False])
    if not len(passed) or not len(failed):
        warnings.warn("Both pass and fail groups are required; stratified reward plot was omitted.")
        return False
    data = [failed, passed]
    fig, ax = plt.subplots(figsize=(7.8, 5.4), constrained_layout=True)
    violin = ax.violinplot(data, positions=[1, 2], widths=0.72, showmeans=False, showmedians=False)
    for body, color in zip(violin["bodies"], [COLORS["orange"], COLORS["blue"]]):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.3)
    for key in ("cbars", "cmins", "cmaxes"):
        violin[key].set_color(COLORS["gray"])
    box = ax.boxplot(data, positions=[1, 2], widths=0.28, patch_artist=True, showfliers=False,
                     medianprops={"color": "black", "linewidth": 2})
    for patch, color in zip(box["boxes"], [COLORS["orange"], COLORS["blue"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    rng = np.random.default_rng(20250308)
    for position, values, color in zip([1, 2], data, [COLORS["orange"], COLORS["blue"]]):
        jitter = rng.uniform(-0.08, 0.08, len(values))
        ax.scatter(position + jitter, values, s=15, color=color, alpha=0.45, edgecolors="none")
        ax.text(
            position,
            0.97,
            f"n={len(values)}\nmean={np.mean(values):.3f}\nmedian={np.median(values):.3f}",
            ha="center",
            va="top",
            fontsize=9,
            transform=ax.get_xaxis_transform(),
            bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 2},
        )
    ax.set_xticks([1, 2], ["Failed", "Passed"])
    ax.set_ylabel("Reward score")
    ax.set_title("Reward distribution by AgentBench result")
    ax.legend(handles=[Line2D([], [], color="black", linewidth=2, label="Box center: median")], loc="lower right")
    save(fig, output / "reward_by_agentbench_result.png")
    return True


def main() -> None:
    args = parse_args()
    try:
        global plt, Line2D
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as exc:
        raise SystemExit(
            "Matplotlib is required to generate plots. Install the pinned project requirements first."
        ) from exc
    input_path = args.input.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    rows = load_results(input_path)
    components = discover_fields(rows, "components")
    metrics = discover_fields(rows, "metrics")
    if not components:
        raise SystemExit("No finite numeric component fields were found.")
    output.mkdir(parents=True, exist_ok=True)
    configure_style()
    label_counts = load_pass_labels(rows, input_path)

    plot_reward_distribution(rows, output)
    plot_component_distributions(rows, components, output)
    plot_correlation(rows, components, metrics, output)
    plot_scatter(rows, components, metrics, output)
    plot_component_summary(rows, components, output)
    stratified = plot_reward_by_result(rows, output)

    rewards = series(rows, "", "reward")
    print(
        f"Summary: n={len(rewards)}, mean={np.mean(rewards):.4f}, median={np.median(rewards):.4f}, "
        f"min={np.min(rewards):.4f}, max={np.max(rewards):.4f}"
    )
    print(f"Discovered {len(components)} component fields and {len(metrics)} raw metric fields.")
    if label_counts:
        print(f"AgentBench labels: {label_counts[0]} passed, {label_counts[1]} failed.")
    generated = len(EXPECTED_FILES) if stratified else len(EXPECTED_FILES) - 1
    print(f"Generated {generated} plot files in {output}")


if __name__ == "__main__":
    main()
