"""Static figures for REPORT.md: cost-vs-quality Pareto charts and a grid of embedding thumbnails.

Colours follow the dataviz reference palette (light mode). Colour encodes the algorithm
family, the marker encodes the library, and every point carries a direct name label, so
identity never relies on colour alone.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from drbench.report.tables import Summary

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
FRONTIER = "#9a9893"
CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
FAMILY_ORDER = ("t-sne", "umap", "ncvis", "pacmap", "pca")
LIBRARY_MARKERS = {"openTSNE": "o", "manifolds-rs": "s", "umap-learn": "^", "ncvis": "D",
                   "scikit-learn": "P"}
MARKER_SIZE = 64
LABEL_ABOVE = (6, 5)
LABEL_BELOW = (6, -12)
LABEL_CLEARANCE = (0.18, 0.05)  # axes-fraction box within which two labels would collide
THUMBNAIL_POINT_SIZE = 0.3
THUMBNAIL_MAX_POINTS = 30_000
DPI = 160

PANELS = (
    ("fit_s", "local", "Fit time (s, log)", "Local score (mean trustworthiness)", True),
    ("fit_s", "global", "Fit time (s, log)", "Global score (random-pair ρ)", True),
    ("peak_rss_mb", "local", "Peak RSS (MB)", "Local score (mean trustworthiness)", False),
)


def family_color(family: str) -> str:
    return CATEGORICAL[FAMILY_ORDER.index(family) % len(CATEGORICAL)]


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)


def pareto_front(costs: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Indices of points not dominated by any cheaper-or-equal point with a higher score, by cost."""
    front, best = [], -np.inf
    for index in np.argsort(costs):
        if scores[index] > best:
            front.append(index)
            best = scores[index]
    return np.array(front)


def label_offsets(ax, xs: np.ndarray, ys: np.ndarray) -> list[tuple[int, int]]:
    """Place each label above its point unless it would collide with an earlier label."""
    to_axes = ax.transAxes.inverted().transform
    points = to_axes(ax.transData.transform(np.column_stack([xs, ys])))
    placed: list[tuple[float, float]] = []
    offsets = []
    for x, y in points:
        clashes = any(abs(x - px) < LABEL_CLEARANCE[0] and abs(y - py) < LABEL_CLEARANCE[1]
                      for px, py in placed)
        offsets.append(LABEL_BELOW if clashes else LABEL_ABOVE)
        placed.append((x, y if not clashes else y - LABEL_CLEARANCE[1]))
    return offsets


def plot_pareto(summaries: list[Summary], dataset: str, path: Path) -> None:
    rows = [s for s in summaries if s.cost and s.score_mean]
    fig, axes = plt.subplots(1, len(PANELS), figsize=(16, 5), facecolor=SURFACE)
    for ax, (cost_field, score_field, x_label, y_label, log_x) in zip(axes, PANELS):
        style_axes(ax)
        costs = np.array([s.cost[cost_field] for s in rows])
        scores = np.array([s.score_mean[score_field] for s in rows])
        front = pareto_front(costs, scores)
        ax.step(costs[front], scores[front], where="post", color=FRONTIER, linewidth=1.2,
                linestyle="--", zorder=1)
        if log_x:
            ax.set_xscale("log")
        ax.scatter(costs, scores, s=0)
        ax.autoscale_view()  # label placement needs final data limits
        offsets = label_offsets(ax, costs, scores)
        for s, cost, score, offset in zip(rows, costs, scores, offsets):
            ax.errorbar(cost, score, yerr=s.score_sd[score_field], fmt="none",
                        ecolor=family_color(s.family), elinewidth=1, alpha=0.6, zorder=2)
            ax.scatter(cost, score, s=MARKER_SIZE, color=family_color(s.family),
                       marker=LIBRARY_MARKERS.get(s.library, "o"), edgecolors=SURFACE,
                       linewidths=2, zorder=3)
            ax.annotate(s.algorithm, (cost, score), textcoords="offset points", xytext=offset,
                        fontsize=8, color=TEXT_PRIMARY)
        ax.set_xlabel(x_label, color=TEXT_SECONDARY, fontsize=9)
        ax.set_ylabel(y_label, color=TEXT_SECONDARY, fontsize=9)
    families = [f for f in FAMILY_ORDER if any(s.family == f for s in rows)]
    libraries = sorted({s.library for s in rows})
    handles = [plt.Line2D([], [], color=family_color(f), marker="o", linestyle="", label=f) for f in families]
    handles += [plt.Line2D([], [], color=TEXT_SECONDARY, marker=LIBRARY_MARKERS.get(lib, "o"),
                           linestyle="", label=lib) for lib in libraries]
    handles.append(plt.Line2D([], [], color=FRONTIER, linestyle="--", label="Pareto front"))
    fig.suptitle(f"Cost vs quality — {dataset} (median cost, mean ± sd score over seeds)",
                 color=TEXT_PRIMARY, fontsize=11)
    fig.tight_layout()
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=len(handles),
               frameon=False, fontsize=8, labelcolor=TEXT_SECONDARY)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def plot_thumbnails(embeddings: dict[str, np.ndarray], labels: np.ndarray, title: str, path: Path) -> None:
    n_cols = 4
    n_rows = int(np.ceil(len(embeddings) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), facecolor=SURFACE)
    rng = np.random.default_rng(0)
    shown = rng.permutation(len(labels))[:THUMBNAIL_MAX_POINTS]
    colors = np.array(CATEGORICAL)[labels[shown] % len(CATEGORICAL)]
    for ax in np.ravel(axes):
        ax.set_axis_off()
    for ax, (name, Y) in zip(np.ravel(axes), embeddings.items()):
        ax.scatter(Y[shown, 0], Y[shown, 1], s=THUMBNAIL_POINT_SIZE, c=colors, linewidths=0,
                   rasterized=True)
        ax.set_title(name, color=TEXT_PRIMARY, fontsize=10)
        ax.set_aspect("equal", adjustable="datalim")
    fig.suptitle(title, color=TEXT_PRIMARY, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


CURVE_LINESTYLES = {"mrs-tsne-fft": "--", "umap-learn-legacy": "--"}
DEFAULT_POINT_RING = 150
CURVE_PANELS = (
    ("fit_s", "local", "Wall time (s, log)", "Local score (mean trustworthiness)"),
    ("fit_s", "global", "Wall time (s, log)", "Global score (random-pair ρ)"),
    ("cpu_s", "local", "CPU time (s, log)", "Local score (mean trustworthiness)"),
)


def plot_curves(points: list[dict], dataset: str, path: Path) -> None:
    """One line per algorithm across budgets; the ×1 (library default) point is ringed."""
    by_algorithm: dict[str, list[dict]] = {}
    for point in sorted(points, key=lambda p: p["budget"]):
        by_algorithm.setdefault(point["algorithm"], []).append(point)

    fig, axes = plt.subplots(1, len(CURVE_PANELS), figsize=(17, 5.5), facecolor=SURFACE)
    for ax, (cost_field, score_field, x_label, y_label) in zip(axes, CURVE_PANELS):
        style_axes(ax)
        ax.set_xscale("log")
        ends = []
        for name, line in by_algorithm.items():
            color = family_color(line[0]["family"])
            xs = np.array([p[cost_field] for p in line])
            ys = np.array([p[score_field] for p in line])
            ax.plot(xs, ys, color=color, linewidth=2, linestyle=CURVE_LINESTYLES.get(name, "-"),
                    marker=LIBRARY_MARKERS.get(line[0]["library"], "o"), markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=2)
            default = [p for p in line if p["budget"] == 1.0]
            if default:
                ax.scatter(default[0][cost_field], default[0][score_field], s=DEFAULT_POINT_RING,
                           facecolors="none", edgecolors=color, linewidths=1.5, zorder=3)
            ends.append((name, xs[-1], ys[-1]))
        ax.autoscale_view()
        offsets = label_offsets(ax, np.array([e[1] for e in ends]), np.array([e[2] for e in ends]))
        for (name, x, y), offset in zip(ends, offsets):
            ax.annotate(name, (x, y), textcoords="offset points", xytext=offset, fontsize=8,
                        color=TEXT_PRIMARY)
        ax.set_xlabel(x_label, color=TEXT_SECONDARY, fontsize=9)
        ax.set_ylabel(y_label, color=TEXT_SECONDARY, fontsize=9)

    families = [f for f in FAMILY_ORDER if any(p["family"] == f for p in points)]
    handles = [plt.Line2D([], [], color=family_color(f), linewidth=2, label=f) for f in families]
    handles += [plt.Line2D([], [], color=TEXT_SECONDARY, marker=LIBRARY_MARKERS.get(lib, "o"),
                           linestyle="", label=lib) for lib in sorted({p["library"] for p in points})]
    handles.append(plt.Line2D([], [], color=TEXT_SECONDARY, marker="o", markerfacecolor="none",
                              markersize=10, linestyle="", label="library default (×1)"))
    fig.suptitle(f"Quality vs time — {dataset} (epoch budgets ×⅛ … ×2 of each default, one seed)",
                 color=TEXT_PRIMARY, fontsize=11)
    fig.tight_layout()
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=len(handles),
               frameon=False, fontsize=8, labelcolor=TEXT_SECONDARY)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


# Projection quality map: local (trustworthiness) vs global (pair ρ), one panel per N, one point
# per algorithm. Colour slot = position in the sorted name list, so a colour follows its algorithm.
MAP_LABEL_GAP = 0.05  # minimum vertical gap between labels, axes fraction
MAP_LABEL_DX = 7


def algorithm_colors(names: list[str]) -> dict[str, str]:
    return {name: CATEGORICAL[i % len(CATEGORICAL)] for i, name in enumerate(sorted(names))}


def spread_positions(desired: list[float], gap: float) -> list[float]:
    """Positions (ascending input) at least `gap` apart, each colliding group centred on the
    mean of its desired positions, so labels move as little as possible."""
    groups: list[list[float]] = []  # each: desired positions of one group
    def placed(group: list[float]) -> list[float]:
        start = float(np.mean(group)) - gap * (len(group) - 1) / 2
        return [start + gap * i for i in range(len(group))]
    for y in desired:
        groups.append([y])
        while len(groups) > 1 and placed(groups[-2])[-1] + gap > placed(groups[-1])[0]:
            groups[-2:] = [groups[-2] + groups[-1]]
    return [y for group in groups for y in placed(group)]


def spread_labels(ax, labelled: list[tuple[str, float, float]]) -> None:
    """Labels right of their points, spread vertically so none overlap."""
    to_axes = ax.transAxes.inverted().transform
    placed = sorted(
        ((name, x, y, to_axes(ax.transData.transform((x, y)))[1]) for name, x, y in labelled),
        key=lambda item: item[3],
    )
    targets = spread_positions([item[3] for item in placed], MAP_LABEL_GAP)
    for (name, x, y, y_axes), target in zip(placed, targets):
        dy_points = (target - y_axes) * ax.bbox.height * 72 / ax.figure.dpi
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(MAP_LABEL_DX, dy_points),
                    va="center", fontsize=8, color=TEXT_PRIMARY)


def plot_quality_map(points, dataset: str, path: Path) -> None:
    """Local vs global score per algorithm, one panel per N, mean ± sd over seeds.
    Up and right is better on both axes."""
    names = sorted({p.algorithm for p in points})
    colors = algorithm_colors(names)
    sizes = sorted({p.n for p in points})
    fig, axes = plt.subplots(1, len(sizes), figsize=(5.8 * len(sizes), 5.2), facecolor=SURFACE,
                             sharex=True, sharey=True)
    for ax, n in zip(np.atleast_1d(axes), sizes):
        style_axes(ax)
        labelled = []
        for p in (p for p in points if p.n == n):
            x, y = p.mean["global"], p.mean["local"]
            ax.errorbar(x, y, xerr=p.sd["global"], yerr=p.sd["local"], fmt="none",
                        ecolor=colors[p.algorithm], elinewidth=1, alpha=0.6, zorder=2)
            ax.scatter(x, y, s=MARKER_SIZE, color=colors[p.algorithm],
                       marker=LIBRARY_MARKERS.get(p.library, "o"), edgecolors=SURFACE,
                       linewidths=2, zorder=3)
            labelled.append((p.algorithm, x, y))
        ax.set_title(f"N = {n:,}", color=TEXT_PRIMARY, fontsize=10)
        ax.set_xlabel("Global: random-pair Spearman ρ  →", color=TEXT_SECONDARY, fontsize=9)
        ax.autoscale_view()
        spread_labels(ax, labelled)
    np.atleast_1d(axes)[0].set_ylabel("Local: mean trustworthiness  →", color=TEXT_SECONDARY,
                                      fontsize=9)
    right = max(p.mean["global"] for p in points)
    for ax in np.atleast_1d(axes):
        ax.set_xlim(right=right + 0.09)  # room for labels right of the points
    handles = [plt.Line2D([], [], color=TEXT_SECONDARY, marker=LIBRARY_MARKERS.get(lib, "o"),
                          linestyle="", label=lib)
               for lib in sorted({p.library for p in points})]
    fig.suptitle(f"Projection quality — {dataset} (mean ± sd over seeds; up and right is better)",
                 color=TEXT_PRIMARY, fontsize=11)
    fig.tight_layout()
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=len(handles),
               frameon=False, fontsize=8, labelcolor=TEXT_SECONDARY)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
