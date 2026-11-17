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
                   "ncvis-plus": "v",
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
def algorithm_colors(names: list[str]) -> dict[str, str]:
    return {name: CATEGORICAL[i % len(CATEGORICAL)] for i, name in enumerate(sorted(names))}


def short_name(algorithm: str) -> str:
    return algorithm.removesuffix("-rinnd-default").removesuffix("-rinnd-fast")


# Label candidates around a point: unit direction, horizontal and vertical alignment. Each is
# tried at growing distances; beyond the first a leader line joins label and point.
LABEL_GAP_PT = 7
LABEL_DIRECTIONS = (
    ((1, 0), "left", "center"),
    ((-1, 0), "right", "center"),
    ((0, 1), "center", "bottom"),
    ((0, -1), "center", "top"),
    ((1, 1), "left", "bottom"),
    ((1, -1), "left", "top"),
    ((-1, 1), "right", "bottom"),
    ((-1, -1), "right", "top"),
)
LABEL_DISTANCES = (1, 2.5, 4, 6)  # multiples of LABEL_GAP_PT
LABEL_CANDIDATES = tuple(
    ((dx * LABEL_GAP_PT * reach, dy * LABEL_GAP_PT * reach), ha, va, reach)
    for reach in LABEL_DISTANCES for (dx, dy), ha, va in LABEL_DIRECTIONS
)
LEADER_PENALTY_PX2 = 40.0  # cost per distance step, so a label moves out only to clear overlap
LABEL_CLEARANCE_PX = 6  # minimum free space around every label
MARKER_HALF_PX = 9


def overlap_area(a, b) -> float:
    width = min(a.x1, b.x1) - max(a.x0, b.x0)
    height = min(a.y1, b.y1) - max(a.y0, b.y0)
    return max(width, 0.0) * max(height, 0.0)


def box_center(box) -> np.ndarray:
    return np.array([(box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2])


def place_labels(ax, labelled: list[tuple[str, float, float]]) -> None:
    """Direct labels: per point, the first candidate position whose padded text box is clear
    of the other labels, the other points and the axes edge, and sits nearer its own point
    than any other (unless a leader line joins them); else the least bad one. Labels moved out
    get a leader line."""
    from matplotlib.text import Text
    from matplotlib.transforms import Bbox

    renderer = ax.figure.canvas.get_renderer()
    frame = ax.get_window_extent(renderer)
    centers = {name: np.array(ax.transData.transform((x, y))) for name, x, y in labelled}
    markers = {name: Bbox.from_extents(*(c - MARKER_HALF_PX), *(c + MARKER_HALF_PX))
               for name, c in centers.items()}
    placed = []
    for name, x, y in sorted(labelled, key=lambda item: -item[2]):
        obstacles = placed + [box for other, box in markers.items() if other != name]
        others = [c for other, c in centers.items() if other != name]
        best, best_cost = None, np.inf
        for offset, ha, va, reach in LABEL_CANDIDATES:
            leader = ({"arrowstyle": "-", "color": FRONTIER, "linewidth": 0.7, "shrinkA": 2,
                       "shrinkB": MARKER_HALF_PX * 0.6} if reach > 1 else None)
            text = ax.annotate(name, (x, y), textcoords="offset points", xytext=offset, ha=ha,
                               va=va, fontsize=9, color=TEXT_PRIMARY, arrowprops=leader)
            # Text extent only (Annotation.get_window_extent would include the leader line);
            # update_positions first, since Text's extent alone ignores the annotation offset.
            text.update_positions(renderer)
            box = Text.get_window_extent(text, renderer).expanded(1, 1).padded(LABEL_CLEARANCE_PX)
            outside = box.width * box.height - overlap_area(box, frame)
            center = box_center(box)
            # Without a leader line, a label nearer another point than its own reads as that
            # point's label.
            misleading = reach == 1 and any(
                np.linalg.norm(center - c) < np.linalg.norm(center - centers[name]) for c in others)
            cost = (outside + sum(overlap_area(box, other) for other in obstacles)
                    + misleading * box.width * box.height + LEADER_PENALTY_PX2 * (reach - 1))
            if cost < best_cost:
                if best is not None:
                    best[0].remove()
                best, best_cost = (text, box), cost
            else:
                text.remove()
            if cost == 0:
                break
        placed.append(best[1])


LOCAL_OUTLIER_GAP = 0.1  # a lowest point this far below the next one is pinned to the bottom edge
Y_MARGIN = 0.15  # of the local span, above and below the data
OUTLIER_ROW = 0.1  # extra room below the data for pinned points, as a share of the local span


def local_outliers(points) -> set[tuple[str, int]]:
    """(algorithm, n) of points whose local score lies more than LOCAL_OUTLIER_GAP below every
    other point at that N (PCA here). Drawing them in place would squeeze all other methods into
    a thin band, so they are pinned to the bottom edge and labelled with their value."""
    outliers = set()
    for n in {p.n for p in points}:
        ranked = sorted((p for p in points if p.n == n), key=lambda p: p.mean["local"])
        if len(ranked) > 1 and ranked[1].mean["local"] - ranked[0].mean["local"] > LOCAL_OUTLIER_GAP:
            outliers.add((ranked[0].algorithm, n))
    return outliers


def plot_quality_map(points, dataset: str, path: Path) -> None:
    """Local vs global score (mean over seeds) per algorithm, one panel per N stacked
    vertically, each point labelled directly. Up and right is better on both axes. Local-score
    outliers (PCA) sit on a cut-off row at the bottom with their value in the label."""
    names = sorted({p.algorithm for p in points})
    colors = algorithm_colors(names)
    markers = {p.algorithm: LIBRARY_MARKERS.get(p.library, "o") for p in points}
    sizes = sorted({p.n for p in points})
    outliers = local_outliers(points)
    inliers = [p.mean["local"] for p in points if (p.algorithm, p.n) not in outliers]
    low, high = min(inliers), max(inliers)
    span = high - low
    pinned_y = low - (Y_MARGIN + OUTLIER_ROW / 2) * span
    cut_y = low - Y_MARGIN * span
    fig, axes = plt.subplots(len(sizes), 1, figsize=(8, 5.2 * len(sizes)), facecolor=SURFACE,
                             sharex=True, sharey=True)
    labelled: dict[int, list[tuple[str, float, float]]] = {}
    for ax, n in zip(np.atleast_1d(axes), sizes):
        style_axes(ax)
        labelled[n] = []
        for p in (p for p in points if p.n == n):
            pinned = (p.algorithm, n) in outliers
            y = pinned_y if pinned else p.mean["local"]
            ax.scatter(p.mean["global"], y, s=MARKER_SIZE * 1.6, color=colors[p.algorithm],
                       marker=markers[p.algorithm], edgecolors=SURFACE, linewidths=1.5, zorder=3)
            label = short_name(p.algorithm)
            labelled[n].append((f"{label} (local {p.mean['local']:.2f})" if pinned else label,
                                p.mean["global"], y))
        if outliers:
            ax.axhline(cut_y, color=FRONTIER, linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
        ax.set_title(f"N = {n:,}", color=TEXT_PRIMARY, fontsize=10, loc="left")
        ax.set_ylabel("Local (trustworthiness) →", color=TEXT_SECONDARY, fontsize=9)
        ax.set_xlabel("Global (random-pair Spearman ρ) →", color=TEXT_SECONDARY, fontsize=9)
        ax.tick_params(labelbottom=True)
        ax.margins(x=0.25)  # room for labels on either side of the outermost points
    np.atleast_1d(axes)[0].set_ylim(low - (Y_MARGIN + OUTLIER_ROW) * span if outliers
                                    else low - Y_MARGIN * span, high + Y_MARGIN * span)
    for ax in np.atleast_1d(axes):
        ax.set_yticks([t for t in ax.get_yticks() if t >= cut_y or not outliers])
    fig.suptitle(f"Projection quality — {dataset} (mean over 3 seeds, up and right is better)",
                 color=TEXT_PRIMARY, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    for ax, n in zip(np.atleast_1d(axes), sizes):
        place_labels(ax, labelled[n])
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
