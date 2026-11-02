"""IEEE-sized figure helpers (single column ~3.5 in, 300 dpi)."""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "figure.dpi": 150, "savefig.dpi": 300, "axes.grid": True, "grid.alpha": .3})
FIG_DIR = "outputs/figures"
COLS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]


def save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(f"{FIG_DIR}/{name}.png", bbox_inches="tight")
    fig.savefig(f"{FIG_DIR}/{name}.pdf", bbox_inches="tight")


def flow_diagram(steps, name, ncols=1, w=3.5, h=None, box_w=0.8):
    """Vertical/serpentine flow of labelled boxes joined by arrows."""
    n = len(steps)
    h = h or 0.42 * n + 0.2
    fig, ax = plt.subplots(figsize=(w, h)); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, n)
    for i, s in enumerate(steps):
        y = n - i - 0.5
        ax.add_patch(FancyBboxPatch((0.5 - box_w / 2, y - 0.3), box_w, 0.6, boxstyle="round,pad=0.01,rounding_size=0.04",
                                    fc="#EAF0F8", ec="#1F3A5F", lw=0.8))
        ax.text(0.5, y, s, ha="center", va="center", fontsize=7)
        if i < n - 1:
            ax.annotate("", xy=(0.5, y - 0.62), xytext=(0.5, y - 0.3),
                        arrowprops=dict(arrowstyle="-|>", color="#1F3A5F", lw=0.8))
    save(fig, name); return fig


def _box(ax, x, y, w, h, txt, fc="#EAF0F8", fs=6.5):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.005,rounding_size=0.02",
                                fc=fc, ec="#1F3A5F", lw=0.8))
    ax.text(x, y, txt, ha="center", va="center", fontsize=fs)


def _arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color="#1F3A5F", lw=0.8))


def architecture(name, info, fname):
    """Conceptual architecture of the best model, using the parameters actually used."""
    fig, ax = plt.subplots(figsize=(3.5, 2.3)); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    if name.startswith("MLP"):
        sizes = [info["n_features"]] + list(info["hidden"]) + [1]
        labels = [f"Input\n({sizes[0]})"] + [f"Dense {s}\nReLU+Drop" for s in sizes[1:-1]] + ["Output\n(1, sigmoid)"]
        n = len(labels); xs = np.linspace(0.11, 0.89, n)
        for x, l in zip(xs, labels):
            _box(ax, x, 0.5, 0.16, 0.42, l, fs=6)
        for a, b in zip(xs[:-1], xs[1:]):
            _arrow(ax, a + 0.08, 0.5, b - 0.08, 0.5)
        ax.text(0.5, 0.15, f"Adam, BCE (class-weighted), early stopping", ha="center", fontsize=6.5)
    elif name == "Logistic Regression":
        _box(ax, 0.15, 0.5, 0.26, 0.5, f"Input features\n({info['n_features']}, scaled)", fs=6)
        _box(ax, 0.5, 0.5, 0.24, 0.5, "Weighted sum\nw$^T$x + b", fs=6)
        _box(ax, 0.85, 0.5, 0.24, 0.5, "Sigmoid\nP(readmit)", fs=6)
        _arrow(ax, 0.28, 0.5, 0.38, 0.5); _arrow(ax, 0.62, 0.5, 0.73, 0.5)
        ax.text(0.5, 0.15, "Balanced class weights", ha="center", fontsize=6.5)
    elif name == "Random Forest":
        _box(ax, 0.13, 0.5, 0.2, 0.5, f"Input features\n({info['n_features']})")
        for i, y in enumerate([0.82, 0.62, 0.42, 0.22]):
            _box(ax, 0.5, y, 0.22, 0.13, f"Tree {i + 1}" if i < 3 else f"Tree {info['n_estimators']}", fs=6)
            _arrow(ax, 0.23, 0.5, 0.39, y); _arrow(ax, 0.61, y, 0.75, 0.5)
        _box(ax, 0.86, 0.5, 0.22, 0.5, "Average of\ntree votes\n$\rightarrow$ P(readmit)", fs=6)
        ax.text(0.5, 0.06, f"depth$\leq${info['max_depth']}, bootstrap samples", ha="center", fontsize=6.5)
    else:  # XGBoost
        _box(ax, 0.1, 0.5, 0.16, 0.5, f"Input\nfeatures\n({info['n_features']})", fs=6)
        xs = [0.32, 0.5, 0.68]
        for i, x in enumerate(xs):
            _box(ax, x, 0.5, 0.13, 0.4, f"Tree {i + 1}" if i < 2 else f"Tree {info['n_estimators']}", fs=6)
            _arrow(ax, 0.18, 0.5, x - 0.065, 0.5) if i == 0 else _arrow(ax, xs[i - 1] + 0.065, 0.5, x - 0.065, 0.5)
        ax.text(0.5, 0.78, "Sequential boosting: each tree fits residual errors", ha="center", fontsize=6.5)
        _box(ax, 0.9, 0.5, 0.15, 0.4, "Sum of\nleaf scores\n$\rightarrow$ sigmoid", fs=6)
        _arrow(ax, 0.745, 0.5, 0.825, 0.5)
        ax.text(0.5, 0.15, f"depth={info['max_depth']}, lr={info['learning_rate']}, best iter={info['best_iteration']}",
                ha="center", fontsize=6.5)
    save(fig, fname); return fig


def flow_grid(steps, name, ncols=4, w=7.0, h=1.9):
    """Serpentine two-row flow diagram (row 1 left->right, row 2 right->left) for full-width figures."""
    nrows = -(-len(steps) // ncols)
    fig, ax = plt.subplots(figsize=(w, h)); ax.axis("off"); ax.set_xlim(0, ncols); ax.set_ylim(0, nrows)
    bw, bh = 0.88, 0.6
    pos = []
    for i, s in enumerate(steps):
        r, c = divmod(i, ncols)
        cx = c + 0.5 if r % 2 == 0 else ncols - c - 0.5
        cy = nrows - r - 0.5
        pos.append((cx, cy))
        ax.add_patch(FancyBboxPatch((cx - bw / 2, cy - bh / 2), bw, bh, boxstyle="round,pad=0.01,rounding_size=0.05",
                                    fc="#EAF0F8", ec="#1F3A5F", lw=0.8))
        ax.text(cx, cy, f"{i + 1}. {s}", ha="center", va="center", fontsize=6.5)
    for (x0, y0), (x1, y1) in zip(pos[:-1], pos[1:]):
        if abs(y0 - y1) < 1e-9:
            d = 1 if x1 > x0 else -1
            ax.annotate("", xy=(x1 - d * bw / 2, y1), xytext=(x0 + d * bw / 2, y0), arrowprops=dict(arrowstyle="-|>", color="#1F3A5F", lw=0.9))
        else:
            ax.annotate("", xy=(x1, y1 + bh / 2), xytext=(x0, y0 - bh / 2), arrowprops=dict(arrowstyle="-|>", color="#1F3A5F", lw=0.9))
    save(fig, name); return fig
