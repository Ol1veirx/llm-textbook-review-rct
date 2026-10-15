"""Figures 5 and 6 of the paper."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

from .data import ARMS

COLORS = {"Control": "#F3A07F", "Experimental": "#8FD3C7"}


def _style():
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})


def fig5_boxplots(itt, path):
    """Total responses and precision by group."""
    _style()
    data = itt.assign(PrecisionPct=100 * itt["Precision"])
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    for ax, (col, label) in zip(axes, [("Responses", "Total responses"), ("PrecisionPct", "Precision (%)")]):
        sns.boxplot(data=data, x="Arm", y=col, order=list(ARMS), hue="Arm", palette=COLORS, legend=False,
                    width=.55, linewidth=.9, ax=ax,
                    flierprops=dict(marker="o", markersize=3, markerfacecolor="none", markeredgecolor="0.3"))
        ax.set_xlabel("")
        ax.set_ylabel(label)
    axes[0].set_title("(a) Total responses")
    axes[1].set_title("(b) Precision")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig6_scatter(itt, path):
    """Total responses against correct responses, with Spearman's rho."""
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    for ax, arm in zip(axes, ARMS):
        g = itt[itt["Arm"] == arm]
        x, y = g["Responses"].to_numpy(float), g["Correct"].to_numpy(float)
        rho, p = stats.spearmanr(x, y)
        ax.scatter(x, y, color=COLORS[arm], alpha=.85, edgecolors="none", s=34)
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, slope * xs + intercept, color=COLORS[arm], lw=1.5)
        ax.text(.05, .95, f"Spearman $\\rho$ = {rho:.2f}\n" + ("p < 0.001" if p < .001 else f"p = {p:.3f}"),
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=.3", fc="white", ec="0.7", alpha=.9))
        ax.set_title(f"{arm} group")
        ax.set_xlabel("Total responses")
        ax.set_ylabel("Correct responses")
        ax.grid(axis="y", ls="--", alpha=.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
