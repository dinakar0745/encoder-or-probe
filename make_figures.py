"""Paper figures from the four mitigation.csv files.

Expects results_nct_c1/, results_nct_c001/, results_pcam_c1/, results_pcam_c001/
(each containing mitigation.csv) in the current folder. Writes PNG and PDF to paper_figures/.

    python make_figures.py
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Fixed model order, colour and marker (colour-blind-checked palette; markers back up colour in print).
MODELS = {
    "phikon":    ("Phikon",     "#2a78d6", "o"),
    "phikon-v2": ("Phikon-v2",  "#eb6834", "s"),
    "kaiko-b16": ("Kaiko B/16", "#1baf7a", "^"),
    "dinov2-l":  ("DINOv2-L",   "#eda100", "D"),
}
LEVELS = {
    "blur": [0.5, 1, 2, 3, 3.5, 4, 5, 6],
    "jpeg": [70, 50, 30, 25, 20, 15, 10, 5],
    "stain": [0.1, 0.2, 0.3, 0.4, 0.5],
}
XLABEL = {"blur": "Gaussian blur σ (px)", "jpeg": "JPEG quality", "stain": "Stain shift α"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dc"

plt.rcParams.update({
    "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "figure.facecolor": "white", "savefig.facecolor": "white",
})


def load(folder: str) -> dict:
    """(model, probe, kind, level) -> (mean, sd), as percentages."""
    out = {}
    with open(Path(folder) / "mitigation.csv") as f:
        for r in csv.DictReader(f):
            out[(r["model"], r["probe"], r["kind"], float(r["level"]))] = (
                100 * float(r["balanced_accuracy"]), 100 * float(r["std"]))
    return out


def curve(data, model, probe, kind):
    pts = [data.get((model, probe, "clean", 0.0))] + [data.get((model, probe, kind, float(v))) for v in LEVELS[kind]]
    mean = np.array([p[0] if p else np.nan for p in pts])
    sd = np.array([p[1] if p else np.nan for p in pts])
    return mean, sd


def ticks(kind):
    return ["clean"] + [f"{v:g}" for v in LEVELS[kind]]


def style_axis(ax, kind, ylim=(0, 100), xlabel=True):
    ax.set_xticks(range(len(ticks(kind))), ticks(kind))
    ax.set_ylim(*ylim)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=2)
    if xlabel:
        ax.set_xlabel(XLABEL[kind])


def draw(ax, data, model, probe, kind, dashed=False, band=True):
    name, colour, marker = MODELS[model]
    mean, sd = curve(data, model, probe, kind)
    xs = np.arange(len(mean))
    ax.plot(xs, mean, color=colour, marker=marker, markersize=3.5, linewidth=1.5,
            linestyle=(0, (3, 2)) if dashed else "-", markeredgecolor="white", markeredgewidth=0.5)
    if band:
        ax.fill_between(xs, mean - sd, mean + sd, color=colour, alpha=0.15, linewidth=0)
    return mean


def label_ends(ax, ends, x):
    """Direct labels at the right edge, nudged apart so they do not overlap."""
    order = sorted(ends, key=lambda e: e[1])
    ys = []
    for _, y, _ in order:
        ys.append(y if not ys else max(y, ys[-1] + 6.5))
    for (text, _, _), y in zip(order, ys):
        ax.text(x, y, text, va="center", ha="left", fontsize=7, color=INK)


def save(fig, name):
    out = Path("paper_figures")
    out.mkdir(exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote paper_figures/{name}.png and .pdf")


def model_handles():
    return [plt.Line2D([], [], color=c, marker=m, markersize=4, linewidth=1.5, label=n) for n, c, m in MODELS.values()]


def fig1(nct):
    """Ranking flip under JPEG on NCT: clean probe (left) against mixed probe (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    for ax, probe, title in [(axes[0], "clean", "Probe trained on clean images"),
                             (axes[1], "aug_all", "Probe trained on clean + degraded images")]:
        ends = []
        for model in MODELS:
            mean = draw(ax, nct, model, probe, "jpeg")
            ends.append((MODELS[model][0], mean[-2], model))  # label at JPEG Q10
        style_axis(ax, "jpeg")
        ax.set_title(title, fontsize=8, loc="left", color=INK)
        ax.axvspan(5.5, 7.5, color="#f1f0ea", zorder=0, linewidth=0)
    axes[0].set_ylabel("Balanced accuracy (%)")
    axes[0].text(6.5, 3, "ranking\nreverses", ha="center", va="bottom", fontsize=7, color=MUTED)
    fig.legend(handles=model_handles(), loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    save(fig, "fig1_ranking_flip")


def fig2(nct, pcam):
    """Recovery grid: datasets (rows) by artifact (columns), clean probe dashed, mixed probe solid."""
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.3), sharey="row",
                             gridspec_kw={"width_ratios": [9, 9, 6]})
    for r, (data, dname, ylim) in enumerate([(nct, "NCT-CRC-HE (9 classes)", (0, 100)),
                                             (pcam, "PatchCamelyon (2 classes)", (40, 100))]):
        for c, kind in enumerate(["blur", "jpeg", "stain"]):
            ax = axes[r, c]
            for model in MODELS:
                draw(ax, data, model, "clean", kind, dashed=True, band=False)
                draw(ax, data, model, "aug_all", kind)
            style_axis(ax, kind, ylim=ylim, xlabel=(r == 1))
            if c == 0:
                ax.set_ylabel(f"{dname}\nBalanced accuracy (%)")
    if True:
        axes[1, 0].axhline(50, color=MUTED, linewidth=0.6, linestyle=":")
        axes[1, 0].text(0.1, 51, "chance", fontsize=6.5, color=MUTED, va="bottom")
    probes = [plt.Line2D([], [], color=INK, linewidth=1.5, linestyle=(0, (3, 2)), label="clean probe"),
              plt.Line2D([], [], color=INK, linewidth=1.5, label="mixed probe")]
    fig.legend(handles=model_handles() + probes, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout()
    save(fig, "fig2_recovery_grid")


def fig3(pcam_c1, pcam_c001):
    """Regularisation lever: clean-probe accuracy under JPEG on PatchCamelyon at two probe strengths."""
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.2), sharey=True)
    for ax, model in zip(axes, MODELS):
        name, colour, marker = MODELS[model]
        for data, dashed in [(pcam_c1, True), (pcam_c001, False)]:
            draw(ax, data, model, "clean", "jpeg", dashed=dashed)
        style_axis(ax, "jpeg", ylim=(40, 100))
        ax.set_xticks(range(0, 9, 2), ticks("jpeg")[::2])
        ax.axhline(50, color=MUTED, linewidth=0.6, linestyle=":")
        ax.set_title(name, fontsize=8, loc="left", color=INK)
    axes[0].set_ylabel("Balanced accuracy (%)")
    handles = [plt.Line2D([], [], color=INK, linewidth=1.5, linestyle=(0, (3, 2)), label="C = 1 (default)"),
               plt.Line2D([], [], color=INK, linewidth=1.5, label="C = 0.001 (strong)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    save(fig, "fig3_regularisation")


def main():
    nct_c1, pcam_c1, pcam_c001 = load("results_nct_c1"), load("results_pcam_c1"), load("results_pcam_c001")
    fig1(nct_c1)
    fig2(nct_c1, pcam_c1)
    fig3(pcam_c1, pcam_c001)


if __name__ == "__main__":
    main()
