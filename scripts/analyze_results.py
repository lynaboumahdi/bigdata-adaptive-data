"""
Résume les runs du benchmark et produit un tableau + des graphiques.

Lancer depuis la racine du projet, après avoir créé results/run1.csv,
results/run2.csv, results/run3.csv :
    python scripts/analyze_results.py

Produit :
    results/summary_median.csv   médiane des runs
    results/summary_range.csv    min - max des runs
    results/benchmark_charts.png graphiques pour le rapport
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path("results")
files = sorted(RESULTS.glob("run*.csv"))
if not files:
    sys.exit("Aucun fichier results/run*.csv trouvé. Lance d'abord le benchmark.")

frames = [pd.read_csv(f, index_col="loader") for f in files]
print(f"{len(files)} run(s) lu(s) : {[f.name for f in files]}")
stack = pd.concat(frames, keys=range(len(frames)), names=["run", "loader"])
by_loader = stack.groupby(level="loader", sort=False)
median = by_loader.median()
low = by_loader.min()
high = by_loader.max()

median.round(3).to_csv(RESULTS / "summary_median.csv", encoding="utf-8")
range_table = low.round(2).astype(str) + " – " + high.round(2).astype(str)
range_table.to_csv(RESULTS / "summary_range.csv", encoding="utf-8")

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 20)
print("\n=== Médiane des runs ===")
print(median.round(3).to_string())
print("\n=== Min – max des runs ===")
print(range_table.to_string())

# ---- graphiques -------------------------------------------------------
COLORS = {  # palette validée, ordre fixe : un loader = une couleur
    "csv": "#2a78d6",
    "parquet": "#eb6834",
    "binary": "#1baf7a",
    "binary_fixed": "#eda100",
    "mmap": "#e87ba4",
}
LABELS = {
    "csv": "CSV",
    "parquet": "Parquet",
    "binary": "Binary",
    "binary_fixed": "Binary corrigé",
    "mmap": "Mmap",
}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
loaders = list(median.index)

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
})


def fmt(v):
    return f"{v:.2f}" if v < 10 else f"{v:.0f}" if v >= 100 else f"{v:.1f}"


def grouped(ax, prefix, title):
    sizes = [32, 256, 1024]
    width = 0.8 / len(loaders)
    for i, name in enumerate(loaders):
        vals = [median.loc[name, f"{prefix} {s} (ms)"] for s in sizes]
        errs = [
            [median.loc[name, f"{prefix} {s} (ms)"] - low.loc[name, f"{prefix} {s} (ms)"] for s in sizes],
            [high.loc[name, f"{prefix} {s} (ms)"] - median.loc[name, f"{prefix} {s} (ms)"] for s in sizes],
        ]
        x = np.arange(len(sizes)) + (i - (len(loaders) - 1) / 2) * width
        ax.bar(x, vals, width * 0.9, color=COLORS[name], label=LABELS[name])
        ax.errorbar(x, vals, yerr=errs, fmt="none", ecolor=INK2, elinewidth=1, capsize=2)
        for xi, v, hi in zip(x, vals, [high.loc[name, f"{prefix} {s} (ms)"] for s in sizes]):
            ax.text(xi, hi * 1.2, fmt(v), ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_yscale("log")
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([f"{s} images" for s in sizes])
    ax.set_ylabel("Temps par batch (ms, échelle log)")
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def simple(ax, column, title, ylabel):
    vals = [median.loc[n, column] for n in loaders]
    ax.bar(range(len(loaders)), vals, 0.6, color=[COLORS[n] for n in loaders])
    errs = [[median.loc[n, column] - low.loc[n, column] for n in loaders],
            [high.loc[n, column] - median.loc[n, column] for n in loaders]]
    ax.errorbar(range(len(loaders)), vals, yerr=errs, fmt="none", ecolor=INK2, elinewidth=1, capsize=3)
    top = max(high[column].max(), 1e-9)
    for i, n in enumerate(loaders):
        ax.text(i, high.loc[n, column] + top * 0.03, fmt(vals[i]), ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_ylim(0, top * 1.15)
    ax.set_xticks(range(len(loaders)))
    ax.set_xticklabels([LABELS[n] for n in loaders], rotation=20)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


fig, axes = plt.subplots(2, 2, figsize=(13, 9), facecolor="#fcfcfb")
for a in axes.flat:
    a.set_facecolor("#fcfcfb")
simple(axes[0, 0], "init (s)", "Temps de démarrage", "secondes")
simple(axes[0, 1], "mémoire init (Mo)", "Mémoire ajoutée au démarrage", "Mo")
grouped(axes[1, 0], "aléatoire", "Batch aléatoire (images éparpillées)")
grouped(axes[1, 1], "séquentiel", "Batch séquentiel (images consécutives)")
handles, labels = axes[1, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=len(loaders), frameon=False)
fig.suptitle(
    f"Comparaison des 4 loaders (médiane de {len(files)} runs, barres d'erreur = min–max)",
    x=0.02, ha="left", fontsize=13, color=INK,
)
fig.tight_layout(rect=(0, 0.05, 1, 0.96))
out = RESULTS / "benchmark_charts.png"
fig.savefig(out, dpi=150)
print(f"\nGraphiques enregistrés dans {out}")