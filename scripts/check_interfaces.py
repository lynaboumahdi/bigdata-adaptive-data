"""
Vérifie que les loaders ont la même interface et se comportent pareil.

Lancer depuis la racine du projet :
    python scripts/check_interfaces.py

Pour chaque loader on teste : formes, types (dtype), valeurs comparées aux
données d'origine, ordre des indices (désordonnés + doublons), et ce qui se
passe avec un indice hors limites ou négatif.
"""
import contextlib
import io
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

LOADERS = {
    "csv": ("loaders.csv_loader", "CSVLoader", "data/canonical/fashion_mnist.csv"),
    "parquet": ("loaders.parquet_loader", "ParquetLoader", "data/parquet"),
    "binary": ("loaders.binary_loader", "BinaryLoader", "data/binary_shard"),
    "binary_fixed": ("loaders.binary_loader_fixed", "BinaryLoader", "data/binary_shard"),
    "mmap": ("loaders.mmap_loader", "MmapLoader", "data/mmap"),
}

ref_x = np.load("data/raw/x_train.npy")
ref_y = np.load("data/raw/y_train.npy")

INDICES = [5, 59999, 6000, 5, 5999, 14500, 6001]


def outcome(loader, idx):
    """Renvoie 'erreur : NomDeLException' ou 'accepté'."""
    try:
        loader.load_batch(idx)
        return "accepté"
    except Exception as e:  # noqa: BLE001
        return f"erreur : {type(e).__name__}"


rows = []
for name, (module_name, class_name, path) in LOADERS.items():
    row = {"loader": name}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            module = __import__(module_name, fromlist=[class_name])
            loader = getattr(module, class_name)(path)
    except Exception as e:  # noqa: BLE001
        row["résultat"] = f"IMPOSSIBLE À CRÉER ({type(e).__name__}: {e})"
        rows.append(row)
        continue

    with contextlib.redirect_stdout(io.StringIO()):
        x_list, y_list = loader.load_batch(INDICES)            # liste Python
        x_arr, y_arr = loader.load_batch(np.array(INDICES))    # tableau numpy

    row["forme images"] = str(x_list.shape)
    row["forme labels"] = str(y_list.shape)
    row["type images"] = str(x_list.dtype)
    row["type labels"] = str(y_list.dtype)
    row["valeurs OK"] = bool(
        np.array_equal(x_list, ref_x[INDICES]) and np.array_equal(y_list, ref_y[INDICES])
    )
    row["liste = tableau"] = bool(np.array_equal(x_list, x_arr) and np.array_equal(y_list, y_arr))
    row["indice 60000"] = outcome(loader, [60000])
    row["indice -1"] = outcome(loader, [-1])
    row["attribut labels"] = hasattr(loader, "labels") if name in ("csv", "parquet", "mmap") else "-"
    rows.append(row)

columns = ["loader", "forme images", "forme labels", "type images", "type labels",
           "valeurs OK", "liste = tableau", "indice 60000", "indice -1", "attribut labels"]
widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in columns}
print(" | ".join(c.ljust(widths[c]) for c in columns))
print("-+-".join("-" * widths[c] for c in columns))
for r in rows:
    if "résultat" in r:
        print(f"{r['loader']} : {r['résultat']}")
    else:
        print(" | ".join(str(r.get(c, "")).ljust(widths[c]) for c in columns))

problems = []
good = [r for r in rows if "résultat" not in r]
if len({r["type images"] for r in good}) > 1:
    problems.append("Les loaders ne renvoient pas le même type d'images (dtype).")
if len({r["type labels"] for r in good}) > 1:
    problems.append("Les loaders ne renvoient pas le même type de labels (dtype).")
if any(not r["valeurs OK"] for r in good):
    problems.append("Au moins un loader renvoie des valeurs différentes des données d'origine.")
if len({r["indice 60000"] for r in good}) > 1 or len({r["indice -1"] for r in good}) > 1:
    problems.append("Les loaders réagissent différemment aux indices invalides.")
if len(good) != len(rows):
    problems.append("Au moins un loader n'a pas pu être créé.")

print()
if problems:
    print("Points à regarder :")
    for p in problems:
        print(" -", p)
else:
    print("Aucune différence d'interface détectée.")