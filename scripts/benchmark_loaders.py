"""
Benchmark des 4 loaders : CSV, Parquet, Binary, Mmap.

Lancer depuis la racine du projet :
    python scripts/benchmark_loaders.py

Chaque loader est mesuré dans un processus séparé, pour que la mémoire
mesurée ne soit pas faussée par les loaders précédents.

Mesures :
  - temps d'initialisation (chargement/ouverture des données)
  - mémoire ajoutée par l'initialisation (RSS, en Mo)
  - temps moyen d'un batch aléatoire (tailles 32, 256, 1024)
  - temps moyen d'un batch séquentiel (indices contigus)
"""
import csv
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import psutil

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

NUM_IMAGES = 60000
BATCH_SIZES = [32, 256, 1024]
REPEATS = 20
SEED = 42
RESULT_MARKER = "RESULT:"

LOADERS = {
    "csv": ("loaders.csv_loader", "CSVLoader", "data/canonical/fashion_mnist.csv"),
    "parquet": ("loaders.parquet_loader", "ParquetLoader", "data/parquet"),
    "binary": ("loaders.binary_loader", "BinaryLoader", "data/binary_shard"),
    "binary_fixed": ("loaders.binary_loader_fixed", "BinaryLoader", "data/binary_shard"),
    "mmap": ("loaders.mmap_loader", "MmapLoader", "data/mmap"),
}


def rss_mb():
    return psutil.Process().memory_info().rss / (1024 ** 2)


def prewarm(path):
    """Lit une fois tous les fichiers du loader pour les mettre dans le cache
    du système. Sans cela, le premier loader qui lit des fichiers payait la
    lecture sur le disque, et les suivants non : la comparaison était faussée.
    Les mesures sont donc faites "à chaud" (fichiers déjà en cache)."""
    p = Path(path)
    files = [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file()]
    for f in files:
        with open(f, "rb") as fh:
            while fh.read(8 * 1024 * 1024):
                pass


def run_one(name):
    """Mesure un seul loader et affiche une ligne RESULT:{json}."""
    module_name, class_name, path = LOADERS[name]
    module = __import__(module_name, fromlist=[class_name])
    loader_class = getattr(module, class_name)

    prewarm(path)
    mem_before = rss_mb()
    start = time.perf_counter()
    loader = loader_class(path)
    init_time = time.perf_counter() - start
    mem_added = rss_mb() - mem_before

    rng = np.random.default_rng(SEED)
    result = {
        "loader": name,
        "init_s": init_time,
        "init_mem_mb": mem_added,
        "random_ms": {},
        "sequential_ms": {},
    }

    for batch_size in BATCH_SIZES:
        # un premier appel non mesuré pour "chauffer" le cache
        loader.load_batch(rng.choice(NUM_IMAGES, batch_size, replace=False))

        random_times = []
        for _ in range(REPEATS):
            indices = rng.choice(NUM_IMAGES, batch_size, replace=False)
            start = time.perf_counter()
            loader.load_batch(indices)
            random_times.append(time.perf_counter() - start)

        sequential_times = []
        for _ in range(REPEATS):
            first = int(rng.integers(0, NUM_IMAGES - batch_size))
            indices = np.arange(first, first + batch_size)
            start = time.perf_counter()
            loader.load_batch(indices)
            sequential_times.append(time.perf_counter() - start)

        result["random_ms"][batch_size] = statistics.mean(random_times) * 1000
        result["sequential_ms"][batch_size] = statistics.mean(sequential_times) * 1000

    print(RESULT_MARKER + json.dumps(result))


def run_all():
    results = []
    for name in LOADERS:
        print(f"--- Mesure de {name} ...", flush=True)
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--one", name],
            capture_output=True,
            text=True,
        )
        line = next(
            (l for l in proc.stdout.splitlines() if l.startswith(RESULT_MARKER)),
            None,
        )
        if proc.returncode != 0 or line is None:
            print(f"ERREUR pour {name} :")
            print(proc.stderr[-2000:] or proc.stdout[-2000:])
            continue
        results.append(json.loads(line[len(RESULT_MARKER):]))
    return results


def show_and_save(results):
    header = ["loader", "init (s)", "mémoire init (Mo)"]
    for b in BATCH_SIZES:
        header.append(f"aléatoire {b} (ms)")
    for b in BATCH_SIZES:
        header.append(f"séquentiel {b} (ms)")

    rows = []
    for r in results:
        row = [r["loader"], f"{r['init_s']:.3f}", f"{r['init_mem_mb']:.1f}"]
        row += [f"{r['random_ms'][str(b)]:.2f}" for b in BATCH_SIZES]
        row += [f"{r['sequential_ms'][str(b)]:.2f}" for b in BATCH_SIZES]
        rows.append(row)

    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    print()
    for line in [header] + rows:
        print(" | ".join(str(x).ljust(w) for x, w in zip(line, widths)))

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "benchmark_results.csv"
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"\nRésultats enregistrés dans {out_file}")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--one":
        run_one(sys.argv[2])
    else:
        show_and_save(run_all())