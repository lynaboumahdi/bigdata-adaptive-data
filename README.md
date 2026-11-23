# Big Data : chargement de Fashion-MNIST avec 4 formats de stockage

Ce projet stocke les 60 000 images d'entraînement de Fashion-MNIST (28 × 28 pixels, 784 pixels par image, un label par image) dans quatre formats, puis compare leurs performances de chargement :

| Format | Loader | Données lues |
|---|---|---|
| CSV | `CSVLoader` | `data/canonical/fashion_mnist.csv` |
| Parquet | `ParquetLoader` | dossier `data/parquet` (écrit par Spark) |
| Binary | `BinaryLoader` | 10 shards `.bin` dans `data/binary_shard` |
| Mmap | `MmapLoader` | fichiers `.npy` lus par mappage mémoire dans `data/mmap` |

Les quatre loaders ont la même interface : `load_batch(indices)` renvoie `(images, labels)`, avec des images de forme `(n, 28, 28)` et des labels de forme `(n,)`, tous deux en `uint8`.

## Installation

```
git clone https://github.com/lynaboumahdi/bigdata-adaptive-data.git
cd bigdata-adaptive-data
python -m venv venv
source venv/Scripts/activate        # Windows (Git Bash). Linux/Mac : source venv/bin/activate
pip install -r requirements.txt
```

Java 17 ou 21 est nécessaire uniquement pour régénérer le Parquet avec Spark. Sous Windows, Spark demande en plus `winutils.exe` (variable `HADOOP_HOME`). Le dossier `data/parquet` est déjà dans le dépôt, donc ce n'est pas nécessaire pour utiliser le projet.

## Préparer les données

Toutes les commandes se lancent depuis la racine du projet, dans cet ordre :

| Étape | Commande | Résultat |
|---|---|---|
| 1 | `python scripts/download_dataset.py` | `data/raw/*.npy` |
| 2 | `python scripts/inspect_dataset.py` | dimensions et types (60000 × 28 × 28, `uint8`) |
| 3 | `mkdir -p data/canonical` puis `python scripts/create_canonical.py` | `data/canonical/fashion_mnist.csv` |
| 4 | `python scripts/spark_parquet.py` | `data/parquet` (Spark) |
| 5 | `python scripts/create_mmap.py` | `data/mmap` |
| 6 | `python data/binary_shard/create_binary_shards.py` | shards `.bin` et `manifest.json` dans `data/binary_shard` |
| 7 | `python scripts/validate_binary_shards.py` | vérifie les shards contre les données d'origine |

Le CSV est suivi par Git LFS.

## Vérifier

```
python scripts/test_all_loaders.py      # les 4 loaders renvoient les mêmes images et labels
python scripts/check_interfaces.py      # formes, types, indices invalides
```

## Benchmark

```
python scripts/benchmark_loaders.py
cp results/benchmark_results.csv results/run1.csv
```

Refaire ces deux commandes pour `run2.csv` et `run3.csv`, puis :

```
python scripts/analyze_results.py
```

Le benchmark mesure chaque loader dans un processus séparé : temps de démarrage, mémoire ajoutée au démarrage, et temps moyen d'un `load_batch` sur 20 répétitions, avec 32, 256 et 1024 images, en indices aléatoires et en indices consécutifs. Les fichiers sont lus une fois avant la mesure : les résultats correspondent donc à des fichiers déjà en cache dans le système.

`analyze_results.py` écrit `results/summary_median.csv`, `results/summary_range.csv` et `results/benchmark_charts.png`.

## Résultats

Médiane de 3 runs, sous Windows. `binary` est le loader d'origine et `binary_fixed` la version corrigée (voir plus bas).

| Loader | Démarrage (s) | Mémoire au démarrage (Mo) | Aléatoire 32 (ms) | Aléatoire 256 (ms) | Aléatoire 1024 (ms) | Séquentiel 1024 (ms) |
|---|---|---|---|---|---|---|
| CSV | 2,57 | 145 | 0,03 | 0,13 | 0,36 | 0,20 |
| Parquet | 0,68 | 86 | 0,03 | 0,15 | 0,32 | 0,14 |
| Mmap | 0,003 | 0,1 | 0,12 | 0,57 | 0,92 | 0,14 |
| Binary (original) | 0 | 0 | 57,1 | 71,7 | 89,6 | 15,7 |
| Binary corrigé | 0 | 0 | 5,7 | 20,9 | 48,5 | 1,3 |

Taille sur disque : CSV 127 Mo, Binary 45 Mo, Mmap 45 Mo, Parquet 39 Mo, données brutes `.npy` 53 Mo.

### Lecture des résultats

- **CSV et Parquet** chargent tout en mémoire au démarrage, puis servent les batchs en moins d'une milliseconde. Parquet démarre environ 4 fois plus vite que CSV et utilise moins de mémoire.
- **Mmap** ne charge presque rien au démarrage et renvoie un batch de 1024 images en moins de 1 ms. C'est le meilleur compromis entre démarrage, mémoire et vitesse.
- **Binary** est le plus lent par batch. La version corrigée est environ 10 fois plus rapide que l'original à 32 images, 3 fois à 256 et 1,8 fois à 1024 : le gain se réduit quand le batch grossit.
- Les mesures varient d'un run à l'autre (par exemple le démarrage de CSV va de 2,3 à 5,5 s). Les écarts entre runs sont dans `results/summary_range.csv`.

### Limites

- Mesures faites sur une seule machine Windows, avec 3 runs et des fichiers en cache. Sans cache (premier accès au disque), Mmap et Binary seraient plus lents.
- La mémoire de Mmap ne compte que l'ouverture du fichier : les pages lues ensuite s'ajoutent.
- Le benchmark mesure uniquement `load_batch` et le démarrage. Le temps de création de chaque format n'est pas comparé.
- Le Parquet vient de Spark, mais les loaders le lisent avec pandas : Spark n'intervient pas dans le chargement.

## Corrections apportées pendant la vérification

- `validate_binary_shards.py` : chemins `dat/raw` et `binary_shards` corrigés en `data/raw` et `binary_shard`.
- `inspect_dataset.py` : `"/nFichier"` corrigé en `"\nFichier"`.
- `csv_loader.py` : l'attribut `label` renommé en `labels`, comme dans les autres loaders.
- `csv_loader.py` et `parquet_loader.py` : lecture en `uint8` au lieu de `int64` (CSV) et `int32` (Parquet), et images rangées ligne par ligne en mémoire. Les batchs sont environ dix fois plus rapides et la mémoire passe d'environ 500 Mo à moins de 150 Mo.
- `parquet_loader.py` : le DataFrame et la mémoire tampon de pyarrow sont libérés après le chargement.
- `scripts/loaders/binary_loader_fixed.py` : version de `BinaryLoader` qui ne lit que les images demandées au lieu de relire des shards entiers, avec une erreur `IndexError` pour les indices hors limites. Le loader d'origine reste dans `binary_loader.py` pour la comparaison.
- `scripts/benchmark_loaders.py`, `analyze_results.py` et `check_interfaces.py` : scripts de mesure et de vérification.

## Comportements à connaître

- Un indice négatif (par exemple `-1`) est accepté par CSV, Parquet et Mmap, qui renvoient alors l'image depuis la fin comme le fait numpy. Binary (version corrigée) le refuse avec `IndexError`.
- Un indice supérieur ou égal à 60 000 lève `IndexError` pour CSV, Parquet, Mmap et Binary corrigé, et `FileNotFoundError` pour le Binary d'origine.