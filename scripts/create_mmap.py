import numpy as np
from pathlib import Path
import shutil
import time

INPUT_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/mmap")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

start_time = time.perf_counter()

shutil.copy2(
    INPUT_DIR / "x_train.npy",
    OUTPUT_DIR / "x_train.npy"
)

shutil.copy2(
    INPUT_DIR / "y_train.npy",
    OUTPUT_DIR / "y_train.npy"
)

conversion_time = time.perf_counter() - start_time

print("copie des fichiers terminée ! ")
print(f"Temps de préparation : {conversion_time:.4f} secondes")

images = np.load(
    OUTPUT_DIR / "x_train.npy",
    mmap_mode="r"
)

labels = np.load(
    OUTPUT_DIR / "y_train.npy",
    mmap_mode="r"
)

print("\n--- Informations Mmap")
print("images :", images.shape)
print("labels:",labels.shape)
print("type des images : ",images.dtype)
print("type des labels : ",labels.dtype)
print("type de l'objet images : ",type(images))
