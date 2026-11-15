import numpy as np 
from pathlib import Path

files = sorted(Path("data/raw").glob("*.npy"))

for file in files : 
    data = np.load(file, mmap_mode="r")

    print(f"/nFichier : {file.name}")
    print(f"Dimensions : {data.shape}")
    print(f"Type : {data.dtype}")
    print(f"Taille : {file.stat().st_size / (1024**2):.2f} MB")