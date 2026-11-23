import numpy as np
import json
from pathlib import Path

RAW_DIR = Path("data/raw")
SHARD_DIR = Path("data/binary_shard")

original_images = np.load(RAW_DIR / "x_train.npy")
original_labels = np.load(RAW_DIR / "y_train.npy")

with open(SHARD_DIR / "manifest.json", "r", encoding="utf-8") as f: 
    manifest = json.load(f)

total_images = 0 
all_images_correct = True
all_labels_correct = True

for shard in manifest["shards"]:
    start = shard["start_index"]
    end = shard["end_index"]

    images = np.fromfile(
        SHARD_DIR / shard["image_file"],
        dtype = np.uint8 
    ).reshape(-1, 28, 28)

    labels = np.fromfile(
        SHARD_DIR / shard["label_file"],
        dtype=np.uint8
    )

    expected_images = original_images[start:end]
    expected_labels = original_labels[start:end]

    images_ok = np.array_equal(images, expected_images)
    labels_ok = np.array_equal(labels, expected_labels)

    all_images_correct &= images_ok
    all_labels_correct &= labels_ok

    total_images += len(images)

    print(
        f"Shard {shard['shard_id']} :"
        f"images={images_ok}, labels={labels_ok}"
    )

print("\n --- Résultat de validation ---")
print("Nombre total d'images : ",total_images)
print("Images identiques : ", all_images_correct)
print("Labels identiques : ", all_labels_correct)


if(
    total_images == len(original_images)
    and all_images_correct
    and all_labels_correct
):
    print("\nValidation réussie !")
else: 
    print("\nErreur : les données ne correspondent pas .")