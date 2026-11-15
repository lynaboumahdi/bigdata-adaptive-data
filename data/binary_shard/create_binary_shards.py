import numpy as np
from pathlib import Path
import json
import time


INPUT_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/binary_shard")

SHARD_SIZE = 6000

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("chargement du dataset...")

images = np.load(INPUT_DIR / "x_train.npy")
labels = np.load(INPUT_DIR / "y_train.npy")

print("Images : ",images.shape)
print("labels : ",labels.shape)

assert len(images) == len(labels), "Images et labels incompatibles"

num_samples = len(images)
num_shards = (num_samples + SHARD_SIZE - 1 ) // SHARD_SIZE

print("Nombre d'images:", num_samples)
print("Nombre de shards :", num_shards)


start_time = time.perf_counter()

shards_info = []

for shard_id in range(num_shards):
    start = shard_id * SHARD_SIZE
    end = min(start + SHARD_SIZE, num_samples)

    shard_images = images[start:end]
    shard_labels = labels[start:end]

    image_file = OUTPUT_DIR / f"train_images_{shard_id:03d}.bin"
    label_file = OUTPUT_DIR / f"train_labels_{shard_id:03d}.bin"

    with open(image_file, "wb") as f: 
        f.write(shard_images.tobytes())

    with open(label_file, "wb") as f:
        f.write(shard_labels.tobytes())

    shards_info.append({
        "shard_id":shard_id,
        "start_index": start,
        "end_index": end,
        "num_samples":len(shard_images),
        "image_file":image_file.name,
        "label_file":label_file.name,
        "image_shape":list(shard_images.shape),
        "image_dtype":str(shard_images.dtype),
        "label_dtype":str(shard_labels.dtype)
    })
    print(f"Shard {shard_id +1}/{num_shards} créé : {len(shard_images)} images")
           
manifest = {
    "dataset" : "Fashion-MNIST",
    "split":"train",
    "total_samples":num_samples,
    "image_shape":list(images.shape[1:]),
    "image_dtype":str(images.dtype),
    "label_dtype":str(labels.dtype),
    "shard_size": SHARD_SIZE,
    "num_shards":num_shards,
    "shards" : shards_info
}

manifest_file = OUTPUT_DIR / "manifest.json"

with open(manifest_file,"w" , encoding="utf-8") as f:
    json.dump(manifest, f, indent=4)

conversion_time = time.perf_counter() - start_time
print("\nConversion terminée ! ")
print(f"Temps de conversion : {conversion_time:.4f} secondes")
print("Manifeste créé : ", manifest_file)
