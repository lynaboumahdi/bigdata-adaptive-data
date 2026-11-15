from loaders.mmap_loader import MmapLoader
import numpy as np

loader = MmapLoader("data/mmap")

indices = [5,100,200,500,1000]

x_batch, y_batch = loader.load_batch(indices)

print("\n ---Test Mmap Loader ---")

print("Dimensions des images :",x_batch.shape)
print("Dimensions des labels : ",y_batch.shape)
print("Labels :",y_batch)

print("Type des images :",type(loader.images))


original_images = np.load("data/raw/x_train.npy")
original_labels = np.load("data/raw/y_train.npy")

print(
    "Images identiques :",
    np.array_equal(x_batch, original_images[indices])
)
print(
    "Labels identiques :",
    np.array_equal(y_batch, original_labels[indices])
)