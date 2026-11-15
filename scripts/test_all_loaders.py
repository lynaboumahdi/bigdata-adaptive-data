from loaders.csv_loader import CSVLoader
from loaders.parquet_loader import ParquetLoader
from loaders.binary_loader import BinaryLoader
from loaders.mmap_loader import MmapLoader
import numpy as np

csv_loader = CSVLoader("data/canonical/fashion_mnist.csv")
parquet_loader = ParquetLoader("data/parquet")
binary_loader = BinaryLoader("data/binary_shard")
mmap_loader = MmapLoader("data/mmap")

indices = [5,14500,6000,500,5999,59999]

csv_x , csv_y = csv_loader.load_batch(indices)

parquet_x , parquet_y = parquet_loader.load_batch(indices)

binary_x , binary_y = binary_loader.load_batch(indices)

mmap_x , mmap_y = mmap_loader.load_batch(indices)

print("\n --- Dimensions des quatre loaders ---")

print("CSV : ",csv_x.shape , csv_y.shape)
print("Parquet : ",parquet_x.shape, parquet_y.shape)
print("binary : ",binary_x.shape , binary_y.shape)
print("Mmap : ",mmap_x.shape, mmap_y.shape)


original_images = np.load("data/raw/x_train.npy")
original_labels = np.load("data/raw/y_train.npy")

expected_x = original_images[indices]
expected_y = original_labels[indices]

print("\n --- comparaison des images ---")

print("csv : ",np.array_equal(csv_x,expected_x))
print("parquet :",np.array_equal(parquet_x,expected_x))
print("binary :",np.array_equal(binary_x,expected_x))
print("mmap :",np.array_equal(mmap_x,expected_x))


print("\n --- Comparaison des labels ---")

print("csv  :",np.array_equal(csv_y,expected_y))
print("parquet :",np.array_equal(parquet_y,expected_y))
print("binary :", np.array_equal(binary_y,expected_y))
print("mmap :",np.array_equal(mmap_y,expected_y))

