from loaders.parquet_loader import ParquetLoader

loader = ParquetLoader("data/parquet")

indices = list(range(100))

x_batch , y_batch = loader.load_batch(indices)

print("\n --- Test Parquet Loader ---")
print("Dimensions des images : ",x_batch.shape)
print("Dimensions des labels : ",y_batch.shape)

print("label de la premiere image : ",y_batch[0])