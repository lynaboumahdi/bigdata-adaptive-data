from loaders.binary_loader import BinaryLoader

loader = BinaryLoader("data/binary_shard")

indices = [5,100,6500,14500,59999]

x_batch,y_batch = loader.load_batch(indices)

print("\n ---Test load_batch ---")

print("Dimensions des images : ",x_batch.shape)
print("Dimensions des labels : ",y_batch.shape)

print("Labels:",y_batch)