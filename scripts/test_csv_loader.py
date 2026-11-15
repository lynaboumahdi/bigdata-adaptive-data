from loaders.csv_loader import CSVLoader

loader = CSVLoader("data/canonical/fashion_mnist.csv")

indices = list(range(100))

x_batch,y_batch = loader.load_batch(indices)

print("\n ---Test CSV Loader ---")
print("Dimensions des images : ",x_batch.shape)
print("Dimensions des labels : ",y_batch.shape)
print("premiere image : ",x_batch[0])
print("Label de la premiere image : ",y_batch[0])