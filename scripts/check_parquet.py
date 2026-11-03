
import pandas as pd

csv = pd.read_csv("data/canonical/fashion_mnist.csv")
parquet = pd.read_parquet("data/parquet")

# Uniformiser les types des colonnes
csv = csv.astype("int32")
parquet = parquet.astype("int32")

print("CSV :", csv.shape)
print("Parquet :", parquet.shape)
print("Données identiques :", csv.equals(parquet))
