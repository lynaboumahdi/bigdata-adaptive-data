import pandas as pd
import numpy as np

class CSVLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = pd.read_csv(file_path, dtype=np.uint8)
        self.images = np.ascontiguousarray(self.data.drop(columns=["label"]).to_numpy())
        self.labels = self.data["label"].to_numpy()
        print("CSV Loader initialisé !")
        print("Nombre d'images : ",len(self.images))
    def load_batch(self, indices):
        x_batch = self.images[indices]
        x_batch = x_batch.reshape(-1,28,28)
        y_batch = self.labels[indices]
        return x_batch, y_batch