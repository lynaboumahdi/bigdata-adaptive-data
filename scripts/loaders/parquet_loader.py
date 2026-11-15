import pandas as pd

class ParquetLoader:
    def __init__(self, file_path):
        self.data = pd.read_parquet(file_path)

        self.images = self.data.drop(columns=["label"]).to_numpy()
        self.labels = self.data["label"].to_numpy()
        
        print("Parquet Loader initialisé ! ")
        print("nombres d'images : ",len(self.images))

    def load_batch(self,indices):

        x_batch = self.images[indices]
        x_batch = x_batch.reshape(-1,28,28)
        y_batch = self.labels[indices]

        return x_batch,y_batch