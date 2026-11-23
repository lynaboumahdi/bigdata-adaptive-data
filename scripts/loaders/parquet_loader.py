import numpy as np
import pandas as pd
import pyarrow as pa


class ParquetLoader:
    def __init__(self, file_path):
        data = pd.read_parquet(file_path)

        # on garde seulement des tableaux uint8 (comme Binary et Mmap)
        self.images = np.ascontiguousarray(data.drop(columns=["label"]).to_numpy(dtype=np.uint8))
        self.labels = data["label"].to_numpy(dtype=np.uint8)

        # on libère le DataFrame et la mémoire tampon de pyarrow
        del data
        pa.default_memory_pool().release_unused()

        print("Parquet Loader initialisé ! ")
        print("nombres d'images : ", len(self.images))

    def load_batch(self, indices):

        x_batch = self.images[indices]
        x_batch = x_batch.reshape(-1, 28, 28)
        y_batch = self.labels[indices]

        return x_batch, y_batch