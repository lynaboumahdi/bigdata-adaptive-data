import numpy as np
from pathlib import Path

class MmapLoader:

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)

        self.images = np.load(
            self.data_dir / "x_train.npy",
            mmap_mode="r"
        )

        self.labels = np.load(
            self.data_dir / "y_train.npy",
            mmap_mode="r"
        )

        self.num_images = len(self.images)

        print("Mmap Loader initialisé !")
        print("Nombre d'images : ",self.num_images)

    def load_batch(self, indices):
        x_batch = self.images[indices]
        y_batch = self.labels[indices]

        return x_batch, y_batch
