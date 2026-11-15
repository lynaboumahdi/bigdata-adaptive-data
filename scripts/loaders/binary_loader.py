import numpy as np
from pathlib import Path

class BinaryLoader:
    def __init__(self, data_dir, shard_size=6000):
        self.data_dir = Path(data_dir)
        self.shard_size = shard_size
        self.num_images = 60000

        print("Binary Loader initialisé !")
        print("Nombre d'images : ",self.num_images)
        print("Taille du shard : ",self.shard_size)

    def get_shard_info(self, index):
        shard_index = index // self.shard_size

        local_index = index % self.shard_size

        return shard_index,local_index

    def load_shard(self, shard_index):

        image_path = self.data_dir / f"train_images_{shard_index:03d}.bin"
        label_path = self.data_dir / f"train_labels_{shard_index:03d}.bin"

        images = np.fromfile(image_path, dtype=np.uint8)

        labels = np.fromfile(label_path, dtype=np.uint8)

        images = images.reshape(-1,28,28)

        return images, labels 

    def load_batch(self, indices):
        indices = np.asarray(indices, dtype=np.int64)

        x_batch = np.empty((len(indices),28,28), dtype=np.uint8)
        y_batch = np.empty(len(indices), dtype=np.uint8)

        loaded_shards = {}

        for output_position, index in enumerate(indices):
            shard_index, local_index = self.get_shard_info(int(index))
            if shard_index not in loaded_shards:
                loaded_shards[shard_index] = self.load_shard(shard_index)
            images,labels = loaded_shards[shard_index]

            x_batch[output_position]= images[local_index]
            y_batch[output_position] = labels[local_index]

        return x_batch,y_batch
