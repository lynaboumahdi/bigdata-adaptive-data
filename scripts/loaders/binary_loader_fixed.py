import numpy as np
from pathlib import Path


class BinaryLoader:
    """Loader de shards binaires.

    Différence avec la première version : load_batch ne relit plus des
    shards entiers (4,5 Mo chacun) pour récupérer quelques images. Chaque
    image fait 784 octets et chaque label 1 octet, donc on calcule leur
    position dans le fichier et on ne lit que ce qui est demandé.
    Les images consécutives d'un même shard sont lues en une seule fois.
    """

    IMAGE_BYTES = 28 * 28

    def __init__(self, data_dir, shard_size=6000):
        self.data_dir = Path(data_dir)
        self.shard_size = shard_size
        self.num_images = 60000

        print("Binary Loader initialisé !")
        print("Nombre d'images : ", self.num_images)
        print("Taille du shard : ", self.shard_size)

    def get_shard_info(self, index):
        shard_index = index // self.shard_size
        local_index = index % self.shard_size
        return shard_index, local_index

    def _image_path(self, shard_index):
        return self.data_dir / f"train_images_{shard_index:03d}.bin"

    def _label_path(self, shard_index):
        return self.data_dir / f"train_labels_{shard_index:03d}.bin"

    def load_shard(self, shard_index):
        """Charge un shard entier (utile pour valider, pas pour les batchs)."""
        images = np.fromfile(self._image_path(shard_index), dtype=np.uint8)
        labels = np.fromfile(self._label_path(shard_index), dtype=np.uint8)
        images = images.reshape(-1, 28, 28)
        return images, labels

    def load_batch(self, indices):
        indices = np.asarray(indices, dtype=np.int64)

        if indices.size and (indices.min() < 0 or indices.max() >= self.num_images):
            raise IndexError("Indice hors limites (0 à %d)" % (self.num_images - 1))

        x_batch = np.empty((len(indices), 28, 28), dtype=np.uint8)
        y_batch = np.empty(len(indices), dtype=np.uint8)

        shard_ids = indices // self.shard_size
        local_ids = indices % self.shard_size

        for shard_index in np.unique(shard_ids):
            # positions (dans le batch demandé) des images de ce shard
            positions = np.nonzero(shard_ids == shard_index)[0]
            # on lit dans l'ordre du fichier, on remet dans l'ordre demandé ensuite
            order = np.argsort(local_ids[positions], kind="stable")
            positions = positions[order]
            locals_sorted = local_ids[positions]

            # coupe en séries d'images consécutives (une lecture par série)
            breaks = np.nonzero(np.diff(locals_sorted) != 1)[0] + 1
            starts = np.concatenate(([0], breaks))
            ends = np.concatenate((breaks, [len(locals_sorted)]))

            with open(self._image_path(shard_index), "rb") as fi, \
                 open(self._label_path(shard_index), "rb") as fl:
                for s, e in zip(starts, ends):
                    first = int(locals_sorted[s])
                    count = int(e - s)
                    out_positions = positions[s:e]

                    fi.seek(first * self.IMAGE_BYTES)
                    block = np.fromfile(fi, dtype=np.uint8,
                                        count=count * self.IMAGE_BYTES)
                    x_batch[out_positions] = block.reshape(count, 28, 28)

                    fl.seek(first)
                    y_batch[out_positions] = np.fromfile(fl, dtype=np.uint8,
                                                         count=count)

        return x_batch, y_batch