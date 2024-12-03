from os import PathLike
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ImageSequenceDataset(Dataset):
    def __init__(
        self,
        csv_file: PathLike,
        sequence_length: int,
        image_shape: Tuple[int, int, int],
        transform: Optional[Callable] = None,
    ):
        self.images = pd.read_csv(csv_file, header=None)
        self.sequence_length = sequence_length
        self.image_shape = image_shape

        if transform:
            self.transform = transform

    def __len__(self):
        n_images = self.images.shape[0]
        return n_images // self.sequence_length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()

        if idx >= len(self):
            raise IndexError("Index out of range")

        offset = self.sequence_length * idx
        images = np.array(self.images.iloc[offset : offset + self.sequence_length, :])
        images = images.reshape((self.sequence_length, *self.image_shape))
        return images
