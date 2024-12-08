import os
from os import PathLike
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

FULL_SEQUENCE_LENGTH = 16
TRAIN_SEQUENCE_LENGTH = 14
TEST_SEQUENCE_LENGTH = 2
N_SEQUENCES_N6 = 400
N_SEQUENCES_N3 = 100
IMAGE_SHAPE = (36, 36, 1)
TEST_INDICIES = [3, 15]


class FullSequenceDataset(Dataset):
    def __init__(
        self,
        train_path: PathLike,
        test_path: PathLike,
        n_sequences: int = N_SEQUENCES_N6,
        sequence_length: int = FULL_SEQUENCE_LENGTH,
        image_shape: Tuple[int, int, int] = IMAGE_SHAPE,
        transform: Optional[Callable] = None,
    ):
        assert (
            TRAIN_SEQUENCE_LENGTH + TEST_SEQUENCE_LENGTH == sequence_length
        ), f"Sum of train and test sequence length should be equal to {sequence_length}"

        self.image_shape = image_shape
        self.sequence_length = sequence_length

        train = ImageSequenceDataset(
            train_path, TRAIN_SEQUENCE_LENGTH, image_shape, transform
        ).numpy()

        test = ImageSequenceDataset(
            test_path, TEST_SEQUENCE_LENGTH, image_shape, transform
        ).numpy()

        self.sequences = np.zeros((n_sequences, sequence_length, *image_shape))

        curr_train_idx = 0
        curr_test_idx = 0
        for i in range(sequence_length):
            if i in TEST_INDICIES:
                self.sequences[:, i] = test[:, curr_test_idx]
                curr_test_idx += 1
            else:
                self.sequences[:, i] = train[:, curr_train_idx]
                curr_train_idx += 1

    def __len__(self):
        return self.sequences.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()

        if idx >= len(self):
            raise IndexError("Index out of range")

        images = self.sequences[idx]
        return images


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

        if self.transform:
            images = np.array([self.transform(image) for image in images])

        return images

    def numpy(self) -> np.ndarray:
        return np.array([seq for seq in self])


class Datasets:
    def __init__(self, base_path: PathLike, transform: Optional[Callable] = None):
        self.base_path = base_path
        self.transform = transform

    def n6_train(self):
        path = os.path.join(self.base_path, "n6-train.csv")
        return ImageSequenceDataset(
            path,
            TRAIN_SEQUENCE_LENGTH,
            IMAGE_SHAPE,
            self.transform,
        )

    def n6_test(self):
        path = os.path.join(self.base_path, "n6-test.csv")
        return ImageSequenceDataset(
            path,
            TEST_SEQUENCE_LENGTH,
            IMAGE_SHAPE,
            self.transform,
        )

    def n3_train(self):
        path = os.path.join(self.base_path, "n3-train.csv")
        return ImageSequenceDataset(
            path,
            TRAIN_SEQUENCE_LENGTH,
            IMAGE_SHAPE,
            self.transform,
        )

    def n3_test(self):
        path = os.path.join(self.base_path, "n3-test.csv")
        return ImageSequenceDataset(
            path,
            TEST_SEQUENCE_LENGTH,
            IMAGE_SHAPE,
            self.transform,
        )

    def n6_full(self):
        train_path = os.path.join(self.base_path, "n6-train.csv")
        test_path = os.path.join(self.base_path, "n6-test.csv")
        return FullSequenceDataset(train_path, test_path, transform=self.transform)

    def n3_full(self):
        train_path = os.path.join(self.base_path, "n3-train.csv")
        test_path = os.path.join(self.base_path, "n3-test.csv")
        return FullSequenceDataset(
            train_path, test_path, transform=self.transform, n_sequences=N_SEQUENCES_N3
        )
