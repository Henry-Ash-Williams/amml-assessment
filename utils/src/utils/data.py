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

MU_N6 = 21.030820915316358
STD_N6 = 55.19628817097921

MU_N3 = 19.947500482253087
STD_N3 = 52.85921747418803


class FullSequenceDataset(Dataset):
    def __init__(
        self,
        train_path: PathLike,
        test_path: PathLike,
        n_sequences: int = N_SEQUENCES_N6,
        sequence_length: int = FULL_SEQUENCE_LENGTH,
        image_shape: Tuple[int, int, int] = IMAGE_SHAPE,
        transform: Optional[Callable] = None,
        mask_test: bool = False,
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
                self.sequences[:, i] = (
                    test[:, curr_test_idx]
                    if not mask_test
                    else np.zeros_like(test[:, curr_test_idx])
                ).reshape(self.sequences[:, i].shape)
                curr_test_idx += 1
            else:
                self.sequences[:, i] = train[:, curr_train_idx].reshape(
                    self.sequences[:, i].shape
                )
                curr_train_idx += 1

    def numpy(self):
        return np.array([sequence for sequence in self.sequences])

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
        """
        A dataset containing a sequence of images.

        Parameters:
        csv_file (PathLike): A path to the csv file containing the dataset
        sequence_length (int): The length of each image sequence
        image_shape (Tuple[int, int, int]): The resolution of each image
        transform (Callable): A set of transformations to apply to the images,
        defaults to the identity transformation.
        """
        self.images = pd.read_csv(csv_file, header=None)
        self.sequence_length = sequence_length
        self.image_shape = image_shape
        self.transform = transform

    def __len__(self):
        """
        Return the number of sequences contained within the dataset
        """
        n_images = self.images.shape[0]
        return n_images // self.sequence_length

    def __getitem__(self, idx):
        """
        Gets the sequence at a given index.
        """
        if not isinstance(idx, int):
            raise Exception("Unsupported key type!")

        if idx >= len(self):
            raise IndexError("Index out of range")

        offset = self.sequence_length * idx
        images = np.array(
            self.images.iloc[offset : offset + self.sequence_length, :]
        ).reshape((-1, *self.image_shape))
        images_buf = np.zeros((self.sequence_length, *self.image_shape))

        if self.transform:
            images = np.array([self.transform(image) for image in images])

        images_buf = images

        return images_buf

    def numpy(self) -> np.ndarray:
        return np.array([seq for seq in self])


class Datasets:
    def __init__(
        self,
        base_path: PathLike,
        transform: Optional[Callable] = None,
    ):
        self.base_path = base_path
        self.transform = transform

    def n6_train(self, image_shape=IMAGE_SHAPE):
        path = os.path.join(self.base_path, "n6-train.csv")

        return ImageSequenceDataset(
            path,
            TRAIN_SEQUENCE_LENGTH,
            image_shape,
            self.transform,
        )

    def n6_test(self, image_shape=IMAGE_SHAPE):
        path = os.path.join(self.base_path, "n6-test.csv")
        return ImageSequenceDataset(
            path,
            TEST_SEQUENCE_LENGTH,
            image_shape,
            self.transform,
        )

    def n3_train(self, image_shape=IMAGE_SHAPE):
        path = os.path.join(self.base_path, "n3-train.csv")
        return ImageSequenceDataset(
            path,
            TRAIN_SEQUENCE_LENGTH,
            image_shape,
            self.transform,
        )

    def n3_test(self, image_shape=IMAGE_SHAPE):
        path = os.path.join(self.base_path, "n3-test.csv")
        return ImageSequenceDataset(
            path,
            TEST_SEQUENCE_LENGTH,
            image_shape,
            self.transform,
        )

    def n6_full(self, **kwargs):
        train_path = os.path.join(self.base_path, "n6-train.csv")
        test_path = os.path.join(self.base_path, "n6-test.csv")
        return FullSequenceDataset(
            train_path, test_path, transform=self.transform, **kwargs
        )

    def n3_full(self, **kwargs):
        train_path = os.path.join(self.base_path, "n3-train.csv")
        test_path = os.path.join(self.base_path, "n3-test.csv")
        return FullSequenceDataset(
            train_path,
            test_path,
            transform=self.transform,
            n_sequences=N_SEQUENCES_N3,
            **kwargs,
        )
