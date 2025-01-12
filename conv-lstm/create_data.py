from typing import Literal

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset

import utils.data as data

DEBUG = True

EXCLUDED_SEQUENCES = [
    12,
    33,
    41,
    44,
    54,
    64,
    91,
    99,
    103,
    109,
    174,
    188,
    197,
    203,
    205,
    221,
    228,
    229,
    252,
    294,
    333,
    351,
    354,
]


def get_datasets(
    denoise: bool = False,
    dset: Literal["n3", "n6"] = "n6",
    exclude_outliers: bool = False,
):

    def create_dataset(data, idxs):
        dataset = []

        for i, sequence in enumerate(data):
            if i in EXCLUDED_SEQUENCES and exclude_outliers:
                print(f"Skipping {i}")
                continue
            for idx in idxs:
                dataset.append(sequence[idx])

        dataset = np.array(dataset)
        return dataset[:, :-1], dataset[:, -1]

    def denoise_fn(img):
        img = np.uint8(img * 255)
        img = cv2.fastNlMeansDenoising(img, h=10)
        return img / 255.0

    ds = data.Datasets("/Users/henrywilliams/Documents/uni/amml/assessment/data")
    if dset == "n6":
        sequences = ds.n6_full().numpy()
    else:
        sequences = ds.n3_full().numpy()

    sequences = (sequences - sequences.min()) / (sequences.max() - sequences.min())
    idxs = np.array([[i + j for j in range(3)] for i in range(14)])

    train_mask = np.isin(idxs, data.TEST_INDICIES)
    test_mask = np.isin(idxs[:, -1], data.TEST_INDICIES)
    train_idxs = idxs[~train_mask.any(axis=1)]
    test_idxs = idxs[test_mask]

    train_X, train_y = create_dataset(sequences, train_idxs)
    test_X, test_y = create_dataset(sequences, test_idxs)

    if denoise:
        train_y = np.array([denoise_fn(y) for y in train_y])
        test_y = np.array([denoise_fn(y) for y in test_y])

    train_y = train_y.reshape(-1, 1, 36, 36, 1)
    test_y = test_y.reshape(-1, 1, 36, 36, 1)

    train_X = torch.tensor(train_X, dtype=torch.float32).permute(0, 4, 1, 2, 3)
    train_y = torch.tensor(train_y, dtype=torch.float32).permute(0, 4, 1, 2, 3)
    test_X = torch.tensor(test_X, dtype=torch.float32).permute(0, 4, 1, 2, 3)
    test_y = torch.tensor(test_y, dtype=torch.float32).permute(0, 4, 1, 2, 3)

    train = TensorDataset(train_X, train_y)
    test = TensorDataset(test_X, test_y)

    return train, test


def sample(dataset, title=None):
    idx = np.random.choice(len(dataset))
    x, y = dataset[idx]
    fig, axs = plt.subplots(1, 3)
    if title is not None:
        fig.suptitle(title)
    axs[0].imshow(x[:, 0].reshape(36, 36))
    axs[0].title.set_text("X_1")
    axs[1].imshow(x[:, 1].reshape(36, 36))
    axs[1].title.set_text("X_2")
    axs[2].imshow(y[0].reshape(36, 36))
    axs[2].title.set_text("Y")

    [ax.axis("off") for ax in axs]
    plt.show()


if __name__ == "__main__":
    print("N6")
    train_n6, test_n6 = get_datasets(denoise=False, dset="n6")
    print("N3")
    train_n3, test_n3 = get_datasets(denoise=False, dset="n3")
    print("N6 Denoised")
    train_n6_denoised, test_n6_denoised = get_datasets(denoise=True, dset="n6")
    print("N3 Denoised")
    train_n3_denoised, test_n3_denoised = get_datasets(denoise=True, dset="n3")

    if not DEBUG:
        torch.save(
            train_n6,
            "/Users/henrywilliams/Documents/uni/amml/assessment/conv-lstm/data/train-n6.pt",
        )
        torch.save(
            test_n6,
            "/Users/henrywilliams/Documents/uni/amml/assessment/conv-lstm/data/test-n6.pt",
        )
        torch.save(
            train_n3,
            "/Users/henrywilliams/Documents/uni/amml/assessment/conv-lstm/data/train-n3.pt",
        )
        torch.save(
            test_n3,
            "/Users/henrywilliams/Documents/uni/amml/assessment/conv-lstm/data/test-n3.pt",
        )
        torch.save(
            train_n6_denoised,
            "/Users/henrywilliams/Documents/uni/amml/assessment/conv-lstm/data/train-n6-denoised.pt",
        )
        torch.save(
            test_n6_denoised,
            "/Users/henrywilliams/Documents/uni/amml/assessment/conv-lstm/data/test-n6-denoised.pt",
        )
        torch.save(
            train_n3_denoised,
            "/Users/henrywilliams/Documents/uni/amml/assessment/conv-lstm/data/train-n3-denoised.pt",
        )
        torch.save(
            test_n3_denoised,
            "/Users/henrywilliams/Documents/uni/amml/assessment/conv-lstm/data/test-n3-denoised.pt",
        )
    else:
        sample(train_n6, title="Train N6")
        sample(test_n6, title="Test N6")
        sample(train_n3, title="Train N3")
        sample(test_n3, title="Test N3")
        sample(train_n6_denoised, title="Train N6 (Denoised)")
        sample(test_n6_denoised, title="Test N6 (Denoised)")
        sample(train_n3_denoised, title="Train N3 (Denoised)")
        sample(test_n3_denoised, title="Test N3 (Denoised)")

    train_n6, test_n6 = get_datasets(denoise=False, dset="n6", exclude_outliers=True)
    train_n3, test_n3 = get_datasets(denoise=False, dset="n3", exclude_outliers=True)
    train_n6_denoised, test_n6_denoised = get_datasets(
        denoise=True, dset="n6", exclude_outliers=True
    )
    train_n3_denoised, test_n3_denoised = get_datasets(
        denoise=True, dset="n3", exclude_outliers=True
    )
    if not DEBUG:
        torch.save(
            train_n6,
            "/Users/henrywilliams/Documents/uni/amml/assessment/conv-lstm/data/train-n6-no-outliers.pt",
        )
        torch.save(
            test_n6,
            "/Users/henrywilliams/Documents/uni/amml/assessment/conv-lstm/data/test-n6-no-outliers.pt",
        )
        torch.save(
            train_n3,
            "/Users/henrywilliams/Documents/uni/amml/assessment/conv-lstm/data/train-n3-no-outliers.pt",
        )
        torch.save(
            test_n3,
            "/Users/henrywilliams/Documents/uni/amml/assessment/conv-lstm/data/test-n3-no-outliers.pt",
        )
        torch.save(
            train_n6_denoised,
            "/Users/henrywilliams/Documents/uni/amml/assessment/conv-lstm/data/train-n6-denoised-no-outliers.pt",
        )
        torch.save(
            test_n6_denoised,
            "/Users/henrywilliams/Documents/uni/amml/assessment/conv-lstm/data/test-n6-denoised-no-outliers.pt",
        )
        torch.save(
            train_n3_denoised,
            "/Users/henrywilliams/Documents/uni/amml/assessment/conv-lstm/data/train-n3-denoised-no-outliers.pt",
        )
        torch.save(
            test_n3_denoised,
            "/Users/henrywilliams/Documents/uni/amml/assessment/conv-lstm/data/test-n3-denoised-no-outliers.pt",
        )
    else:
        sample(train_n6, title="Train N6")
        sample(test_n6, title="Test N6")
        sample(train_n3, title="Train N3")
        sample(test_n3, title="Test N3")
        sample(train_n6_denoised, title="Train N6 (Denoised)")
        sample(test_n6_denoised, title="Test N6 (Denoised)")
        sample(train_n3_denoised, title="Train N3 (Denoised)")
        sample(test_n3_denoised, title="Test N3 (Denoised)")
