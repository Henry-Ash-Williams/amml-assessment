import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from model import ImagePredictor
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import utils.data as data
import wandb


def get_device():
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )

    return device


def get_args():
    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=64,
        help="Batch size used while training",
    )
    arg_parser.add_argument(
        "-e", "--epochs", type=int, default=20, help="Number of training epochs"
    )
    arg_parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate used while training"
    )
    arg_parser.add_argument(
        "-l",
        "--layers",
        type=int,
        default=3,
        help="Number of layers in the ConvLSTM model",
    )
    arg_parser.add_argument(
        "-k", "--kernels", type=int, default=64, help="Number of kernels in each layer"
    )
    arg_parser.add_argument(
        "-a",
        "--activation",
        type=str,
        choices=["tanh", "relu"],
        default="relu",
        help="Activation function used by the model",
    )

    arg_parser.add_argument(
        "-o",
        "--optimizer",
        type=str,
        default="Adam",
        help="Optimisation algoritm used during training",
    )

    arg_parser.add_argument(
        "-w",
        "--weight",
        type=str,
        default=None,
        help="Weighting of noise in output",
    )
    arg_parser.add_argument(
        "-t",
        "--threshold",
        type=str,
        default=None,
        help="Threshold of image to which noise is applied",
    )
    return arg_parser.parse_args()


def create_model(args, device):
    model = ImagePredictor(
        in_channels=1,
        num_kernels=args.kernels,
        num_layers=args.layers,
        kernel_size=(3, 3),
        padding=(1, 1),
        activation=args.activation,
        image_size=(36, 36),
        device=device,
    ).to(device)
    return model


def get_datasets(args, device):

    def create_dataset(data, idxs):
        dataset = []

        for sequence in data:
            for idx in idxs:
                dataset.append(sequence[idx])

        dataset = np.array(dataset)
        return dataset[:, :-1], dataset[:, -1]

    ds = data.Datasets("/Users/henrywilliams/Documents/uni/amml/assessment/data")
    n6 = ds.n6_full().numpy()

    n6 = (n6 - n6.min()) / (n6.max() - n6.min())
    idxs = np.array([[i + j for j in range(3)] for i in range(14)])

    train_mask = np.isin(idxs, data.TEST_INDICIES)
    test_mask = np.isin(idxs[:, -1], data.TEST_INDICIES)
    train_idxs = idxs[~train_mask.any(axis=1)]
    test_idxs = idxs[test_mask]

    train_X, train_y = create_dataset(n6, train_idxs)
    test_X, test_y = create_dataset(n6, test_idxs)

    train_y = train_y.reshape(-1, 1, 36, 36, 1)
    test_y = test_y.reshape(-1, 1, 36, 36, 1)

    train_X = torch.tensor(train_X, dtype=torch.float32, device=device).permute(
        0, 4, 1, 2, 3
    )
    train_y = torch.tensor(train_y, dtype=torch.float32, device=device).permute(
        0, 4, 1, 2, 3
    )
    test_X = torch.tensor(test_X, dtype=torch.float32, device=device).permute(
        0, 4, 1, 2, 3
    )
    test_y = torch.tensor(test_y, dtype=torch.float32, device=device).permute(
        0, 4, 1, 2, 3
    )

    train = TensorDataset(train_X, train_y)
    test = TensorDataset(test_X, test_y)

    train_loader = DataLoader(train, args.batch_size, shuffle=True)
    test_loader = DataLoader(test, args.batch_size, shuffle=True)

    return train_loader, test_loader


def get_optimiser(name):
    return getattr(torch.optim, name)


def train(model, train, test, args):
    optim = get_optimiser(args.optimizer)(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    for epoch in range(args.epochs):
        model.train()
        torch.mps.empty_cache()
        for input, target in tqdm(
            train,
            desc=f"Train epoch {epoch + 1}/{args.epochs}",
            leave=False,
        ):
            optim.zero_grad()
            output = model(
                input, threshold=args.threshold, weight=args.weight
            ).unsqueeze(1)
            loss = criterion(output, target)
            loss.backward()
            optim.step()
            wandb.log({"train_loss": loss.item()})

        test_loss = 0
        model.eval()
        for input, target in tqdm(
            test,
            desc=f"Test epoch {epoch + 1}/{args.epochs}",
            leave=False,
        ):
            with torch.no_grad():
                output = model(
                    input, threshold=args.threshold, weight=args.weight
                ).unsqueeze(1)
            loss = criterion(output, target)
            test_loss += loss.item()
        test_loss /= len(test.dataset)
        wandb.log({"test_loss": test_loss})


def generate_image(model, test_loader):
    fig, axs = plt.subplots(4, 4, dpi=300)

    test = np.random.choice(len(test_loader.dataset), 16)
    test_X, test_y = test_loader.dataset[test]

    test_y_hat = model(test_X.reshape(16, 1, 2, 36, 36))
    diff = (test_y.reshape(16, 1, 36, 36) - test_y_hat).abs()

    for i, ax in enumerate(axs.flatten()):
        ax.imshow(diff[i].cpu().detach().reshape(36, 36))
        ax.set_xticks([])
        ax.set_yticks([])

    wandb.log({"test_images": fig})
    plt.close(fig)


def main(args, device):
    model = create_model(args, device)
    train_loader, test_loader = get_datasets(args, device)
    train(model, train_loader, test_loader, args)
    generate_image(model, test_loader)


if __name__ == "__main__":
    wandb.login()
    args = get_args()
    device = get_device()
    run = wandb.init(project="ConvLSTM", config=args.__dict__)

    try:
        main(args, device)
    except Exception as e:
        print(e)
        wandb.finish(1)
        exit(1)
    wandb.finish()
